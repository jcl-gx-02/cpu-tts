"""文本规范化 + 中英语种路由（供 vits_tts_server.py 使用）。

包含两部分：
1. normalize_domain_text()：电力调度/告警文本的保守规范化
   （HTML 实体、保护段号 I/II/III 段、数值单位等）；
2. resolve_voice_spans()：按语种把文本切成片段并选择对应音色，
   用于 voice="auto" 的「piper zh + piper en 双模型路由」。

实测依据见 docs/kokoro-v1.1-zh-cpu-notes.md：
  "&#x20;"       -> 被当正文读，且会误触发英文音色
  "相间距离I段"   -> 读成 "距离 i 段"（应为"一段"）
  "7.594A"      -> 丢掉安培单位（应为"七点五九四安"）
"""

import os
import re
from typing import List, Optional, Tuple

# 语种片段短于该值时并入相邻片段，避免 "GPU"/"F5"/"AC" 这类短词来回切音色
AUTO_MIN_SPAN_CHARS = int(os.getenv("AUTO_MIN_SPAN_CHARS", "4"))
# 领域规范化总开关（供 /health 展示）
DOMAIN_NORMALIZE = os.getenv("DOMAIN_NORMALIZE", "true").lower() not in ("0", "false", "no")

ZH_LANG = "z"
EN_LANG = "a"

# --------------------------------------------------------------------------
# 1. 领域规范化
# --------------------------------------------------------------------------
_HTML_ENTITIES = [
    ("&#x20;", " "), ("&#160;", " "), ("&nbsp;", " "),
    ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
    ("&apos;", "'"), ("&#39;", "'"),
]

# 常见符号 -> 中文读法（只在中文语境生效）
#   °C/°F 必须在 ° 之前处理；"%"/"‰" 需要把符号挪到数字前面；"#"/"№" 挪到数字后面
_SYMBOL_PREFIX = [  # 30% -> 百分之30
    ("%", "百分之"),
    ("‰", "千分之"),
]
_SYMBOL_SUFFIX = [  # #2 -> 2号
    ("#", "号"),
    ("№", "号"),
]
_SYMBOL_MAP = [  # 直接替换（长串在前）
    ("℃", "摄氏度"), ("°C", "摄氏度"), ("℉", "华氏度"), ("°F", "华氏度"),
    ("±", "正负"), ("×", "乘"), ("÷", "除"), ("·", "点"),
    ("≤", "小于等于"), ("≥", "大于等于"), ("≈", "约等于"), ("≠", "不等于"),
    ("→", "到"), ("←", "到"), ("↑", "上升"), ("↓", "下降"),
    ("~", "到"), ("～", "到"), ("°", "度"),
    ("μ", "微"), ("²", "平方"), ("³", "立方"), ("√", "根号"), ("∞", "无穷"),
    ("&", "和"), ("@", "艾特"),
]

# 罗马数字（保护段号 I/II/III、母线 Ⅰ母/Ⅱ母、第III类…）
_FULLWIDTH_ROMAN = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V",
                    "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X"}
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}
_ROMAN_TOKEN_RE = re.compile(r"(?<![A-Za-z])(第)?(I{1,3}|IV|VI{0,3}|IX|XI{0,3})(?![A-Za-z])")
# 单个 I / V 只有后面跟这些领域后缀时才当罗马数字，避免误伤字母（如 V相、I/O）
_ROMAN_SUFFIX_RE = re.compile(r"\s*(段|母|类|次|号|级|侧|线|回|路|组|套|部分|间隔|象限)")


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN_VALUES[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _roman_sub(match: "re.Match") -> str:
    """III段 -> 三段，II母 -> 二母，第III部分 -> 第三部分。"""
    if not _zh_dominant(match.string, match.start(), match.end(), window=16):
        return match.group(0)
    token = match.group(2)
    if len(token) == 1 and not _ROMAN_SUFFIX_RE.match(match.string, match.end()):
        return match.group(0)  # 单个 I/V：没有领域后缀就不动，交给字母名映射
    return (match.group(1) or "") + _int_to_cn(str(_roman_to_int(token)))

_UNIT_RULES = [
    (re.compile(r"(?<=\d)\s*kV", re.I), "千伏"),
    (re.compile(r"(?<=\d)\s*kA", re.I), "千安"),
    (re.compile(r"(?<=\d)\s*kWh", re.I), "千瓦时"),
    (re.compile(r"(?<=\d)\s*kW", re.I), "千瓦"),
    (re.compile(r"(?<=\d)\s*MVA", re.I), "兆伏安"),
    (re.compile(r"(?<=\d)\s*MVAr", re.I), "兆乏"),
    (re.compile(r"(?<=\d)\s*MW", re.I), "兆瓦"),
    (re.compile(r"(?<=\d)\s*km", re.I), "千米"),
    (re.compile(r"(?<=\d)\s*Hz", re.I), "赫兹"),
    (re.compile(r"(?<=\d)\s*mA", re.I), "毫安"),
    (re.compile(r"(?<=\d)\s*ms", re.I), "毫秒"),
    (re.compile(r"(?<=\d)\s*s(?![A-Za-z\u4e00-\u9fff])"), "秒"),
    (re.compile(r"(?<=\d)\s*V(?![A-Za-z])", re.I), "伏"),
    # 安培：仅当紧跟数字且后面不是字母/汉字（避免把 "A相" 的相别 A 改掉）
    (re.compile(r"(?<=\d)A(?![A-Za-z\u4e00-\u9fff])"), "安"),
    (re.compile(r"(?<=\d)\s*Ω"), "欧姆"),
]

# piper 中文模型的 number.fst 处理小数不可靠：实测 "1.359" 读成"in 三百五十九"，
# 而写成中文"一点三五九"就完全正确。电力量测值几乎都是小数，所以统一转成中文读法。
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千"]
_DECIMAL_RE = re.compile(r"(\d+)\.(\d+)")
# 字母数字混合 token（开关编号/设备编号），如 11F5、AC380V、F5
_ALNUM_TOKEN_RE = re.compile(r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*")
# 纯数字的长设备号后接这些词时也逐位读（1198号 -> 一一九八号；35号 仍读三十五号）
_DEVICE_NO_RE = re.compile(r"(\d{4,})(?=\s*(?:号|线|间隔|开关|刀闸|杆|塔|设备))")

# 大写字母/缩写 -> 中文近似读法（用相近的汉字让中文音色读出来，避免音色切换）
#   只在中文语境、且是「独立的大写串」时生效；真正的英文单词/句子不受影响
LETTER_NAMES_ZH = os.getenv("LETTER_NAMES_ZH", "true").lower() not in ("0", "false", "no")
LETTER_NAMES_FILE = os.getenv("LETTER_NAMES_FILE", "/app/letter_names.json")

_DEFAULT_LETTER_NAMES = {
    "A": "诶", "B": "必", "C": "西", "D": "迪", "E": "伊", "F": "艾弗",
    "G": "记", "H": "艾尺", "I": "爱", "J": "杰", "K": "开", "L": "艾勒",
    "M": "艾姆", "N": "恩", "O": "欧", "P": "皮", "Q": "丘", "R": "阿尔",
    "S": "艾丝", "T": "提", "U": "优", "V": "维", "W": "达布留",
    "X": "艾克斯", "Y": "歪", "Z": "兹德",
}


def _load_letter_names() -> dict:
    """字母名映射：优先读外部 JSON（便于按听感逐字调整），否则用内置默认值。"""
    names = dict(_DEFAULT_LETTER_NAMES)
    try:
        import json

        if LETTER_NAMES_FILE and os.path.exists(LETTER_NAMES_FILE):
            with open(LETTER_NAMES_FILE, encoding="utf-8") as f:
                override = json.load(f)
            names.update({str(k).upper(): str(v) for k, v in override.items()})
    except Exception:  # noqa: BLE001
        pass
    return names


_LETTER_NAMES = _load_letter_names()
# 独立的大写串（前后都不是字母），1~8 个字母
_UPPER_RUN_RE = re.compile(r"(?<![A-Za-z])[A-Z]{1,8}(?![A-Za-z])")


def _upper_run_to_chinese(match: "re.Match") -> str:
    """AC -> 诶西，GPU -> 记皮优，F -> 艾弗（交给中文音色读，音色不再跳动）。"""
    if not LETTER_NAMES_ZH:
        return match.group(0)
    token = match.group(0)
    if not _zh_dominant(match.string, match.start(), match.end()):
        return token  # 纯英文语境（例如全大写英文句子）保持原样，交给英文音色
    return "".join(_LETTER_NAMES.get(ch, ch) for ch in token)


def _split_alnum_token(match: "re.Match") -> str:
    """编号类字母数字混合 token：数字逐位读，字母交中文近似音。

    1198F6 -> 一一九八 艾弗 六；11F5 -> 一一 艾弗 五；AC380 -> AC 三百八十。

    开关/设备编号在调度术语里是逐位读的（"1198" 不能读成"一千一百九十八"），
    所以这里对数字段做逐位转换；带单位的大数值（110kV 等）不在此规则内，仍按数量读。
    """
    token = match.group(0)
    if not _zh_context(match.string, match.start(), match.end(), skip_alnum=True):
        return token
    parts = re.findall(r"[A-Za-z]+|\d+", token)
    return " ".join(
        "".join(_CN_DIGITS[int(d)] for d in p) if p.isdigit() else p for p in parts
    )


def _device_no_sub(match: "re.Match") -> str:
    """1198号 -> 一一九八号（长设备号逐位读）。"""
    if not _zh_dominant(match.string, match.start(), match.end(), window=16):
        return match.group(0)
    return "".join(_CN_DIGITS[int(d)] for d in match.group(1))


def _int_to_cn(num: str) -> str:
    """0~9999 的整数转中文读法（10 -> 十，105 -> 一百零五）。"""
    n = int(num)
    if n == 0:
        return "零"
    if n > 9999:
        return "".join(_CN_DIGITS[int(d)] for d in num)
    out, zero = "", False
    digits = str(n)
    ln = len(digits)
    for i, d in enumerate(digits):
        v = int(d)
        pos = ln - i - 1
        if v == 0:
            zero = True
            continue
        if zero and out:
            out += "零"
        zero = False
        out += _CN_DIGITS[v] + _CN_UNITS[pos]
    out = out.replace("一十", "十") if out.startswith("一十") else out
    return out


def _decimal_sub(match: "re.Match") -> str:
    int_part, frac = match.group(1), match.group(2)
    return _int_to_cn(int_part) + "点" + "".join(_CN_DIGITS[int(d)] for d in frac)


def _zh_context(text: str, start: int, end: int, window: int = 12, skip_alnum: bool = False) -> bool:
    """判断匹配位置是否处于中文语境（向左找最近的汉字/字母，找不到再看右边）。

    skip_alnum=True 时跳过字母/数字（用于「字母数字拆分」这类零宽匹配，
    否则匹配点紧邻的字母会把语境误判成英文）。
    """
    j = start - 1
    if skip_alnum:
        # 只跳过「与匹配点紧邻的同一 token 内的字母/数字」，之后按正常规则判断语境
        while j >= 0 and text[j].isalnum() and not _CJK.match(text[j]):
            j -= 1
    steps = 0
    while j >= 0 and steps < window:
        ch = text[j]
        if _CJK.match(ch):
            return True
        if _LATIN.match(ch):
            return False
        j -= 1
        steps += 1
    k = end
    if skip_alnum:
        while k < len(text) and text[k].isalnum() and not _CJK.match(text[k]):
            k += 1
    steps = 0
    while k < len(text) and steps < window:
        ch = text[k]
        if _CJK.match(ch):
            return True
        if _LATIN.match(ch):
            return False
        k += 1
        steps += 1
    # 左右都没有字母（例如 "2×3"、"45°" 这种纯数字/符号片段）：按中文语境处理
    return True


def _zh_dominant(text: str, start: int, end: int, window: int = 20) -> bool:
    """按周边窗口判断语境：窗口内汉字占优才算中文语境。

    用于「大写字母/缩写 -> 中文近似读法」：
      "一次设备 QF、QS 位置"  窗口内汉字多 -> 两个都转中文（QS 前面是 "、" 和 "F"，
                                逐字符判断会被误判成英文）
      "The SCADA system ..."  窗口内没有汉字 -> 保持原样，交给英文音色
    """
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    seg = text[lo:hi]
    cjk = len(_CJK.findall(seg))
    latin = len(_LATIN.findall(seg))
    # 只要有汉字、且不是被大量英文包围（如 "III母、IV母" 这种缩写密集的短句也算中文）
    return cjk > 0 and cjk * 4 >= latin


def _sub_zh_context(pattern: "re.Pattern", repl, text: str) -> str:
    """只对中文语境做替换，纯英文句子原样保留。"""

    def _f(match: "re.Match") -> str:
        if _zh_context(text, match.start(), match.end()):
            return repl(match) if callable(repl) else repl
        return match.group(0)

    return pattern.sub(_f, text)


def _sub_zh_context_skip_alnum(pattern: "re.Pattern", repl, text: str) -> str:
    """同上，但语境判断时跳过相邻的字母/数字（用于字母数字拆分）。"""

    def _f(match: "re.Match") -> str:
        if _zh_context(text, match.start(), match.end(), skip_alnum=True):
            return repl(match) if callable(repl) else repl
        return match.group(0)

    return pattern.sub(_f, text)


def normalize_domain_text(text: str) -> str:
    """电力领域规范化；设置 DOMAIN_NORMALIZE=false 可关闭。"""
    if not text or os.getenv("DOMAIN_NORMALIZE", "true").lower() in ("0", "false", "no"):
        return text

    out = text
    for entity, repl in _HTML_ENTITIES:
        out = out.replace(entity, repl)
    out = re.sub(r"&#x?[0-9A-Fa-f]+;?", " ", out)  # 其余实体兜底当空格
    out = out.replace("\u200b", "").replace("\ufeff", "")
    # 全角数字/字母 -> 半角
    out = "".join(
        chr(ord(c) - 0xFEE0)
        if 0xFF10 <= ord(c) <= 0xFF19 or 0xFF21 <= ord(c) <= 0xFF3A or 0xFF41 <= ord(c) <= 0xFF5A
        else c
        for c in out
    )
    # 全角罗马数字先转半角，再统一按罗马数字处理（比原来的「只处理 X段」更完整）
    for fw, ascii_roman in _FULLWIDTH_ROMAN.items():
        out = out.replace(fw, ascii_roman)
    out = _ROMAN_TOKEN_RE.sub(_roman_sub, out)

    # 符号 -> 中文读法（仅中文语境；先把符号换成占位符再判断上下文，避免符号本身干扰）
    def _sym_sub(match: "re.Match", repl: str) -> str:
        return repl if _zh_context(out, match.start(), match.end()) else match.group(0)

    import re as _re

    for sym, cn in _SYMBOL_PREFIX:  # 30% -> 百分之30
        out = _re.sub(
            rf"(\d+(?:\.\d+)?)\s*{_re.escape(sym)}",
            lambda m: (cn + m.group(1)) if _zh_context(out, m.start(), m.end()) else m.group(0),
            out,
        )
    for sym, cn in _SYMBOL_SUFFIX:  # #2 -> 2号
        out = _re.sub(
            rf"{_re.escape(sym)}\s*(\d+)",
            lambda m: (m.group(1) + cn) if _zh_context(out, m.start(), m.end()) else m.group(0),
            out,
        )
    for sym, cn in _SYMBOL_MAP:
        out = _re.sub(_re.escape(sym), lambda m: _sym_sub(m, cn), out)

    # 先处理单位（需要"数字后紧跟单位"的位置关系），再把小数转成中文读法
    for pattern, repl in _UNIT_RULES:
        # skip_alnum：像 AC380V 这种前缀字母不影响"是否中文语境"的判断
        out = _sub_zh_context_skip_alnum(pattern, repl, out)
    out = _sub_zh_context_skip_alnum(_DECIMAL_RE, _decimal_sub, out)

    # 字母与数字相邻时拆开：11F5 -> 11 F 5（数字由中文音色读，字母由英文音色读；
    # 否则 "F5" 会整段交给英文音色读成 "F five"）
    out = _ALNUM_TOKEN_RE.sub(_split_alnum_token, out)
    # 长设备号逐位读：1198号 -> 一一九八号
    out = _DEVICE_NO_RE.sub(_device_no_sub, out)

    # 大写字母/缩写 -> 中文近似读法（AC -> 诶西，GPU -> 记皮优）
    # 这样电力场景里的相别、设备缩写都由中文音色读，不会出现音色跳动；
    # 含小写字母的英文单词/句子不受影响，仍走英文音色。
    out = _UPPER_RUN_RE.sub(_upper_run_to_chinese, out)
    out = re.sub(r"[ \t\r\f\v]+", " ", out)
    out = re.sub(r"\s*\n\s*", " ", out)
    return out.strip()


# --------------------------------------------------------------------------
# 2. 语种切分
# --------------------------------------------------------------------------
_CJK = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_LATIN = re.compile(r"[A-Za-z]")
# 纯大写缩写（2~8 个字母），如 AC / GPU / FTU / SCADA —— 交给英文音色按字母读
_ACRONYM = re.compile(r"^[A-Z]{2,8}$")


def _classify(ch: str) -> Optional[str]:
    if _CJK.match(ch):
        return ZH_LANG
    if _LATIN.match(ch):
        return EN_LANG
    return None  # 数字/标点/空格等中性字符


def split_by_language(text: str, default_lang: str = ZH_LANG) -> List[Tuple[str, str]]:
    """按语种切分；中性字符跟随前一个片段，段首中性字符并入后一个片段。"""
    runs: List[List[str]] = []
    neutral_start = ""
    for ch in text:
        lang = _classify(ch)
        if lang is None:
            if runs:
                runs[-1][1] += ch
            else:
                neutral_start += ch
            continue
        if neutral_start:
            ch = neutral_start + ch
            neutral_start = ""
        if runs and runs[-1][0] == lang:
            runs[-1][1] += ch
        else:
            runs.append([lang, ch])
    if neutral_start:
        runs.append([default_lang, neutral_start])
    return [(lang, txt) for lang, txt in runs if txt and (_CJK.search(txt) or _LATIN.search(txt))]


def merge_short_spans(spans: List[Tuple[str, str]], min_chars: int) -> List[Tuple[str, str]]:
    """过短的语种片段并入相邻片段（音色不来回跳）。

    例外：**任何含拉丁字母的片段都不并入中文片段**，一律交给英文音色。
    实测中文（piper zh）词表里没有拉丁字符：
      "故障相别AC相"  -> "故障相别箱"（读错）
      "GPU 监控"      -> GPU 整段丢失（静音）
      "A相"           -> "A" 静音
    所以单个字母也必须走英文音色，否则会丢字。
    """
    if min_chars <= 1 or len(spans) <= 1:
        return spans
    merged: List[List[str]] = []
    for lang, text in spans:
        if lang == EN_LANG:
            merged.append([lang, text])          # 含字母 -> 独立成段，交给英文音色
            continue
        # 只并入「前一个同语种」片段；不能并进英文片段（否则英文音色会去读中文）
        if merged and len(text) < min_chars and merged[-1][0] == lang:
            merged[-1][1] += text
        else:
            merged.append([lang, text])
    if len(merged) > 1 and len(merged[0][1]) < min_chars and merged[0][0] == merged[1][0]:
        merged[1][1] = merged[0][1] + merged[1][1]
        merged.pop(0)
    return [(lang, txt) for lang, txt in merged if txt]


# --------------------------------------------------------------------------
# 3. 音色决议：zh -> female/male，en -> en_female/en_male
# --------------------------------------------------------------------------
ZH_VOICE_FEMALE = os.getenv("AUTO_VOICE_ZH_FEMALE", "female")
ZH_VOICE_MALE = os.getenv("AUTO_VOICE_ZH_MALE", "male")
EN_VOICE_FEMALE = os.getenv("AUTO_VOICE_EN_FEMALE", "en_female")
EN_VOICE_MALE = os.getenv("AUTO_VOICE_EN_MALE", "en_male")


def pick_voice(lang: str, gender: str) -> str:
    if lang == EN_LANG:
        return EN_VOICE_MALE if gender == "male" else EN_VOICE_FEMALE
    return ZH_VOICE_MALE if gender == "male" else ZH_VOICE_FEMALE


def resolve_voice_spans(
    text: str, voice: str, lang_code: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """返回 [(voice_id, lang, text)]。

    voice="auto" / "auto:male" / "auto:female" 触发言种路由；其它取值保持单音色行为。
    """
    v = (voice or "").strip().lower()
    if not v.startswith("auto"):
        return [(voice, lang_code or ZH_LANG, text)]

    gender = "male" if ":male" in v else "female"
    spans = merge_short_spans(split_by_language(text), AUTO_MIN_SPAN_CHARS)
    routed = [(pick_voice(lang, gender), lang, span_text) for lang, span_text in spans]

    # 全部落在同一音色时按原文处理，避免无谓的片段切分改变停顿
    if len({voice_id for voice_id, _, _ in routed}) <= 1:
        if not routed:
            return [(pick_voice(ZH_LANG, gender), ZH_LANG, text)]
        voice_id, _, _ = routed[0]
        lang = routed[0][1] if len({lg for _, lg, _ in routed}) == 1 else ZH_LANG
        return [(voice_id, lang, text)]
    return routed
