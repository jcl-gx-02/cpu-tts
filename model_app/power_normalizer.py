"""Electric-power domain text normalizer (v3.0).

The module combines common Chinese TTS normalization (date, clock, percentage,
range, symbols and telephone numbers) with electricity-specific symbols,
units, equipment identifiers, standards and dispatch phrases.

Time/date/telephone spans are rendered first and protected from later generic
digit and colon rules. This is important because ``8:35`` is a clock time,
while ``1:2`` is a ratio.

Normalization functions are pure. The optional letter-name JSON is read once at
import time; there is no model loading or request-time global state.
"""

from __future__ import annotations

import html
import json
import os
import re

_DIGITS = "零一二三四五六七八九"
_SUPPORTED_LANGUAGES = {"zh", "all_zh", "auto"}
_SUPPORTED_ACRONYM_STYLES = {"english", "chinese"}

# 26字母汉字注音表（可由 LETTER_NAMES_FILE 覆盖）
_DEFAULT_LETTER_NAMES_CN = {
    "A": "诶",
    "B": "必",
    "C": "西",
    "D": "迪",
    "E": "伊",
    "F": "艾弗",
    "G": "记",
    "H": "艾尺",
    "I": "爱",
    "J": "杰",
    "K": "开",
    "L": "艾勒",
    "M": "艾姆",
    "N": "恩",
    "O": "欧",
    "P": "屁",
    "Q": "丘",
    "R": "阿尔",
    "S": "矮丝",
    "T": "提",
    "U": "优",
    "V": "维",
    "W": "达不留",
    "X": "艾克斯",
    "Y": "歪",
    "Z": "贼",
}


def _load_letter_names() -> dict[str, str]:
    names = dict(_DEFAULT_LETTER_NAMES_CN)
    path = os.getenv("LETTER_NAMES_FILE", "/app/letter_names.json")
    try:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                overrides = json.load(stream)
            names.update(
                {
                    str(key).upper(): str(value)
                    for key, value in overrides.items()
                    if len(str(key)) == 1 and str(key).isalpha()
                }
            )
    except (OSError, ValueError, TypeError):
        pass
    return names


_LETTER_NAMES_CN = _load_letter_names()

# 电力调度单位映射表（全部小写键）
_UNIT_MAP = {
    "kv": "千伏",
    "v": "伏",
    "mv": "毫伏",
    "ka": "千安",
    "a": "安",
    "ma": "毫安",
    "mw": "兆瓦",
    "kw": "千瓦",
    "gw": "吉瓦",
    "w": "瓦",
    "mva": "兆伏安",
    "kva": "千伏安",
    "mvar": "兆乏",
    "kvar": "千乏",
    "var": "乏",
    "mwh": "兆瓦时",
    "kwh": "千瓦时",
    "km": "千米",
    "cm": "厘米",
    "mm": "毫米",
    "m": "米",
    "ms": "毫秒",
    "s": "秒",
    "min": "分钟",
    "h": "小时",
    "mpa": "兆帕",
    "kpa": "千帕",
    "pa": "帕",
    "hz": "赫兹",
    "khz": "千赫兹",
    "mhz": "兆赫兹",
    "ghz": "吉赫兹",
    "℃": "摄氏度",
    "°c": "摄氏度",
    "Ω": "欧姆",
    "ω": "欧姆",
    "kΩ": "千欧",
    "mΩ": "毫欧",
    "MΩ": "兆欧",
}

_UNIT_TOKEN = (
    r"(?:万千瓦时|万千瓦|千瓦时|兆瓦时|千伏安|兆伏安|千赫兹|兆赫兹|吉赫兹|"
    r"摄氏度|千伏|毫伏|兆瓦|千瓦|吉瓦|兆乏|千乏|千安|毫安|兆帕|千帕|赫兹|欧姆|千欧|兆欧|"
    r"千米|厘米|毫米|毫秒|分钟|小时|米|秒|"
    r"MVar|Mvar|kVar|kvar|MVA|kVA|MWh|kWh|GHz|MHz|kHz|MPa|Mpa|kPa|kpa|"
    r"MW|Mw|kW|KW|GW|kV|KV|mV|kA|mA|km|cm|mm|ms|min|Hz|℃|°C|kΩ|MΩ|mΩ|Ω|V|A|W|m|s|h|伏|安|瓦|乏)"
)
_POWER_UNIT_TOKEN = (
    r"(?:万千瓦|兆瓦|千瓦|吉瓦|兆伏安|千伏安|兆乏|千乏|"
    r"MVar|Mvar|kVar|kvar|MVA|kVA|MW|Mw|kW|KW|GW)"
)

_EQUIPMENT_NOUN = (
    r"(?:接地刀闸|隔离开关|断路器|发变组|联络变|高厂变|启备变|站用变|所用变|"
    r"调相机|电抗器|电容器|消弧线圈|备自投|重合闸|汇控柜|"
    r"刀闸|开关|机组|主变|联变|母线|高抗|低抗|装置|保护|安控|稳控|间隔|馈线|站号)"
)


def is_supported(language: str) -> bool:
    """Return True for languages where the electric-power rules apply (L3)."""
    return (language or "").strip().lower() in _SUPPORTED_LANGUAGES


def _digits_to_cn(text: str) -> str:
    return "".join(_DIGITS[int(ch)] for ch in str(text) if ch.isdigit())


def _section_to_cn(value: int) -> str:
    """Convert 0..9999 to Chinese; internal helper for _int_to_cn."""
    if value == 0:
        return ""
    out = []
    zero_pending = False
    for position, unit in ((3, "千"), (2, "百"), (1, "十"), (0, "")):
        digit = (value // (10 ** position)) % 10
        if digit == 0:
            if out:
                zero_pending = True
            continue
        if zero_pending:
            out.append("零")
            zero_pending = False
        out.append(_DIGITS[digit] + unit)
    result = "".join(out)
    if result.startswith("一十"):
        result = result[1:]
    return result


def _int_to_cn(value: int) -> str:
    if value == 0:
        return "零"
    if value == 2000:
        # Sheet convention: 2000:1 -> 两千比一
        return "两千"
    if value < 0:
        return "负" + _int_to_cn(-value)
    if value < 10000:
        return _section_to_cn(value)
    high, low = divmod(value, 10000)
    result = _int_to_cn(high) + "万"
    if low == 0:
        return result
    if low < 1000:
        result += "零"
    return result + _section_to_cn(low)


def _smart_ordinal_to_cn(num_str: str) -> str:
    """智能序号：带前导0（01）或3位及以上编号（5011）逐位读；1~2位无前导0（5、14、42）读十进制."""
    num_str = num_str.strip()
    if not num_str.isdigit():
        return num_str
    if (len(num_str) > 1 and num_str.startswith("0")) or len(num_str) >= 3:
        return _digits_to_cn(num_str)
    return _int_to_cn(int(num_str))


def _decimal_to_cn(text: str) -> str:
    text = text.strip()
    sign = ""
    if text.startswith("-"):
        sign, text = "负", text[1:]
    elif text.startswith("+"):
        text = text[1:]
    if "." in text:
        integer, fraction = text.split(".", 1)
        integer_cn = _int_to_cn(int(integer or "0"))
        return sign + integer_cn + "点" + _digits_to_cn(fraction)
    return sign + _int_to_cn(int(text or "0"))


def _percent_value_cn(text: str) -> str:
    try:
        if float(text) == 100.0 and "." not in text:
            return "百"
    except (TypeError, ValueError):
        pass
    return _decimal_to_cn(text)


_ROMAN_TABLE = {
    "I": "一", "II": "二", "III": "三", "IV": "四", "V": "五",
    "VI": "六", "VII": "七", "VIII": "八", "IX": "九", "X": "十",
    "Ⅰ": "一", "Ⅱ": "二", "Ⅲ": "三", "Ⅳ": "四", "Ⅴ": "五",
    "Ⅵ": "六", "Ⅶ": "七", "Ⅷ": "八", "Ⅸ": "九", "Ⅹ": "十",
}


def _roman_value(text: str) -> str:
    return _ROMAN_TABLE.get(text, text)


_PROTECTED_BASE = 0xE000
_PROTECTED_LIMIT = 0xF8FF

_SYMBOL_TRANSLATION = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E",
    "Ｆ": "F", "Ｇ": "G", "Ｈ": "H", "Ｉ": "I", "Ｊ": "J",
    "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "O",
    "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T",
    "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y",
    "Ｚ": "Z",
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e",
    "ｆ": "f", "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j",
    "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n", "ｏ": "o",
    "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t",
    "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y",
    "ｚ": "z",
    "／": "/",
    "＋": "+",
    "－": "-",
    "−": "-",
    "–": "-",
    "＊": "*",
    "％": "%",
    "＃": "#",
})


def _sanitize_common_text(text: str) -> str:
    """Decode transport artifacts and canonicalize full-width ASCII."""
    text = html.unescape(text)
    text = (
        text.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    text = text.translate(_SYMBOL_TRANSLATION)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


_CLOCK_CORE = (
    r"(?:[01]?\d|2[0-3])"
    r"\s*[:：]\s*[0-5]\d"
    r"(?:\s*[:：]\s*[0-5]\d)?"
)
_CLOCK_RANGE_RE = re.compile(
    rf"(?<!\d)({_CLOCK_CORE})\s*(?:至|到|~|～|-|—)\s*"
    rf"({_CLOCK_CORE})(?!\d)"
)
_CLOCK_RE = re.compile(rf"(?<!\d)({_CLOCK_CORE})(?!\d)")
_DATE_CN_RE = re.compile(
    r"(?<!\d)(\d{4})年"
    r"(?:(\d{1,2})月(?:(\d{1,2})[日号])?)?"
)
_DATE_ISO_RE = re.compile(
    r"(?<!\d)(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(?!\d)"
)
_DATE_MD_RE = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]")
_MOBILE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_LANDLINE_RE = re.compile(
    r"(?<!\d)(0\d{2,3})[-—]\s*(\d{7,8})"
    r"(?:(?:[-—]|转)\s*(\d{1,6}))?(?!\d)"
)
_ACDC_VOLTAGE_RE = re.compile(
    r"(?<![A-Za-z])(AC|DC)\s*(\d+(?:\.\d+)?)\s*(kV|V)(?![A-Za-z])",
    re.IGNORECASE,
)


def _clock_part_to_cn(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("0") and len(value) > 1 and int(value) != 0:
        return _DIGITS[0] + _int_to_cn(int(value))
    return _int_to_cn(int(value))


def _render_clock(value: str) -> str:
    parts = re.split(r"\s*[:：]\s*", value.strip())
    hour = _int_to_cn(int(parts[0]))
    if len(parts) == 1:
        return hour + "点"
    minute_value = int(parts[1])
    result = hour + "点" + ("整" if minute_value == 0 else _clock_part_to_cn(parts[1]) + "分")
    if len(parts) == 3:
        result += _clock_part_to_cn(parts[2]) + "秒"
    return result


def _digits_each(group: str) -> str:
    return "".join(_DIGITS[int(ch)] for ch in group if ch.isdigit())


def _normalize_common_atomic(text: str, protected: list[str]) -> str:
    """Protect date/time/telephone atoms before generic numeric rules."""
    text = _protect_sub(
        text,
        _CLOCK_RANGE_RE,
        lambda m: _render_clock(m.group(1)) + "至" + _render_clock(m.group(2)),
        protected,
    )
    text = _protect_sub(text, _CLOCK_RE, lambda m: _render_clock(m.group(1)), protected)

    def replace_date_cn(match):
        result = _digits_each(match.group(1)) + "年"
        if match.group(2):
            result += _int_to_cn(int(match.group(2))) + "月"
        if match.group(3):
            result += _int_to_cn(int(match.group(3))) + match.group(0)[-1]
        return result

    text = _protect_sub(text, _DATE_CN_RE, replace_date_cn, protected)
    text = _protect_sub(
        text,
        _DATE_ISO_RE,
        lambda m: _digits_each(m.group(1))
        + "年"
        + _int_to_cn(int(m.group(3)))
        + "月"
        + _int_to_cn(int(m.group(4)))
        + "日",
        protected,
    )
    text = _protect_sub(
        text,
        _DATE_MD_RE,
        lambda m: _int_to_cn(int(m.group(1)))
        + "月"
        + _int_to_cn(int(m.group(2)))
        + m.group(0)[-1],
        protected,
    )
    text = _protect_sub(text, _MOBILE_RE, lambda m: _digits_each(m.group(1)), protected)
    text = _protect_sub(
        text,
        _LANDLINE_RE,
        lambda m: _digits_each(m.group(1))
        + "，"
        + _digits_each(m.group(2))
        + (("转" + _digits_each(m.group(3))) if m.group(3) else ""),
        protected,
    )
    text = _protect_sub(
        text,
        _ACDC_VOLTAGE_RE,
        lambda m: ("交流" if m.group(1).upper() == "AC" else "直流")
        + _decimal_to_cn(m.group(2))
        + _unit_to_cn(m.group(3)),
        protected,
    )
    return text


def _canonicalize_symbols(text: str, protected: list[str]) -> str:
    """Canonicalize glyph variants while preserving Chinese punctuation like '：' and '——'."""
    if "——" in text:
        text = _protect_sub(text, re.compile(r"——+"), lambda m: "，", protected)
    text = text.replace("—", "-")
    text = re.sub(r"°\s*[Cc]", "℃", text)
    return text.translate(_SYMBOL_TRANSLATION)


def _protect_value(value: str, protected: list[str]) -> str:
    index = len(protected)
    if _PROTECTED_BASE + index > _PROTECTED_LIMIT:
        raise ValueError("too many protected normalized spans")
    protected.append(value)
    return chr(_PROTECTED_BASE + index)


def _protect_sub(
    text: str,
    pattern: re.Pattern,
    renderer,
    protected: list[str],
) -> str:
    def replace(match):
        rendered = renderer(match)
        if rendered == match.group(0):
            return rendered
        return _protect_value(rendered, protected)

    return pattern.sub(replace, text)


def _restore_protected(text: str, protected: list[str]) -> str:
    if not protected:
        return text
    max_char = chr(_PROTECTED_BASE + len(protected) - 1)
    return re.sub(
        rf"[{re.escape(chr(_PROTECTED_BASE))}-{re.escape(max_char)}]",
        lambda match: protected[ord(match.group(0)) - _PROTECTED_BASE],
        text,
    )


def _spaced_letters(text: str, acronym_style: str = "english") -> str:
    letters = "".join(char for char in text.upper() if char.isascii() and char.isalpha())
    if acronym_style == "chinese":
        return "".join(_LETTER_NAMES_CN.get(char, char) for char in letters)
    return " ".join(letters)


def _unit_to_cn(unit: str) -> str:
    u = unit.strip()
    return _UNIT_MAP.get(u, _UNIT_MAP.get(u.lower(), u))


def _sign_to_cn(sign: str, adjustment: bool = False) -> str:
    if sign == "±":
        return "正负"
    if sign == "-":
        return "负"
    if sign == "+":
        return "加" if adjustment else "正"
    return ""


def _render_alnum_mixed(segment: str, acronym_style: str) -> str:
    tokens = re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+", segment.strip().upper())
    out = []
    for tok in tokens:
        if tok[0].isdigit():
            if "." in tok:
                i_part, f_part = tok.split(".", 1)
                out.append(_digits_to_cn(i_part) + "点" + _digits_to_cn(f_part))
            else:
                out.append(_digits_to_cn(tok))
        else:
            out.append(_spaced_letters(tok, acronym_style))
    return "".join(out)


def _render_code_segment(segment: str, acronym_style: str) -> str:
    segment = segment.strip().upper()
    match = re.fullmatch(r"(\d+)([A-Z]*)", segment)
    if match:
        num_str, letters = match.group(1), match.group(2)
        if (len(num_str) > 1 and num_str.startswith("0")) or len(num_str) >= 4:
            number = _digits_to_cn(num_str)
        else:
            number = _int_to_cn(int(num_str))
        return number + (_spaced_letters(letters, acronym_style) if letters else "")
    if segment.isalpha():
        return _spaced_letters(segment, acronym_style)
    return _render_alnum_mixed(segment, acronym_style)


_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"Q/GDW|"
    r"[A-Z]{2,8}[0-9A-Z]{0,6}(?:(?:\s*[-/]\s*)(?:[A-Z]{1,8}|\d+(?:\.\d+)?[A-Z]?)){1,}"
    r")(?![A-Za-z0-9])"
)
_STANDARD_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(GB(?:/T)?|DL(?:/T)?|NB(?:/T)?|JB(?:/T)?|Q/GDW|IEC|IEEE)"
    r"\s*(\d+(?:\.\d+)?(?:\s*-\s*\d{2,4})?)(?![A-Za-z0-9])"
)


def _normalize_standard_codes(
    text: str,
    protected: list[str],
    acronym_style: str,
) -> str:
    def replace(match):
        prefix_parts = re.split(r"/", match.group(1))
        prefix = "杠".join(
            _spaced_letters(part, acronym_style) for part in prefix_parts
        )
        raw_num = match.group(2)
        sub_parts = re.split(r"\s*-\s*", raw_num)
        rendered_nums = []
        for part in sub_parts:
            if "." in part:
                integer, fraction = part.split(".", 1)
                rendered_nums.append(_digits_to_cn(integer) + "点" + _digits_to_cn(fraction))
            else:
                rendered_nums.append(_digits_to_cn(part))
        return prefix + "杠".join(rendered_nums)

    return _protect_sub(text, _STANDARD_CODE_RE, replace, protected)


def _normalize_codes(
    text: str,
    protected: list[str],
    acronym_style: str,
) -> str:
    def replace(match):
        full = match.group(0)
        tokens = re.split(r"\s*[-/]\s*", full)
        if all(t.lower() in _UNIT_MAP for t in tokens):
            return full
        parts = re.split(r"\s*([-/])\s*", full)
        rendered = [_render_code_segment(parts[0], acronym_style)]
        for index in range(1, len(parts), 2):
            sep = parts[index]
            left_seg = parts[index - 1].strip()
            right_seg = parts[index + 1].strip()
            # 型号中两个数字段之间的 "/" 读作 "比"（如 LGJ-630/55 -> 六百三十比五十五）
            if sep == "/" and left_seg.isdigit() and re.match(r"^\d+", right_seg):
                rendered.append("比")
            else:
                rendered.append("杠")
            rendered.append(_render_code_segment(right_seg, acronym_style))
        return "".join(rendered)

    return _protect_sub(text, _CODE_RE, replace, protected)


_TABLE_ID_PATTERN = r"\d+(?:\s*-\s*\d+)+"
_TABLE_RANGE_RE = re.compile(
    rf"([表图])\s*({_TABLE_ID_PATTERN})\s*(?:至|到)\s*({_TABLE_ID_PATTERN})"
)
_TABLE_ONE_RE = re.compile(rf"([表图])\s*({_TABLE_ID_PATTERN})")


def _render_table_id(value: str) -> str:
    return "杠".join(_int_to_cn(int(part)) for part in re.findall(r"\d+", value))


def _normalize_tables_atomic(text: str, protected: list[str]) -> str:
    text = _protect_sub(
        text,
        _TABLE_RANGE_RE,
        lambda match: match.group(1)
        + _render_table_id(match.group(2))
        + "至"
        + _render_table_id(match.group(3)),
        protected,
    )
    return _protect_sub(
        text,
        _TABLE_ONE_RE,
        lambda match: match.group(1) + _render_table_id(match.group(2)),
        protected,
    )


_SECTION_MEASURE_RE = re.compile(
    rf"^\s*{_UNIT_TOKEN}",
    re.IGNORECASE,
)
_SECTION_ATOMIC_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)+)(?:[.．](?=\s|[\u4e00-\u9fff]))?"
    r"(?=\s|[、，,；;：:]|[\u4e00-\u9fff]|$)"
)


def _normalize_sections_atomic(text: str, protected: list[str]) -> str:
    def replace(match):
        if _SECTION_MEASURE_RE.match(text[match.end():]):
            return match.group(0)
        parts = [int(part) for part in match.group(1).split(".")]
        rendered = "点".join(_int_to_cn(part) for part in parts)
        tail = text[match.end():]
        add_pause = (
            len(parts) >= 3
            and re.match(r"\s*[\u4e00-\u9fff]", tail) is not None
            and re.match(r"^\s*(?:条|款|项|目|章|节)", tail) is None
        )
        return rendered + ("，" if add_pause else "")

    return _protect_sub(text, _SECTION_ATOMIC_RE, replace, protected)


_SIGNED_RANGE_QUANTITY_RE = re.compile(
    rf"(?<![A-Za-z0-9])([+\-±]?)\s*(\d+(?:\.\d+)?)\s*(({_UNIT_TOKEN})?)\s*[~～\-]\s*"
    rf"([+\-±]?)\s*(\d+(?:\.\d+)?)\s*({_UNIT_TOKEN})(?![A-Za-z])"
)
_SIGNED_QUANTITY_RE = re.compile(
    rf"(?<![A-Za-z0-9])([+\-±])\s*(\d+(?:\.\d+)?)\s*({_UNIT_TOKEN})(?![A-Za-z])"
)
# 避免将 "2% -100%" 或 "2 -100%" 中的 "-100%" 提前误判为独立带符号百分数
_SIGNED_PERCENT_RE = re.compile(
    r"(?<![A-Za-z0-9%％])(?<!%\s)(?<!％\s)(?<!\d\s)([+\-±])\s*(\d+(?:\.\d+)?)\s*[%％]"
)


def _normalize_signed_values(text: str, protected: list[str]) -> str:
    def replace_range(match):
        s1, v1, u1, _, s2, v2, u2 = match.groups()
        left = _sign_to_cn(s1) + _decimal_to_cn(v1) + (_unit_to_cn(u1) if u1 else "")
        right = _sign_to_cn(s2) + _decimal_to_cn(v2) + _unit_to_cn(u2)
        return left + "至" + right

    text = _protect_sub(text, _SIGNED_RANGE_QUANTITY_RE, replace_range, protected)
    text = _protect_sub(
        text,
        _SIGNED_QUANTITY_RE,
        lambda match: _sign_to_cn(match.group(1))
        + _decimal_to_cn(match.group(2))
        + _unit_to_cn(match.group(3)),
        protected,
    )
    return _protect_sub(
        text,
        _SIGNED_PERCENT_RE,
        lambda match: _sign_to_cn(match.group(1), adjustment=True)
        + "百分之"
        + _percent_value_cn(match.group(2)),
        protected,
    )


_PARAMETER_CHAIN_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?"
    rf"(?:\s*[+/]\s*\d+(?:\.\d+)?(?:\s*[%％])?){{2,}})\s*({_UNIT_TOKEN})(?![A-Za-z])"
)


def _render_parameter_token(token: str) -> str:
    token = token.strip()
    if token.endswith(("%", "％")):
        return "百分之" + _percent_value_cn(token[:-1].strip())
    return _decimal_to_cn(token)


def _normalize_parameter_chains(text: str, protected: list[str]) -> str:
    def replace(match):
        parts = re.split(r"\s*([+/])\s*", match.group(1))
        rendered = [_render_parameter_token(parts[0])]
        for index in range(1, len(parts), 2):
            rendered.append("、" if parts[index] == "/" else "加")
            rendered.append(_render_parameter_token(parts[index + 1]))
        return "".join(rendered) + _unit_to_cn(match.group(2))

    return _protect_sub(text, _PARAMETER_CHAIN_RE, replace, protected)


_COMPOUND_RATE_RE = re.compile(
    rf"(?<![A-Za-z0-9])([+\-±]?)\s*(\d+(?:\.\d+)?)\s*"
    rf"({_POWER_UNIT_TOKEN})\s*/\s*(min|s|h|分钟|秒|小时)\b",
    re.IGNORECASE,
)
_MINUTE_RE = re.compile(r"(?<![/A-Za-z0-9])(\d+(?:\.\d+)?)\s*min\b", re.IGNORECASE)
_RATE_TIME_MAP = {
    "min": "每分钟", "分钟": "每分钟",
    "s": "每秒", "秒": "每秒",
    "h": "每小时", "小时": "每小时",
}


def _normalize_rates(text: str, protected: list[str]) -> str:
    text = _protect_sub(
        text,
        _COMPOUND_RATE_RE,
        lambda match: _sign_to_cn(match.group(1))
        + _decimal_to_cn(match.group(2))
        + _unit_to_cn(match.group(3))
        + _RATE_TIME_MAP.get(match.group(4).lower(), "每" + match.group(4)),
        protected,
    )
    return _protect_sub(
        text,
        _MINUTE_RE,
        lambda match: _decimal_to_cn(match.group(1)) + "分钟",
        protected,
    )


_MULTIPLIER_RE = re.compile(
    rf"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[*×]\s*"
    rf"(\d+(?:\.\d+)?)\s*({_POWER_UNIT_TOKEN})(?![A-Za-z])"
)


def _normalize_multipliers(text: str, protected: list[str]) -> str:
    return _protect_sub(
        text,
        _MULTIPLIER_RE,
        lambda match: _decimal_to_cn(match.group(1))
        + "乘以"
        + _decimal_to_cn(match.group(2))
        + _unit_to_cn(match.group(3)),
        protected,
    )


_HASH_RANGE_PRE_RE = re.compile(
    rf"#\s*(\d+)\s*[-~～]\s*#?\s*(\d+)\s*({_EQUIPMENT_NOUN})"
)
_HASH_RANGE_POST_RE = re.compile(
    rf"(\d+)\s*#\s*[-~～]\s*(\d+)\s*#\s*({_EQUIPMENT_NOUN})"
)
_DURATION_RANGE_RE = re.compile(
    r"(?<![\d.])(\d+)\s*[-~～]\s*(\d+)\s*"
    r"(小时|分钟|秒|天|年|月|日|台|个|次|条|回|套)"
)


def _normalize_ranges_atomic(text: str, protected: list[str]) -> str:
    text = _protect_sub(
        text,
        _HASH_RANGE_PRE_RE,
        lambda m: _int_to_cn(int(m.group(1)))
        + "号到"
        + _int_to_cn(int(m.group(2)))
        + "号"
        + m.group(3),
        protected,
    )
    text = _protect_sub(
        text,
        _HASH_RANGE_POST_RE,
        lambda m: _int_to_cn(int(m.group(1)))
        + "号到"
        + _int_to_cn(int(m.group(2)))
        + "号"
        + m.group(3),
        protected,
    )
    return _protect_sub(
        text,
        _DURATION_RANGE_RE,
        lambda match: _int_to_cn(int(match.group(1)))
        + "到"
        + _int_to_cn(int(match.group(2)))
        + match.group(3),
        protected,
    )


_LIST_PREFIX = r"(^|[。！？；：，,\n])"
_PAREN_LIST_PREFIX = r"(^|[\s。！？；：，,\n\u4e00-\u9fff])"
_LETTER_LIST_PREFIX = r"(^|[。！？；：，,\n\u4e00-\u9fff])"
_MEASURE_AFTER_LIST = (
    r"(?!(?:万千瓦|千伏安|兆伏安|千瓦时|兆瓦时|千伏|kV|KV|伏|V|兆瓦|MW|千瓦|kW|兆乏|Mvar|千乏|kvar|"
    r"兆帕|MPa|赫兹|Hz|kHz|MHz|千安|kA|安|A|摄氏度|℃|"
    r"小时|分钟|秒|天|年|月|日|个|台|套|次|条|回|号|线|段|母|主变|侧|"
    r"[×÷+=<>≤≥]))"
)
_LIST_MARKERS = [
    re.compile(
        _PAREN_LIST_PREFIX
        + r"\s*[（(]\s*(\d{1,3}|[一二三四五六七八九十百]+)\s*[）)]\s*[、，：。]?\s*"
        + _MEASURE_AFTER_LIST
    ),
    re.compile(_LIST_PREFIX + r"\s*(\d{1,3})\s*[）)]\s*[、，：。]?\s*"),
    re.compile(
        _LETTER_LIST_PREFIX
        + r"\s*([A-Za-z])\s*[）)]\s*[、，：。]?\s*"
    ),
    re.compile(
        _LIST_PREFIX
        + r"\s*(\d{1,3})\s*[.．、]\s*"
        + _MEASURE_AFTER_LIST
        + r"(?=[\s\u4e00-\u9fff]|$)"
    ),
    re.compile(_LIST_PREFIX + r"\s*(\d{1,2})\s+" + _MEASURE_AFTER_LIST + r"(?=\S)"),
]


def _normalize_list_markers(text: str, acronym_style: str = "english") -> str:
    def replace(match):
        raw = match.group(2)
        if raw.isdigit():
            raw = _int_to_cn(int(raw))
        elif raw.isalpha():
            raw = _spaced_letters(raw, acronym_style) if acronym_style == "chinese" else raw.upper()
        prefix = match.group(1)
        if prefix and prefix.isspace():
            prefix = "，"
        elif prefix and "\u4e00" <= prefix <= "\u9fff":
            prefix += "，"
        return prefix + raw + "，"

    for pattern in _LIST_MARKERS:
        text = pattern.sub(replace, text)
    return text


_PAREN_PAIR_RE = re.compile(r"(?<![\d.])(\d{1,3})\s*[（(]\s*(\d{1,3})\s*[）)]")


def _normalize_parenthetical_pairs(text: str) -> str:
    def replace(match):
        return _int_to_cn(int(match.group(1))) + "、" + _int_to_cn(int(match.group(2)))

    return _PAREN_PAIR_RE.sub(replace, text)


_NOTE_RE = re.compile(r"(?<![\u4e00-\u9fff])注\s*(\d+)\s*([：:])")


def _normalize_note_labels(text: str) -> str:
    return _NOTE_RE.sub(lambda m: "注" + _int_to_cn(int(m.group(1))) + m.group(2), text)


_DISPATCH_ORDER_RE = re.compile(r"(?<![\d.])(第?\s*)(\d+)\s*(号(?:调度)?令)")


def _normalize_dispatch_orders(text: str) -> str:
    return _DISPATCH_ORDER_RE.sub(
        lambda m: (m.group(1).strip() if m.group(1) else "")
        + _smart_ordinal_to_cn(m.group(2))
        + m.group(3),
        text,
    )


_HASH_M_RE = re.compile(r"(?:#\s*(\d+)|(\d+)\s*#)\s*M(?![A-Za-z])")
_HASH_LIST_PRE_RE = re.compile(
    rf"#\s*(\d+)\s*[、,，]\s*#?\s*(\d+)(?=\s*(?:{_EQUIPMENT_NOUN}|线|号|$))"
)
_HASH_LIST_POST_RE = re.compile(
    rf"(\d+)\s*#\s*[、,，]\s*(\d+)\s*#(?=\s*(?:{_EQUIPMENT_NOUN}|线|号|$))"
)
_HASH_PRE_RE = re.compile(r"#\s*(\d+)\s*号?")
_HASH_POST_RE = re.compile(rf"(\d+)\s*#\s*号?(?=\s*(?:{_EQUIPMENT_NOUN}|线|母|变|机|组))")


def _normalize_hashes(text: str) -> str:
    text = _HASH_LIST_PRE_RE.sub(
        lambda m: _smart_ordinal_to_cn(m.group(1))
        + "号、"
        + _smart_ordinal_to_cn(m.group(2))
        + "号",
        text,
    )
    text = _HASH_LIST_POST_RE.sub(
        lambda m: _smart_ordinal_to_cn(m.group(1))
        + "号、"
        + _smart_ordinal_to_cn(m.group(2))
        + "号",
        text,
    )
    text = _HASH_M_RE.sub(
        lambda m: _smart_ordinal_to_cn(m.group(1) or m.group(2)) + "号母线",
        text,
    )
    text = _HASH_POST_RE.sub(lambda m: _smart_ordinal_to_cn(m.group(1)) + "号", text)
    return _HASH_PRE_RE.sub(lambda m: _smart_ordinal_to_cn(m.group(1)) + "号", text)


_NON_DEVICE_AFTER_HAO = re.compile(r"^\s*(?:建筑|楼|院|门|路|街|道|栋|室|房|大厅|文件|令|文|公告|通知)")
_ID_NUMBER_RE = re.compile(r"(?<![\d.])(\d+)\s*号")


def _normalize_id_numbers(text: str) -> str:
    def replace(match):
        num_str = match.group(1)
        tail = text[match.end():]
        if _NON_DEVICE_AFTER_HAO.match(tail) and not num_str.startswith("0"):
            return _int_to_cn(int(num_str)) + "号"
        return _digits_to_cn(num_str) + "号"

    return _ID_NUMBER_RE.sub(replace, text)


_ID_TOKEN = r"#?\d[0-9A-Za-z]*#?"
_EQUIPMENT_LIST_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_ID_TOKEN}(?:\s*[、,，]\s*{_ID_TOKEN})+)\s*"
    rf"(?=(?:及相关|及|和|与)?\s*{_EQUIPMENT_NOUN})"
)
_EQUIPMENT_ID_RE = re.compile(
    rf"(?<![A-Za-z0-9#])(\d[0-9A-Za-z]*)(?=\s*{_EQUIPMENT_NOUN})"
)
_LINE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9#])(\d[0-9A-Za-z]*)(?=[\s\u4e00-\u9fffⅠⅡⅢⅣⅤA-Za-z]{1,12}线)"
)
_MIXED_ID_RE = re.compile(
    r"(?<![#A-Za-z0-9])(\d[0-9A-Z]*[A-Z][0-9A-Z]*)(?![A-Za-z0-9])"
)


def _mixed_id_to_cn(value: str, acronym_style: str) -> str:
    value = value.strip("#")

    def replace(match):
        char = match.group(0)
        if char.isdigit():
            return _DIGITS[int(char)]
        return _spaced_letters(char, acronym_style)

    return re.sub(r"\d|[A-Za-z]", replace, value)


def _normalize_equipment_numbers(text: str, acronym_style: str) -> str:
    def replace_list(match):
        values = re.split(r"\s*[、,，]\s*", match.group(1))
        add_ordinal_suffix = any(v.startswith("#") or v.endswith("#") for v in values)
        rendered = []
        for value in values:
            suffix = "号" if add_ordinal_suffix else ""
            rendered.append(_mixed_id_to_cn(value, acronym_style) + suffix)
        return "、".join(rendered)

    text = _EQUIPMENT_LIST_RE.sub(replace_list, text)
    text = _EQUIPMENT_ID_RE.sub(
        lambda m: _mixed_id_to_cn(m.group(1), acronym_style),
        text,
    )
    text = _LINE_ID_RE.sub(
        lambda m: _mixed_id_to_cn(m.group(1), acronym_style),
        text,
    )
    return _MIXED_ID_RE.sub(
        lambda m: _mixed_id_to_cn(m.group(1), acronym_style),
        text,
    )


def _normalize_hashes_and_equipment(text: str, acronym_style: str) -> str:
    """Coordinate # markers, equipment numbers, and 号 suffixes."""
    text = _normalize_hashes(text)
    text = _normalize_equipment_numbers(text, acronym_style)
    text = _normalize_id_numbers(text)
    return text


_BOTH_UNIT_SLASH_RE = re.compile(
    rf"(\d+(?:\.\d+)?\s*{_UNIT_TOKEN})\s*/\s*(\d+(?:\.\d+)?\s*{_UNIT_TOKEN})"
)
_TRAILING_UNIT_SLASH_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(\s*{_UNIT_TOKEN})"
)
_MULTI_NUMBER_SLASH_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?){2,})(?![A-Za-z0-9.])"
)
_TWO_NUMBER_SLASH_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![A-Za-z0-9.])"
)
_NAME_SLASH_RE = re.compile(
    r"(?<=[A-Za-z0-9\u4e00-\u9fff])\s*/\s*(?=[A-Za-z\u4e00-\u9fff])"
)
_FRACTION_SLASH_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
    r"(?=\s*(?:接线|断路器|开关))"
)
_CONTRAST_SLASH_RE = re.compile(
    r"(?<![A-Za-z0-9])(解锁|闭锁|投入|退出|合上|断开|送电|受电)"
    r"\s*/\s*(解锁|闭锁|投入|退出|合上|断开|送电|受电)(?![A-Za-z0-9])"
)


def _normalize_slashes(text: str) -> str:
    text = _FRACTION_SLASH_RE.sub(
        lambda m: _decimal_to_cn(m.group(2)) + "分之" + _decimal_to_cn(m.group(1)),
        text,
    )
    text = _CONTRAST_SLASH_RE.sub(
        lambda m: m.group(1) + "比" + m.group(2),
        text,
    )
    text = _BOTH_UNIT_SLASH_RE.sub(lambda m: m.group(1) + "比" + m.group(2), text)
    text = _MULTI_NUMBER_SLASH_RE.sub(lambda m: m.group(1).replace("/", "、"), text)
    text = _TRAILING_UNIT_SLASH_RE.sub(
        lambda m: m.group(1) + "、" + m.group(2) + m.group(3), text
    )
    text = _TWO_NUMBER_SLASH_RE.sub(lambda m: m.group(1) + "比" + m.group(2), text)
    text = _NAME_SLASH_RE.sub("、", text)
    return text.replace("/", "比")


_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*({_UNIT_TOKEN})(?![A-Za-z])",
    re.IGNORECASE,
)
_STANDALONE_UNIT_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(MVar|Mvar|kVar|kvar|MVA|kVA|MWh|kWh|GHz|MHz|kHz|MPa|Mpa|kPa|kpa|"
    r"MW|Mw|kW|KW|GW|kV|KV|kA|km|cm|mm|ms|min|Hz|MΩ|mΩ|℃)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_PU_WITH_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*p\.u\.", re.IGNORECASE)
_PU_RE = re.compile(r"p\.u\.", re.IGNORECASE)


def _normalize_units(text: str) -> str:
    def replace(match):
        number = _decimal_to_cn(match.group(1))
        unit = _unit_to_cn(match.group(2))
        return number + unit

    text = _PU_WITH_NUMBER_RE.sub(lambda m: _decimal_to_cn(m.group(1)) + "标幺", text)
    text = _PU_RE.sub("标幺", text)
    text = _UNIT_RE.sub(replace, text)
    return _STANDALONE_UNIT_RE.sub(lambda m: _unit_to_cn(m.group(1)), text)


_KP_RE = re.compile(r"(?<![A-Za-z])Kp\s*[%％]", re.IGNORECASE)
_KV_RE = re.compile(r"(?<![A-Za-z])Kv\s*[%％]", re.IGNORECASE)


def _normalize_special_phrases(text: str) -> str:
    text = _KP_RE.sub("稳定系数百分比", text)
    text = _KV_RE.sub("短路电压百分比", text)
    return text


_PERCENT_BOTH_RE = re.compile(
    r"([+\-]?\d+(?:\.\d+)?)\s*[%％]\s*(?:[-~～]\s*([+\-]?\d+(?:\.\d+)?)|-\s*(\d+(?:\.\d+)?))\s*[%％]"
)
_PERCENT_RIGHT_RE = re.compile(
    r"([+\-]?\d+(?:\.\d+)?)\s*(?:[-~～]\s*([+\-]?\d+(?:\.\d+)?)|-\s*(\d+(?:\.\d+)?))\s*[%％]"
)
_PERCENT_ONE_RE = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*[%％]")


def _render_single_percent(val_str: str) -> str:
    val_str = val_str.strip()
    sign_cn = ""
    if val_str.startswith("-"):
        sign_cn, val_str = "负", val_str[1:]
    elif val_str.startswith("+"):
        sign_cn, val_str = "正", val_str[1:]
    return sign_cn + "百分之" + _percent_value_cn(val_str)


def _normalize_percentages(text: str) -> str:
    text = _PERCENT_BOTH_RE.sub(
        lambda m: _render_single_percent(m.group(1))
        + "至"
        + _render_single_percent(m.group(2) or m.group(3)),
        text,
    )
    text = _PERCENT_RIGHT_RE.sub(
        lambda m: _decimal_to_cn(m.group(1))
        + "至"
        + _render_single_percent(m.group(2) or m.group(3)),
        text,
    )
    return _PERCENT_ONE_RE.sub(lambda m: _render_single_percent(m.group(1)), text)


_PERMILLE_ONE_RE = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*‰")


def _normalize_permille(text: str) -> str:
    def replace(match):
        value = match.group(1)
        sign = ""
        if value.startswith("-"):
            sign, value = "负", value[1:]
        elif value.startswith("+"):
            value = value[1:]
        return sign + "千分之" + _percent_value_cn(value)

    return _PERMILLE_ONE_RE.sub(replace, text)


def _normalize_comparators(text: str) -> str:
    return (
        text.replace("≥", "大于等于")
        .replace("≤", "小于等于")
        .replace("＞", "大于")
        .replace("＜", "小于")
        .replace(">", "大于")
        .replace("<", "小于")
        .replace("≧", "大于等于")
        .replace("≦", "小于等于")
    )


_COMMON_SYMBOL_MAP = [
    ("≈", "约等于"),
    ("≠", "不等于"),
    ("→", "到"),
    ("←", "到"),
    ("↑", "上升"),
    ("↓", "下降"),
    ("·", "点"),
    ("μ", "微"),
    ("²", "平方"),
    ("³", "立方"),
    ("√", "根号"),
    ("∞", "无穷"),
    ("&", "和"),
    ("@", "艾特"),
]


def _normalize_common_symbols(text: str) -> str:
    for symbol, spoken in _COMMON_SYMBOL_MAP:
        text = text.replace(symbol, spoken)
    return text


_ROMAN_PAIR_RE = re.compile(
    r"(?<![A-Za-z])(VIII|VII|VI|IV|IX|III|II|I|V|X)\s*"
    r"[（(]\s*(VIII|VII|VI|IV|IX|III|II|I|V|X)\s*[）)]\s*"
    r"(线|段|回|侧)"
)
_ROMAN_CONTEXT_RE = re.compile(
    r"(?<![A-Za-z])(VIII|VII|VI|IV|IX|III|II|I|V|X)(?=\s*(?:段|线|类|期|号|回|侧|母))"
)


def _normalize_roman(text: str) -> str:
    text = _ROMAN_PAIR_RE.sub(
        lambda m: _roman_value(m.group(1))
        + m.group(3)
        + "、"
        + _roman_value(m.group(2))
        + m.group(3),
        text,
    )
    text = _ROMAN_CONTEXT_RE.sub(lambda m: _roman_value(m.group(1)), text)
    for k, v in _ROMAN_TABLE.items():
        if not k.isascii():
            text = text.replace(k, v)
    return text


_NEG_RE = re.compile(r"(?<![\w\u4e00-\u9fff])-\s*(\d+(?:\.\d+)?)")
_QUOTED_MINUS_RE = re.compile(r"[“\"']\s*-\s*[”\"']")
_FOR_DIRECTION_MINUS_RE = re.compile(r"为\s*[“\"']?\s*-\s*[”\"']?")
_FORMULA_WORD_MINUS_RE = re.compile(r"(功率|出力|负荷|电流|电压|容量|系数|值|数)\s*-\s*")
_N_FORMULA_MINUS_RE = re.compile(r"\bN\s*-\s*(\d+)(?:\s*-\s*(\d+))?")
_PRODUCT_VERSION_RE = re.compile(r"(?<=[A-Za-z])\s*-\s*((?:V|v)\d+)")
_CJK_ALNUM_CHAIN_RE = re.compile(
    r"([\u4e00-\u9fff]+(?:\s*-\s*[\u4e00-\u9fff0-9]+)+)"
)
_ALNUM_MINUS_RE = re.compile(
    r"(?<![A-Za-z0-9.])([A-Za-z0-9.]+)\s*-\s*([A-Za-z0-9.]+)(?![A-Za-z0-9.])"
)


def _replace_cjk_alnum_chain(match):
    parts = re.split(r"\s*-\s*", match.group(1))
    rendered = [
        _smart_ordinal_to_cn(p) if p.isdigit() else p
        for p in parts
    ]
    return "杠".join(rendered)


def _replace_alnum_minus(match):
    left, right = match.group(1), match.group(2)
    if left.isdigit() or right.isdigit():
        return left + "减" + right
    return match.group(0)


def _normalize_hyphens(text: str) -> str:
    text = _QUOTED_MINUS_RE.sub("负", text)
    text = _FOR_DIRECTION_MINUS_RE.sub("为负", text)
    text = _N_FORMULA_MINUS_RE.sub(
        lambda m: "N减" + m.group(1) + ("减" + m.group(2) if m.group(2) else ""),
        text,
    )
    text = _NEG_RE.sub(lambda m: "负" + m.group(1), text)
    text = _FORMULA_WORD_MINUS_RE.sub(lambda m: m.group(1) + "减", text)
    text = _PRODUCT_VERSION_RE.sub(lambda m: " " + m.group(1), text)
    text = _ALNUM_MINUS_RE.sub(_replace_alnum_minus, text)
    text = _CJK_ALNUM_CHAIN_RE.sub(_replace_cjk_alnum_chain, text)
    return text


_DIRECTION_PLUS_RE = re.compile(r"为\s*[“\"']?\s*\+\s*[”\"']?")
_QUOTED_PLUS_RE = re.compile(r"[“\"']\s*\+\s*[”\"']")
_ALNUM_PLUS_RE = re.compile(r"([A-Za-z0-9]+)\s*\+\s*([A-Za-z0-9]+)")
_CJK_PLUS_RE = re.compile(r"([\u4e00-\u9fff]+)\s*\+\s*([\u4e00-\u9fff]+)")


def _normalize_pluses(text: str) -> str:
    text = _DIRECTION_PLUS_RE.sub("为正", text)
    text = _QUOTED_PLUS_RE.sub("正", text)
    text = _ALNUM_PLUS_RE.sub(lambda m: m.group(1) + "加" + m.group(2), text)
    text = _CJK_PLUS_RE.sub(lambda m: m.group(1) + "加" + m.group(2), text)
    return text.replace("+", "加")


_NUM_LETTER_TIMES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[×*]\s*([A-Za-z]+)")
_NUM_NUM_TIMES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[×*]\s*(\d+(?:\.\d+)?)")
_FORMULA_WORD_TIMES_RE = re.compile(r"(个数|功率|出力|负荷|电流|电压|容量|系数|调制|计算|等于)")


def _normalize_times_sign(text: str) -> str:
    text = _NUM_LETTER_TIMES_RE.sub(lambda m: m.group(1) + "乘以" + m.group(2), text)
    text = _NUM_NUM_TIMES_RE.sub(lambda m: m.group(1) + "乘以" + m.group(2), text)

    def replace_times(match):
        left = text[max(0, match.start() - 12):match.start()]
        right = text[match.end():match.end() + 12]
        context = left + right
        if _FORMULA_WORD_TIMES_RE.search(context):
            return "乘以"
        if re.search(r"[\u4e00-\u9fff]", left[-1:] + right[:1]):
            return "叉"
        return "乘以"

    return re.sub(r"[×*]", replace_times, text)


_ACRONYM_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,})(?![A-Za-z0-9])")
_SINGLE_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z0-9])")
_MIXED_ALNUM_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])"
    r"([A-Za-z0-9]+)(?![A-Za-z0-9.])"
)


def _normalize_acronyms(text: str, acronym_style: str) -> str:
    """Separate uppercase Latin abbreviations; if chinese mode, also convert single uppercase letters."""
    text = _ACRONYM_RE.sub(
        lambda match: _spaced_letters(match.group(1), acronym_style),
        text,
    )
    if acronym_style == "chinese":
        text = _SINGLE_LETTER_RE.sub(
            lambda match: _spaced_letters(match.group(1), acronym_style),
            text,
        )
    return text


def _normalize_mixed_alnum_tokens(text: str, acronym_style: str) -> str:
    """Read remaining equipment-like tokens character by character (SF6, 6631f5)."""
    return _MIXED_ALNUM_RE.sub(
        lambda m: "".join(
            _DIGITS[int(ch)] if ch.isdigit() else _spaced_letters(ch, acronym_style)
            for ch in m.group(1)
        ),
        text,
    )


def _normalize_remaining_numbers(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", lambda m: _decimal_to_cn(m.group(0)), text)


_COLON_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[:：]\s*(\d+(?:\.\d+)?)")
_DOCUMENT_JOIN_RE = re.compile(r"》\s*(?:、|，|,)?\s*《")


def _normalize_document_pauses(text: str) -> str:
    return _DOCUMENT_JOIN_RE.sub("》、《", text)


def _tidy_spaces(text: str) -> str:
    """Remove spaces that only reflect formatting, not spoken separation."""
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"([.、，。：；])\s+(?=[\u4e00-\u9fff])", r"\1", text)
    text = re.sub(r"([.、，。：；])\s+(?=[《（])", r"\1", text)
    text = re.sub(r"\s+(?=[，。；：、）】》])", "", text)
    text = re.sub(r"(?<=[A-Z0-9])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=(?:小于等于|大于等于|小于|大于))", "", text)
    text = re.sub(r"(小于等于|大于等于|小于|大于)\s+(?=[\u4e00-\u9fff])", r"\1", text)
    return text.strip()


def normalize_power_text(
    text: str,
    language: str = "zh",
    acronym_style: str = "english",
) -> str:
    """Normalize one text segment for the electric-power domain.

    Callers that use ``auto`` should segment with LangSegmenter first and pass
    only the Chinese segments here.
    """
    if not text or not is_supported(language):
        return text

    acronym_style = (acronym_style or "english").strip().lower()
    if acronym_style not in _SUPPORTED_ACRONYM_STYLES:
        acronym_style = "english"

    text = _sanitize_common_text(text)
    protected: list[str] = []
    text = _normalize_common_atomic(text, protected)
    text = _canonicalize_symbols(text, protected)

    # Atomic formats are rendered once and protected from later generic rules.
    text = _normalize_standard_codes(text, protected, acronym_style)
    text = _normalize_codes(text, protected, acronym_style)
    text = _normalize_tables_atomic(text, protected)
    text = _normalize_sections_atomic(text, protected)
    text = _normalize_parameter_chains(text, protected)
    text = _normalize_signed_values(text, protected)
    text = _normalize_rates(text, protected)
    text = _normalize_multipliers(text, protected)
    text = _normalize_ranges_atomic(text, protected)

    text = _normalize_list_markers(text, acronym_style)
    text = _normalize_parenthetical_pairs(text)
    text = _normalize_note_labels(text)
    text = _normalize_dispatch_orders(text)
    # Roman numerals must be converted before "/" so ordinals count as names
    # (Ⅰ段母线 / Ⅱ段母线 -> 一段母线、二段母线).
    text = _normalize_roman(text)
    text = _normalize_slashes(text)
    text = _normalize_special_phrases(text)
    text = _normalize_units(text)
    text = _normalize_hashes_and_equipment(text, acronym_style)
    text = _normalize_percentages(text)
    text = _normalize_permille(text)
    text = _normalize_hyphens(text)
    text = _normalize_pluses(text)
    text = _normalize_pluses(text)  # idempotent pass for adjacent terms
    text = _normalize_times_sign(text)
    text = _normalize_mixed_alnum_tokens(text, acronym_style)
    text = _normalize_acronyms(text, acronym_style)
    text = _normalize_comparators(text)
    text = _normalize_common_symbols(text)
    text = (
        text.replace("±", "正负")
        .replace("～", "至")
        .replace("~", "至")
        .replace("°", "度")
        .replace("％", "%")
        .replace("＝", "等于")
        .replace("=", "等于")
    )
    text = _COLON_RATIO_RE.sub(lambda m: m.group(1) + "比" + m.group(2), text)
    text = _normalize_remaining_numbers(text)
    text = _restore_protected(text, protected)
    text = _normalize_document_pauses(text)
    return _tidy_spaces(text)


def sanitize_text(text: str) -> str:
    """Public common sanitation entry point for pure-English spans."""
    return _sanitize_common_text(text or "")
