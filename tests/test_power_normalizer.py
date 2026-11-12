"""Regression tests for the sherpa TTS preprocessing pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "model_app"
sys.path.insert(0, str(MODEL_DIR))

from power_normalizer import normalize_power_text  # noqa: E402
from text_routing import normalize_domain_text  # noqa: E402


class CommonNormalizerTests(unittest.TestCase):
    def assert_spoken(self, source: str, expected: str) -> None:
        self.assertEqual(normalize_domain_text(source), expected)

    def test_clock_time(self) -> None:
        cases = {
            "8：35": "八点三十五分",
            "8:35": "八点三十五分",
            "08：05": "八点零五分",
            "8:35:20": "八点三十五分二十秒",
            "8：35至12：30": "八点三十五分至十二点三十分",
            "8:35-12:30": "八点三十五分至十二点三十分",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assert_spoken(source, expected)

    def test_date_and_datetime(self) -> None:
        cases = {
            "2026年11月12日": "二零二六年十一月十二日",
            "2026-11-12": "二零二六年十一月十二日",
            "2026/11/12 8:35": "二零二六年十一月十二日八点三十五分",
            "11月12日8时35分": "十一月十二日八时三十五分",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assert_spoken(source, expected)

    def test_ratio_is_not_treated_as_time(self) -> None:
        self.assert_spoken("1:2", "一比二")
        self.assert_spoken("变比10：1", "变比十比一")

    def test_power_units_and_symbols(self) -> None:
        cases = {
            "220kV线路": "二百二十千伏线路",
            "A相电流7.594A": "诶相电流七点五九四安",
            "AC380V": "交流三百八十伏",
            "DC220V": "直流二百二十伏",
            "30%": "百分之三十",
            "±1.5%": "正负百分之一点五",
            "5‰": "千分之五",
            "故障测距3.012km": "故障测距三点零一二千米",
            "容抗2MΩ": "容抗二兆欧",
            "B相6.8A。": "必相六点八安。",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assert_spoken(source, expected)

    def test_equipment_and_standard_identifiers(self) -> None:
        cases = {
            "红岭变 1198F6 红荷线": "红岭变一一九八艾弗六红荷线",
            "#2主变": "二号主变",
            "相间距离I段": "相间距离一段",
            "10kV母线Ⅰ段": "十千伏母线一段",
            "Q/GDW 1234-2020": "丘杠记迪达不留一二三四杠二零二零",
            "第5.2.1条": "第五点二点一条",
            "SF6气压低": "矮丝艾弗六气压低",
            "开关6631f5": "开关六六三一艾弗五",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assert_spoken(source, expected)

    def test_telephone_and_entities(self) -> None:
        self.assert_spoken("电话13800138000。", "电话一三八零零一三八零零零。")
        self.assert_spoken(
            "座机010-12345678转801。",
            "座机零一零，一二三四五六七八转八零一。",
        )
        self.assert_spoken("&#x20;红岭变开关跳闸。", "红岭变开关跳闸。")

    def test_english_sentence_is_unchanged(self) -> None:
        source = "Please meet at 8:35 and check 220kV."
        self.assert_spoken(source, source)

    def test_power_normalizer_is_idempotent_for_spoken_output(self) -> None:
        source = "2026年11月12日8：35，红岭变220kV线路A相电流7.594A。"
        once = normalize_power_text(source, acronym_style="chinese")
        twice = normalize_power_text(once, acronym_style="chinese")
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
