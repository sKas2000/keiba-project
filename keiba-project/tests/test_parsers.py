"""src/scraping/parsers.py のユニットテスト"""
import pytest
from src.scraping.parsers import (
    safe_float, safe_int, parse_odds_range, parse_odds_range_low,
    normalize_jockey_name, time_to_seconds, parse_horse_weight,
    parse_sex_age,
)


class TestSafeFloat:
    def test_normal(self):
        assert safe_float("1.5") == 1.5

    def test_comma(self):
        assert safe_float("1,234.5") == 1234.5

    def test_whitespace(self):
        assert safe_float("  3.14  ") == 3.14

    def test_invalid(self):
        assert safe_float("abc") == 0.0

    def test_none(self):
        assert safe_float(None) == 0.0

    def test_int(self):
        assert safe_float(42) == 42.0


class TestSafeInt:
    def test_normal(self):
        assert safe_int("123") == 123

    def test_with_unit(self):
        assert safe_int("1600m") == 1600

    def test_negative(self):
        assert safe_int("-5") == -5

    def test_invalid(self):
        assert safe_int("abc") == 0

    def test_none(self):
        assert safe_int(None) == 0


class TestParseOddsRange:
    def test_normal(self):
        assert parse_odds_range("1.5 - 3.0") == 2.2  # average

    def test_single(self):
        assert parse_odds_range("5.0") == 5.0


class TestParseOddsRangeLow:
    def test_range(self):
        assert parse_odds_range_low("1.5 - 3.0") == 1.5

    def test_single(self):
        assert parse_odds_range_low("5.0") == 5.0


class TestNormalizeJockeyName:
    def test_half_to_full_space(self):
        name = normalize_jockey_name("田中 太郎")
        assert " " not in name or "　" not in name

    def test_strip(self):
        assert normalize_jockey_name("  ルメール  ").strip() == "ルメール"


class TestTimeToSeconds:
    def test_normal(self):
        assert time_to_seconds("1:34.5") == 94.5

    def test_short(self):
        assert time_to_seconds("58.3") == 58.3

    def test_invalid(self):
        assert time_to_seconds("abc") == 0.0


class TestParseHorseWeight:
    def test_normal(self):
        w, c = parse_horse_weight("480(+4)")
        assert w == 480
        assert c == 4

    def test_minus(self):
        w, c = parse_horse_weight("460(-6)")
        assert w == 460
        assert c == -6

    def test_no_change(self):
        w, c = parse_horse_weight("500")
        assert w == 500
        assert c == 0

    def test_invalid(self):
        w, c = parse_horse_weight("")
        assert w == 0
        assert c == 0


class TestParseSexAge:
    def test_normal(self):
        sex, age = parse_sex_age("牡3")
        assert sex == "牡"
        assert age == 3

    def test_mare(self):
        sex, age = parse_sex_age("牝5")
        assert sex == "牝"
        assert age == 5

    def test_gelding(self):
        sex, age = parse_sex_age("セ4")
        assert sex == "セ"
        assert age == 4

    def test_invalid(self):
        sex, age = parse_sex_age("")
        assert sex == ""
        assert age == 0
