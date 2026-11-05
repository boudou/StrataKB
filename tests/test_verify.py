"""Tests for the verify checks: quotes, facts and citations, without a database."""

from __future__ import annotations

from decimal import Decimal

import pytest

from stratakb.rag.verify import MAX_FLAGS, check, extract_facts, normalise_text
from stratakb.rag.verify_tools import MAX_DRAFT_CHARS

PASSAGES = {
    "S1": (
        "7.2 Termination\n"
        "Either party may end this agreement with thirty (30) days' written notice."
    ),
    "S2": (
        "Article 12 Fees\n"
        "A fee of EUR 1,000.00 applies from 4 November 2026. Reject reason AC01\n"
        "is reported in pacs.002; the late rate is 5%."
    ),
    "S3": "Under clause 7(2)(b) the account GR1601101250000000012300695 is checked.",
    "S4": "Requests are answered within 2 business days.",
}

CLEAN = (
    "Either party may end the agreement with 30 days' notice under section 7.2 [S1]: "
    '"Either party may end this agreement with thirty (30) days’ written notice." [S1] '
    "A fee of 1000 EUR applies from 2026-11-04 (Article 12), and AC01 is reported in "
    "pacs.002 with a late rate of 5 percent [S2]. Clause 7(2)(b) checks "
    "GR1601101250000000012300695 [S3]. Requests take two business days [S4]."
)


def _flag_texts(draft: str) -> list[str]:
    return [f.text for f in check(draft, PASSAGES).flags]


class TestCheck:
    def test_clean_draft_verified(self):
        result = check(CLEAN, PASSAGES)
        assert result.flags == []
        assert result.verified
        assert result.quotes_checked == 1
        assert result.facts_checked >= 10

    def test_quote_digit_flagged(self):
        draft = CLEAN.replace("thirty (30) days’ written", "thirty (31) days’ written")
        flags = check(draft, PASSAGES).flags
        assert len(flags) == 1
        assert flags[0].kind == "quote"
        assert "thirty (31) days" in flags[0].text
        assert flags[0].passages == ["S1"]
        assert flags[0].message.startswith('Quote "Either party')

    def test_prose_amount_flagged(self):
        flags = check(CLEAN.replace("A fee of 1000 EUR", "A fee of 1500 EUR"), PASSAGES).flags
        assert len(flags) == 1
        assert flags[0].kind == "amount"
        assert flags[0].text == "1500 EUR"
        assert flags[0].message == '"1500 EUR" not found in S1–S4'

    def test_prose_days_flagged(self):
        draft = CLEAN.replace("with 30 days' notice", "with 13 days' notice")
        assert _flag_texts(draft) == ["13 days"]

    def test_clause_number_flagged(self):
        flags = check(CLEAN.replace("(Article 12)", "(Article 13)"), PASSAGES).flags
        assert len(flags) == 1
        assert flags[0].kind == "clause"
        assert flags[0].text == "Article 13"

    def test_clause_heading_label_matches(self):
        # "section 7.2" is found through the passage's heading "7.2 Termination"
        assert _flag_texts("See section 7.2 [S1].") == []
        assert _flag_texts("See section 7.3 [S1].") == ["section 7.3"]

    def test_unknown_passage_id_flagged(self):
        flags = check(CLEAN + " See also [S5].", PASSAGES).flags
        assert len(flags) == 1
        assert flags[0].kind == "citation"
        assert flags[0].text == "S5"
        assert "S1–S4" in flags[0].message

    def test_citation_range_and_list(self):
        assert _flag_texts("Fees apply [S1-S4] and [S2, S3].") == []
        assert _flag_texts("Fees apply [S3, S6].") == ["S6"]

    def test_quote_checked_against_cited_passage_only(self):
        # The quote is in S1 but cites S2
        draft = '"Either party may end this agreement" [S2]'
        flags = check(draft, PASSAGES).flags
        assert [(f.kind, f.passages) for f in flags] == [("quote", ["S2"])]

    def test_uncited_quote_searches_every_passage(self):
        assert _flag_texts('"the late rate is 5%"') == []

    def test_quote_with_omission(self):
        assert _flag_texts('"A fee of EUR 1,000.00 ... is reported in pacs.002" [S2]') == []

    def test_quote_across_line_break_and_curly_marks(self):
        draft = "“Reject reason AC01 is reported in pacs.002” [S2]"
        assert _flag_texts(draft) == []

    def test_identifier_number_not_counted_twice(self):
        # pacs.008 is one identifier, not also the number 8
        flags = check("Use pacs.008 [S2].", PASSAGES).flags
        assert [(f.kind, f.text) for f in flags] == [("identifier", "pacs.008")]

    def test_changed_code_flagged(self):
        assert _flag_texts("The reason is AC04 [S2].") == ["AC04"]

    def test_unit_matters(self):
        assert _flag_texts("Requests take 2 days [S4].") == ["2 days"]

    def test_repeated_fact_flagged_once(self):
        assert _flag_texts("It is 13 days [S1], yes 13 days [S1].") == ["13 days"]

    def test_unusable_passage_still_given(self):
        # S2 was given but its text could not be used: not an unknown citation
        result = check("Fees [S2].", {"S1": PASSAGES["S1"]}, given=["S1", "S2"])
        assert result.flags == []

    @pytest.mark.parametrize("wrong", ["EUR 30", "EUR 12", "EUR 2026"])
    def test_wrong_amount_equal_to_a_day_count_clause_or_year_flagged(self, wrong):
        passages = {
            "S1": "A fee of EUR 25 applies from 1 March 2026. The payer is notified "
            "within 30 days under Article 12."
        }
        assert check("The fee is EUR 25 [S1].", passages).flags == []
        assert [f.text for f in check(f"The fee is {wrong} [S1].", passages).flags] == [wrong]

    def test_amount_written_apart_from_its_currency_matches(self):
        passages = {"S1": "The amount, 1,000 (EUR), is due."}
        assert check("A sum of EUR 1,000.00 is due [S1].", passages).flags == []

    def test_numbered_list_markers_are_not_facts(self):
        passages = {"S1": "Notice is ninety (90) days.", "S2": "A fee of EUR 250 applies."}
        draft = "Key points:\n1. The fee is EUR 250 [S2].\n2) Notice is ninety days [S1]."
        assert check(draft, passages).flags == []
        # A wrong fact on a list line is still flagged
        draft = "Key points:\n1. The fee is EUR 260 [S2].\n2. Notice is ninety days [S1]."
        assert [f.text for f in check(draft, passages).flags] == ["EUR 260"]

    @pytest.mark.parametrize("wrong", ["EUR 1", "EUR 2", "EUR 3"])
    def test_passage_list_markers_are_not_amounts(self, wrong):
        passages = {
            "S1": "Fees:\n1. A fee of EUR 25 applies.\n2. A second fee of EUR 40 applies.\n"
            "3. Refunds take 10 days."
        }
        assert check("The fee is EUR 25 [S1].", passages).flags == []
        assert [f.text for f in check(f"The fee is {wrong} [S1].", passages).flags] == [wrong]

    def test_passage_list_marker_still_backs_a_heading_number(self):
        passages = {"S1": "12. Termination\nEither party may end the contract."}
        assert check("See section 12 [S1].", passages).flags == []

    def test_large_hostile_draft_bounded(self):
        draft = " ".join(f"Q{i}Z" for i in range(1, 10_000))[:MAX_DRAFT_CHARS]
        passages = {f"S{i}": "A fee of EUR 25 applies within 30 days. " * 50 for i in range(1, 51)}
        result = check(draft, passages)
        assert len(result.flags) == MAX_FLAGS
        assert result.flags_unlisted == result.facts_checked - MAX_FLAGS
        assert not result.verified

    def test_repeated_fact_checked_once(self):
        result = check("It is 30 days [S1], 30 days, thirty days [S1].", PASSAGES)
        assert result.flags == []
        assert result.facts_checked == 1


class TestFormats:
    """Both sides are normalised, so these pairs match each other."""

    @pytest.mark.parametrize(
        ("draft", "passage"),
        [
            ("30 days", "thirty (30) days"),
            ("thirty (30) days", "30 days"),
            ("thirty days", "30 days"),
            ("a 30-day notice", "30 days"),
            ("30 calendar days", "30 days"),
            ("5 working days", "five (5) business days"),
            ("1,000.00", "1000"),
            ("1000", "1,000.00"),
            ("EUR 1,000.00", "1000 EUR"),
            ("EUR 1.000,00", "EUR 1,000"),
            ("€250", "EUR 250.00"),
            ("an amount of EUR 1000", "the amount, 1,000 (EUR), is due"),
            ("2026-11-04", "4 November 2026"),
            ("4 November 2026", "2026-11-04"),
            ("November 4, 2026", "4th of November 2026"),
            ("4 Nov", "2026-11-04"),
            ("November 2026", "4 November 2026"),
            ("in 2026", "2026-11-04"),
            ("5%", "5 per cent"),
            ("12.5 percent", "12.5%"),
            ("Article 12", "ARTICLE 12"),
            ("7(2)(b)", "clause 7(2)(b)"),
            ("GR1601101250000000012300695", "GR16 0110 1250 0000 0001 2300 695"),
            ("pacs.008.001.08", "the pacs.008.001.08 message"),
            ("04/11/2026", "on 04/11/2026"),
        ],
    )
    def test_formats(self, draft, passage):
        assert check(f"{draft} [S1]", {"S1": passage}).flags == []

    @pytest.mark.parametrize(
        ("draft", "passage"),
        [
            ("31 days", "thirty (30) days"),
            ("1,000.01", "1000"),
            ("USD 1,000", "EUR 1,000"),
            ("2026-11-05", "4 November 2026"),
            # dd/mm and mm/dd are not guessed: only the literal text matches
            ("11/04/2026", "04/11/2026"),
            ("6%", "5 per cent"),
            ("Article 12", "Article 1"),
        ],
    )
    def test_formats_that_differ(self, draft, passage):
        assert len(check(f"{draft} [S1]", {"S1": passage}).flags) == 1

    def test_canonical_values(self):
        keys = [f.key for f in extract_facts("thirty (30) days, EUR 1.000,00 on 4 Nov 2026")]
        assert keys == [
            ("quantity", "day", Decimal(30)),
            ("amount", "EUR", Decimal(1000)),
            ("date", "2026-11-04"),
        ]

    def test_words_alone_are_not_facts(self):
        # "one" of the parties is not a number to check
        assert extract_facts("one of the parties may end it") == []

    def test_month_names_are_capitalised(self):
        # "may 5" is a verb and a number, not a date
        assert [f.kind for f in extract_facts("the payer may 5 times retry")] == ["number"]
        assert [f.kind for f in extract_facts("on May 5")] == ["date"]

    def test_normalise_text(self):
        assert normalise_text("“A  b” – c…\n d") == '"A b" - c... d'
        assert normalise_text("a  b\n\n c", keep_lines=True) == "a b\nc"
