"""Tests for the verify checks: quotes, facts and citations, without a database."""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from stratakb.rag.verify import (
    MAX_FLAGS,
    CheckResult,
    check,
    extract_facts,
    normalise_text,
    verdict,
)
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
        assert verdict(result) == "verified"
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
        assert flags[0].message == '"1500 EUR" not found in S2'

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

    def test_citation_of_passage_zero_keeps_its_range(self):
        flags = check('"a quote that is absent" [S0, S1, S2]', PASSAGES).flags
        quote = next(f for f in flags if f.kind == "quote")
        assert "S0–S2" in quote.message

    def test_citation_range_and_list(self):
        assert _flag_texts("Fees apply [S1-S4] and [S2, S3].") == []
        assert _flag_texts("Fees apply [S3, S6].") == ["S6"]

    def test_quote_checked_against_cited_passage_only(self):
        # The quote is in S1 but cites S2
        draft = '"Either party may end this agreement" [S2]'
        flags = check(draft, PASSAGES).flags
        assert [(f.kind, f.passages) for f in flags] == [("quote", ["S2"])]

    def test_uncited_quote_searches_every_passage(self):
        # The draft cites elsewhere, so the quote with no citation of its own may be anywhere
        assert _flag_texts('"the late rate is 5%". Either party may end it [S1].') == []

    def test_no_citations_flagged_even_when_everything_matches(self):
        result = check('"the late rate is 5%" and the fee is EUR 1,000.00.', PASSAGES)
        assert [(f.kind, f.text) for f in result.flags] == [("citation", "no citation")]
        assert result.flags[0].passages == list(PASSAGES)
        assert result.quotes_checked == 1
        assert result.facts_checked == 1
        assert verdict(result) == "flagged"

    def test_no_citation_flag_listed_first_past_the_cap(self):
        draft = " ".join(f"Q{i}Z" for i in range(1, MAX_FLAGS + 20))
        result = check(draft, PASSAGES)
        assert result.flags[0].text == "no citation"
        assert len(result.flags) == MAX_FLAGS
        assert result.flags_unlisted == 20

    def test_citation_inside_a_quote_does_not_count(self):
        # The [S1] is the document's own text, not a citation by the draft
        passages = {"S1": "See [S1] for the form."}
        assert [f.text for f in check('"See [S1] for the form"', passages).flags] == ["no citation"]
        result = check('"See [S1] for the form" [S1]', passages)
        assert result.flags == []
        assert verdict(result) == "verified"

    @pytest.mark.parametrize(
        "draft",
        [
            "Termination also requires approval from the board of directors [S1].",
            '"..." [S1]',
            "[S1]",
        ],
    )
    def test_cited_draft_with_nothing_to_check(self, draft):
        result = check(draft, PASSAGES)
        assert result.flags == []
        assert verdict(result) == "nothing_checked"

    @pytest.mark.parametrize("draft", ["The moon is made of cheese.", "", "   "])
    def test_draft_with_no_citations_and_nothing_to_check_flagged(self, draft):
        result = check(draft, PASSAGES)
        assert [(f.kind, f.text) for f in result.flags] == [("citation", "no citation")]
        assert result.quotes_checked == 0
        assert result.facts_checked == 0
        assert verdict(result) == "flagged"

    def test_quote_with_omission(self):
        assert _flag_texts('"A fee of EUR 1,000.00 ... is reported in pacs.002" [S2]') == []

    def test_quote_of_fragments_flagged(self):
        # Single-word fragments around an omission prove nothing
        assert _flag_texts('"A ... applies ... 5%" [S2]') == ["A ... applies ... 5%"]

    def test_quote_part_inside_a_word_flagged(self):
        # "EUR 1" stops inside "1,000.00", "ee of" starts inside "fee of"
        assert _flag_texts('"A fee of EUR 1" [S2]') == ["A fee of EUR 1"]
        assert _flag_texts('"ee of EUR 1,000.00 applies" [S2]') == ["ee of EUR 1,000.00 applies"]

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
        draft = " ".join(f"Q{i}Z" for i in range(1, 10_000))[:MAX_DRAFT_CHARS - 5] + " [S1]"
        passages = {f"S{i}": "A fee of EUR 25 applies within 30 days. " * 50 for i in range(1, 51)}
        result = check(draft, passages)
        assert len(result.flags) == MAX_FLAGS
        assert result.flags_unlisted == result.facts_checked - MAX_FLAGS
        assert verdict(result) == "flagged"

    @pytest.mark.parametrize(
        "unit",
        ["one ", "twenty one ", "two and ", "three-", "five (5 ", "one" + " " * 40],
    )
    def test_long_run_of_number_words_is_fast(self, unit):
        # An unbounded run of number words backtracked quadratically (20 s at 22k chars)
        draft = (unit * (MAX_DRAFT_CHARS // len(unit) + 1))[:MAX_DRAFT_CHARS] + "("
        started = time.perf_counter()
        extract_facts(draft)
        assert time.perf_counter() - started < 2.0

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


class TestFactScope:
    """A sentence's facts are checked against the passages cited in that sentence."""

    P = {
        "S1": "Either party may terminate with ninety (90) days' written notice under Article 12.",
        "S2": "Refunds take thirty days. A fee of EUR 25 applies from 4 November 2026.",
        "S3": "Clause 7.2 sets the rate at 5%.",
    }
    ALL = ["S1", "S2", "S3"]

    @pytest.mark.parametrize(
        ("draft", "expected"),
        [
            (
                "Either party may terminate with 30 days' written notice [S1].",
                [("30 days", ["S1"])],
            ),
            ("Either party may terminate with 90 days' written notice [S1].", []),
            # No citation in the sentence: any passage given will do
            ("Refunds take 30 days. Either party may end it [S1].", []),
            ("Either party may end it [S1]. Refunds take 13 days.", [("13 days", ALL)]),
            # Several citations in one sentence: either passage will do
            ("Notice is 90 days [S2] and the fee is EUR 25 [S1].", []),
            # A citation right after the end belongs to the sentence before it
            ("Notice is 30 days. [S1] The fee is EUR 25. [S2]", [("30 days", ["S1"])]),
            # ... unless a lower-case word follows it
            (
                "[S1] says the fee is EUR 25. [S2] says notice is 90 days.",
                [("EUR 25", ["S1"]), ("90 days", ["S2"])],
            ),
            # Line breaks end sentences; a citation stays with its own line
            ("Intro:\n[S1] Notice is 90 days.\n[S2] Fee is EUR 25.", []),
            # ... or goes with the line before it when alone on its line
            ("Notice is 90 days\n[S1]\nFee is EUR 25\n[S2]", []),
            ("- Notice is 90 days [S1]\n- Fee is EUR 25 [S2]\n- Rate is 5% [S3]", []),
            ("| Notice | 90 days | [S1] |\n| Fee | EUR 25 | [S2] |", []),
            # A period before a digit does not end a sentence
            ("Under Art. 13 notice is 90 days [S1].", [("Art. 13", ["S1"])]),
            ("The fee starts on Nov. 4 2026 [S1].", [("Nov. 4 2026", ["S1"])]),
            # End punctuation inside a quote ends the sentence
            (
                'It reads "Refunds take thirty days." [S2] Notice is 30 days [S1].',
                [("30 days", ["S1"])],
            ),
            ('It reads "Refunds take thirty days" [S2] and is 90 days.', [("90 days", ["S2"])]),
            # A digit after a citation starts a new sentence
            (
                "Refunds take 90 days [S2]. 30 days' notice is needed [S1].",
                [("90 days", ["S2"]), ("30 days", ["S1"])],
            ),
            # A citation inside a quote is the document's text
            (
                '"See [S1] for it", 90 days [S2].',
                [("See [S1] for it", ALL), ("90 days", ["S2"])],
            ),
            # A passage that was not given: the citation flag and the fact flag
            ("EUR 25 [S9].", [("S9", ALL), ("EUR 25", ["S9"])]),
        ],
    )
    def test_sentence_scope(self, draft, expected):
        flags = check(draft, self.P).flags
        assert [(f.text, f.passages) for f in flags] == expected

    def test_message_names_the_cited_passage(self):
        flags = check("Either party may terminate with 30 days' notice [S1].", self.P).flags
        assert flags[0].message == '"30 days" not found in S1'

    def test_no_citation_flags_with_every_id(self):
        flags = check("It ends [S1]. Refunds take 13 days.", self.P).flags
        assert flags[0].message == '"13 days" not found in S1–S3'
        assert flags[0].passages == self.ALL

    def test_same_fact_in_two_scopes_is_flagged_for_each(self):
        result = check("13 days [S1]. 13 days [S2].", self.P)
        assert result.facts_checked == 1
        assert [(f.text, f.passages) for f in result.flags] == [
            ("13 days", ["S1"]),
            ("13 days", ["S2"]),
        ]

    def test_repeat_in_one_scope_is_checked_once(self):
        assert check("It is 30 days [S1], 30 days, thirty days [S1].", PASSAGES).facts_checked == 1

    def test_citations_on_lines_of_their_own(self):
        draft = "Notice is 90 days. [S2]\n[S1]\nFee is EUR 25.\n[S2]\n[S3]."
        assert verdict(check(draft, self.P)) == "verified"

    def test_citation_followed_by_more_text_opens_the_next_clause(self):
        flags = check("Notice is 90 days. [S2], the fee is EUR 25 [S1].", self.P).flags
        assert flags == []
        # [S2] opens the second clause, so the first sentence cites nothing
        flags = check("Notice is 13 days. [S2], the fee is EUR 25 [S2].", self.P).flags
        assert [(f.text, f.passages) for f in flags] == [("13 days", self.ALL)]

    def test_period_after_trailing_citation_goes_with_it(self):
        flags = check("Notice is 30 days. [S1]. The fee is EUR 25.", self.P).flags
        assert [(f.text, f.passages) for f in flags] == [("30 days", ["S1"])]

    def test_unusable_cited_passage_is_not_searched(self):
        # S2 was given but its text could not be used; S1 alone is searched
        flags = check("Notice is 30 days [S1, S2].", {"S1": self.P["S1"]}, ["S1", "S2"]).flags
        assert [(f.text, f.passages) for f in flags] == [("30 days", ["S1", "S2"])]

    def test_many_sentences_is_fast(self):
        draft = ("x. [S1] " * (MAX_DRAFT_CHARS // 8))[:MAX_DRAFT_CHARS]
        started = time.monotonic()
        check(draft, self.P)
        assert time.monotonic() - started < 2

    def test_run_of_end_marks_is_fast(self):
        started = time.monotonic()
        check("." * MAX_DRAFT_CHARS + "x", self.P)
        assert time.monotonic() - started < 2

    def test_many_facts_under_many_cited_ids_is_fast(self):
        # One sentence of distinct numbers citing tens of thousands of ids
        half = MAX_DRAFT_CHARS // 2
        numbers = " ".join(str(n) for n in range(10000, 10000 + half // 6))[:half]
        markers = " ".join(f"[S{n}-S{n + 50}]" for n in range(1000, 1000 + 51 * half // 14, 51))
        passages = {f"S{n}": f"Passage {n} has no such number." for n in range(1, 51)}
        draft = f"{numbers} {markers}"[:MAX_DRAFT_CHARS]
        started = time.monotonic()
        result = check(draft, passages)
        assert time.monotonic() - started < 2
        assert len(result.flags) == MAX_FLAGS
        assert result.flags_unlisted > 1000


class TestUncheckedSentences:
    """Sentences with no quote and no fact are listed, and leave the verdict alone."""

    P = {"S1": "The fee is EUR 25 from 2026-11-04."}

    def test_moon_sentence_listed(self):
        draft = "The moon is made of cheese. The fee is EUR 25 [S1]."
        result = check(draft, {"S1": "The fee is EUR 25."})
        assert result.flags == []
        assert result.quotes_checked == 0
        assert result.facts_checked == 1
        assert result.unchecked_sentences == ["The moon is made of cheese."]
        assert verdict(result) == "verified"

    def test_covered_draft_lists_none(self):
        result = check("The fee is EUR 25 [S1]. It applies from 2026-11-04 [S1].", self.P)
        assert result.unchecked_sentences == []
        assert result.unchecked_unlisted == 0

    def test_repeated_fact_covers_each_sentence(self):
        # The second sentence's fact is not checked again, but it was compared
        result = check("The fee is EUR 25 [S1]. Again, EUR 25 [S1].", self.P)
        assert result.unchecked_sentences == []

    def test_quote_covers_its_sentence(self):
        result = check('"The fee is EUR 25" [S1]. Nothing else here [S1].', self.P)
        assert result.unchecked_sentences == ["Nothing else here [S1]."]

    def test_empty_quote_does_not_cover(self):
        result = check('See "..." here [S1]. The fee is EUR 25 [S1].', self.P)
        assert result.unchecked_sentences == ['See "..." here [S1].']

    def test_citation_only_line_not_listed(self):
        assert check("The fee is EUR 25.\n[S1]", self.P).unchecked_sentences == []

    def test_list_is_capped(self):
        draft = "\n".join(["No fact here."] * 60 + ["The fee is EUR 25 [S1]."])
        result = check(draft, self.P)
        assert result.unchecked_sentences == ["No fact here."] * MAX_FLAGS
        assert result.unchecked_unlisted == 10
        assert verdict(result) == "verified"

    def test_verdict_counts_outside_flags(self):
        result = check("The fee is EUR 25 [S1].", self.P)
        assert verdict(result) == "verified"
        assert verdict(result, extra_flags=1) == "flagged"

    @pytest.mark.parametrize("draft", ["The moon is made of cheese.", "No facts here [S1]."])
    def test_nothing_checked_lists_none(self, draft):
        # Every sentence is unchecked; listing them would add nothing
        assert check(draft, self.P).unchecked_sentences == []

    def test_check_result_has_no_verdict(self):
        assert not hasattr(CheckResult, "verified")
        assert not hasattr(CheckResult, "nothing_checked")
