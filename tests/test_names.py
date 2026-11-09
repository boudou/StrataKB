"""The names a question asks about, and whether the passages hold them (pure)."""

from __future__ import annotations

import pytest

from stratakb.rag.names import (
    MAX_NAMES,
    NameCheck,
    check_names,
    find_names,
    missing_names_note,
)

IBAN = "DE89 3704 0044 0532 0130 00"


def _names(question: str) -> list[str]:
    return [n.text for n in find_names(question)]


def _missing(question: str, *texts: str) -> tuple[str, ...]:
    return check_names(find_names(question), texts).missing


class TestFindNames:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What does the C function PyConfig_Merge do?", ["PyConfig_Merge"]),
            ("What is the `__match_kwargs__` attribute used for?", ["__match_kwargs__"]),
            ("How does asyncio.TaskPool limit concurrency?", ["asyncio.TaskPool"]),
            ("What does PEP 9123 propose?", ["PEP 9123"]),
            ('What does config["timeout"] set?', ["timeout"]),
            ("What does AC04 mean?", ["AC04"]),
            ("what does AT-44 say", ["AT-44"]),
            ("Which fields does pacs.008 require?", ["pacs.008"]),
            ("How does the TaskPool work?", ["TaskPool"]),
            ("Is RFC 2119 cited by ISO 20022?", ["RFC 2119", "ISO 20022"]),
            ("What does PYTHONQUILLMODE do?", ["PYTHONQUILLMODE"]),
            ("How do I add a HEALTHCHECK to a Dockerfile?", ["HEALTHCHECK"]),
            ("Does PEP 8 say HTTP or PYTHONHASHSEED?", ["PEP 8", "PYTHONHASHSEED"]),
            ("PYTHONQUILLMODE?", ["PYTHONQUILLMODE"]),
            ("Is PYTHONPATH read before PYTHONHOME?", ["PYTHONPATH", "PYTHONHOME"]),
        ],
    )
    def test_names_are_found(self, question, expected):
        assert _names(question) == expected

    @pytest.mark.parametrize(
        "question",
        [
            "What is the maximum line length?",
            "Why was it rejected?",
            "What does clause 2.2 say?",
            "What does Article 5 say about 7.2.1?",
            "e.g. in the U.S. in Q3",
            "see [S1]",
            "x86 and v2 on A4 paper",
            "What does SEPA say about fees?",
            "Does HTTP send JSON to the API?",
            "Is the WSGI server told the DATE?",
            "HOW DO I SET PYTHONQUILLMODE?",
            f"why was {IBAN} rejected?",
            f"why was {IBAN.replace(' ', '')} rejected?",
        ],
    )
    def test_ordinary_words_are_not_names(self, question):
        assert _names(question) == []

    def test_a_dotted_name_does_not_take_the_full_stop(self):
        assert _names("Use asyncio.TaskPool.") == ["asyncio.TaskPool"]

    def test_names_are_kept_once_as_first_written(self):
        assert _names("Is PEP 8 the same as PEP8 or TaskPool as taskPool?") == [
            "PEP 8", "TaskPool"
        ]

    def test_names_come_in_question_order(self):
        assert _names("Does PEP 9123 change asyncio.TaskPool or AC04?") == [
            "PEP 9123", "asyncio.TaskPool", "AC04"
        ]

    def test_at_most_max_names(self):
        question = " ".join(f"name_{i}" for i in range(MAX_NAMES + 5))
        assert len(find_names(question)) == MAX_NAMES


class TestCheckNames:
    @pytest.mark.parametrize("text", ["see PEP8", "pep-0008 says", "PEP: 8 says", "PEP 8"])
    def test_numbered_reference_in_any_spelling(self, text):
        assert _missing("What does PEP 8 say?", text) == ()

    def test_numbered_reference_needs_the_same_number(self):
        assert _missing("What does PEP 8 say?", "PEP 80 says") == ("PEP 8",)

    def test_case_and_spacing_do_not_count(self):
        assert _missing("How does the TaskPool work?", "a taskpool runs") == ()
        assert _missing("How does the TaskPool work?", "a Task Pool runs") == ()

    def test_dotted_name_found_by_its_named_last_part(self):
        assert _missing("What is typing.TYPE_CHECKING?", "if TYPE_CHECKING:") == ()

    def test_dotted_name_needs_its_named_part(self):
        assert _missing("How does asyncio.TaskPool work?", "asyncio.Task runs") == (
            "asyncio.TaskPool",
        )

    def test_dotted_name_found_by_a_tail(self):
        assert _missing("What is os.path.join?", "use path.join here") == ()

    def test_a_file_path_counts(self):
        assert _missing("What does PEP 8 say?", "Style guide text.", "peps/pep-0008.txt") == ()

    def test_bracketed_key(self):
        assert _missing('What does config["timeout"] set?', "the timeout key") == ()

    def test_one_present_one_missing(self):
        check = check_names(find_names("Does AC04 differ from AC99?"), ["AC04 means closed"])
        assert check.names == ("AC04", "AC99")
        assert check.missing == ("AC99",)
        assert not check.none_present

    def test_none_present(self):
        check = check_names(find_names("What is PyConfig_Merge?"), ["PyConfig_Read"])
        assert check.none_present

    def test_no_names_is_never_none_present(self):
        check = check_names([], ["text"])
        assert check == NameCheck((), ())
        assert not check.none_present


class TestNote:
    def test_nothing_missing_is_no_note(self):
        assert missing_names_note(NameCheck(("A_b",), ()), related=True) is None

    def test_note_names_the_missing_in_order(self):
        check = NameCheck(("a_b", "C_d", "e_f"), ("a_b", "e_f"))
        assert missing_names_note(check, related=False) == "Not in any passage: `a_b`, `e_f`."

    def test_related_note(self):
        note = missing_names_note(NameCheck(("a_b",), ("a_b",)), related=True)
        assert note == (
            "Not in any passage: `a_b`. The passages are related text, not the answer."
        )
