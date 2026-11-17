"""evals/sentence_lengths.py: long sentences and the clauses holding them.

The fixture text is built in each test, so no sentence of 2 KB is committed. Only the
tests named so use the test database.
"""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from stratakb.knowledge.ingestion import _parse_file
from stratakb.rag.extractive import MAX_SENTENCE_CHARS
from tests.helpers import REPO_DIR, ingest

SCRIPT = REPO_DIR / "evals" / "sentence_lengths.py"
GOLD_DIR = REPO_DIR / "evals" / "gold"
HEAD = "7.2 Either party may end this agreement "
PATTERN = r"^Identification:\s*(?P<label>AT-[A-Z]?\d{1,3})\b"


def _load():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(chars):
    """A contract whose clause 7.2 is one sentence of chars characters, under heading 7.

    Its sections: the title, 7, 7.1, 7.2 and 8.
    """
    sentence = HEAD + "a" * (chars - len(HEAD) - 1) + "."
    assert len(sentence) == chars
    return (
        "# Agreement\n\nIntro line.\n\n## 7 Termination\n\n7.1 Notice is short.\n\n"
        f"{sentence}\n\n## 8 Payment\n\nInvoices are due within 30 days.\n"
    )


def _folder(tmp_path, files):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


@contextmanager
def _quiet():
    """Keep the script's log lines out of stdout, where its report goes."""
    with patch("stratakb.core.logging.configure_logging"), capture_logs():
        yield


def _report(capsys, *argv):
    with _quiet():
        _load().main([str(a) for a in argv])
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def test_long_sentence_and_its_clauses_are_counted(tmp_path, capsys):
    folder = _folder(tmp_path, {"contract.md": _contract(MAX_SENTENCE_CHARS + 1)})

    report, _ = _report(capsys, "--folder", folder)

    assert report["max_sentence_chars"] == MAX_SENTENCE_CHARS
    [source] = report["sources"]
    assert (source["documents"], source["failed"]) == (1, 0)
    assert (source["sentences"], source["long_sentences"]) == (4, 1)
    assert source["long_pct"] == 25.0
    # 7.2, and the 7 and the title enclosing it; not 7.1 or 8
    assert (source["clauses"], source["clauses_with_long"]) == (5, 3)
    assert source["clauses_with_long_pct"] == 60.0
    assert source["longest"] == {"file": "contract.md", "chars": MAX_SENTENCE_CHARS + 1}


def test_a_sentence_at_the_limit_is_not_long(tmp_path, capsys):
    folder = _folder(tmp_path, {"contract.md": _contract(MAX_SENTENCE_CHARS)})

    [source] = _report(capsys, "--folder", folder)[0]["sources"]

    assert (source["long_sentences"], source["clauses_with_long"]) == (0, 0)
    assert (source["long_pct"], source["clauses_with_long_pct"]) == (0.0, 0.0)
    assert source["longest"]["chars"] == MAX_SENTENCE_CHARS


def test_an_empty_folder_gives_no_percentages(tmp_path, capsys):
    folder = _folder(tmp_path, {})

    [source] = _report(capsys, "--folder", folder)[0]["sources"]

    assert (source["documents"], source["sentences"], source["clauses"]) == (0, 0, 0)
    assert (source["long_pct"], source["clauses_with_long_pct"], source["longest"]) == (
        None, None, None,
    )


def _preset(path, sources="  - name: rules\n    path: rules\n"):
    path.write_text(
        "name: rules\ndisplay_name: Rules\n"
        f"structure_patterns:\n  - '{PATTERN}'\nsources:\n{sources}",
        encoding="utf-8",
    )
    return path


def test_preset_patterns_start_sections(tmp_path, capsys):
    folder = _folder(tmp_path, {
        "rules.txt": "Intro.\n\nIdentification: AT-R051 The name of the thing.\n\nMore text.\n",
    })
    preset = _preset(tmp_path / "preset.yaml")

    [plain] = _report(capsys, "--folder", folder)[0]["sources"]
    [patterned] = _report(capsys, "--folder", folder, "--preset", preset)[0]["sources"]

    assert plain["clauses"] == 0
    assert patterned["clauses"] == 1


def test_a_preset_reads_its_source_folders(tmp_path, capsys):
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "contract.md").write_text(
        _contract(MAX_SENTENCE_CHARS + 1), encoding="utf-8"
    )
    preset = _preset(
        tmp_path / "preset.yaml",
        "  - name: rules\n    path: rules\n  - name: gone\n    path: gone\n",
    )

    report, err = _report(capsys, "--preset", preset)

    assert [(s["name"], s["long_sentences"]) for s in report["sources"]] == [("rules", 1)]
    assert "gone: source folder not found" in err


def test_secret_and_unparsable_files(tmp_path, capsys):
    folder = _folder(tmp_path, {
        "secrets.yaml": "password: hunter2\n",
        "broken.json": "{not json",
        "notes.md": "# Notes\n\nOne sentence.\n",
    })

    with patch("stratakb.knowledge.ingestion._parse_file", wraps=_parse_file) as parse:
        report, err = _report(capsys, "--folder", folder)

    [source] = report["sources"]
    assert (source["documents"], source["failed"]) == (1, 1)
    assert "broken.json" in err
    # The secret-looking file is never opened
    assert sorted(call.args[0].name for call in parse.call_args_list) == [
        "broken.json", "notes.md",
    ]
    assert "secrets" not in err


def test_out_under_the_gold_folder_is_refused(tmp_path, capsys):
    folder = _folder(tmp_path, {"notes.md": "One sentence.\n"})
    target = GOLD_DIR / "batch1" / "report.json"

    with pytest.raises(SystemExit) as e:
        _load().main(["--folder", str(folder), "--out", str(target)])

    assert e.value.code != 0
    assert "frozen" in capsys.readouterr().err
    assert not target.exists()


def test_out_also_writes_the_report(tmp_path, capsys):
    folder = _folder(tmp_path, {"notes.md": "One sentence.\n"})
    out = tmp_path / "report.json"

    report, _ = _report(capsys, "--folder", folder, "--out", out)

    assert json.loads(out.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "no input given"),
        (["--layer", "x", "--folder", "."], "--layer: only with --database"),
        (["--database", "--folder", "."], "--database"),
    ],
    ids=["none", "layer-without-database", "database-and-folder"],
)
def test_inputs_are_checked_before_anything_is_read(argv, message, capsys):
    with pytest.raises(SystemExit) as e:
        _load().main(argv)

    assert e.value.code != 0
    err = capsys.readouterr()
    assert message in err.err
    assert err.out == ""


async def test_database_matches_folder(db_url, layer, tmp_path, capsys):
    folder = _folder(tmp_path, {
        "contract.md": _contract(MAX_SENTENCE_CHARS + 1),
        "notes.md": "# Notes\n\nOne sentence. Another one.\n",
    })
    await ingest(db_url, layer, folder)
    capsys.readouterr()
    script = _load()

    [from_folder] = _report(capsys, "--folder", folder)[0]["sources"]
    [from_database] = await script.read_database(db_url, [layer])

    keys = ["documents", "sentences", "long_sentences", "clauses", "clauses_with_long"]
    assert [from_database[k] for k in keys] == [from_folder[k] for k in keys]
    assert from_database["long_sentences"] == 1
    assert from_database["name"] == layer
    assert from_database["chunks_without_document"] == 0


def test_database_unknown_layer_stops(db_url, capsys):
    """Sync: main runs its own event loop, on the database in DATABASE_URL."""
    with _quiet(), pytest.raises(SystemExit) as e:
        _load().main(["--database", "--layer", "no_such_layer"])

    assert e.value.code != 0
    assert "no such layer: no_such_layer" in str(e.value.code)
    assert capsys.readouterr().out == ""
