"""The frozen gold set under evals/ and the scripts beside it.

None of these tests needs a database or an embedding model: the checker runs on
stub PEP files, the scorer on a stub server.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import REPO_DIR

EVALS = REPO_DIR / "evals"
GOLD_DIR = EVALS / "gold"
CHECKER = EVALS / "check_gold.py"
SCORER = EVALS / "score_gold.py"

# sha256 and row count of every frozen file. A label corrected under the rule in
# evals/README.md changes these and the README's table in the same commit.
PINNED = {
    "batch1/gold.jsonl": (
        "b98bb221db2ba95cbeb5077cb7df6f1cda686c150e71da0a6107a358aa104ce4", 125,
    ),
    "batch2/gold.jsonl": (
        "a384421dbb93d319501d165530e05cbd31a2723f85add74251e81c94358100f9", 90,
    ),
    "batch1/cross_refusals.json": (
        "bd966d310aa51ac6add495248ea8ec9d9bc91179a09da1ae1f6a88200986f4b9", 61,
    ),
    "batch2/cross_refusals.json": (
        "2b1c986ff1bb1e632e2bb925c5cce3c15d4c52c7f7d47c010bb59eb56d639240", 38,
    ),
}
BATCHES = ["batch1", "batch2"]

# What the two scripts read from a row
REQUIRED = {"id", "kind", "question"}
REQUIRED_REFUSAL = REQUIRED | {"group", "absent"}
REQUIRED_ANSWERABLE = REQUIRED | {"tier", "evidence", "source"}


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(batch):
    text = (GOLD_DIR / batch / "gold.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _snapshot():
    """Every file under evals/gold/ with its hash and modification time."""
    return {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in GOLD_DIR.rglob("*")
        if path.is_file()
    }


def _stub_corpus(root, batches=BATCHES):
    """Empty PEP files, one per source the batches name, in root/full and root/small."""
    full, small = root / "full", root / "small"
    full.mkdir()
    small.mkdir()
    for batch in batches:
        for row in _rows(batch):
            if row["kind"] != "refusal":
                (full / row["source"].split(":")[0]).touch()
    (small / "pep-0008.rst").touch()
    return full, small


@pytest.mark.parametrize("name", PINNED)
def test_frozen_file_matches_its_pin(name):
    sha256, count = PINNED[name]
    data = (GOLD_DIR / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256, (
        f"{name} changed: the gold set is frozen (see evals/README.md)"
    )
    if name.endswith(".jsonl"):
        assert len(data.decode("utf-8").splitlines()) == count
    else:
        assert len(json.loads(data)) == count


def test_nothing_else_lies_in_the_gold_folder():
    found = {path.relative_to(GOLD_DIR).as_posix() for path in _snapshot()}
    assert found == set(PINNED)


@pytest.mark.parametrize("name", [n for n in PINNED if n.endswith(".jsonl")])
def test_readme_table_repeats_the_pins(name):
    sha256, count = PINNED[name]
    readme = (EVALS / "README.md").read_text(encoding="utf-8")
    assert f"| `gold/{name}` | {count} | `{sha256}` |" in readme


@pytest.mark.parametrize("batch", BATCHES)
def test_every_row_has_the_fields_the_scripts_read(batch):
    rows = _rows(batch)
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        required = REQUIRED_REFUSAL if row["kind"] == "refusal" else REQUIRED_ANSWERABLE
        assert required <= row.keys(), row["id"]
        patterns = row["absent"] if row["kind"] == "refusal" else row["evidence"]
        assert patterns, row["id"]
        for pattern in patterns:
            re.compile(pattern)


@pytest.mark.parametrize("batch", BATCHES)
def test_cross_refusals_are_full_tier_rows(batch):
    full = {row["id"] for row in _rows(batch) if row.get("tier") == "full"}
    cross = json.loads((GOLD_DIR / batch / "cross_refusals.json").read_text(encoding="utf-8"))
    assert set(cross) <= full


@pytest.mark.parametrize("batch", BATCHES)
def test_checker_leaves_the_gold_folder_alone(batch, tmp_path, capsys):
    full, small = _stub_corpus(tmp_path)
    out = tmp_path / "cross.json"
    before = _snapshot()

    _load(CHECKER).main([
        "--gold", str(GOLD_DIR / batch / "gold.jsonl"),
        "--corpus", str(full),
        "--small", str(small),
        "--out", str(out),
    ])

    assert _snapshot() == before
    # The stub PEPs are empty, so no evidence is found in the small layer
    full_tier = [row["id"] for row in _rows(batch) if row.get("tier") == "full"]
    assert json.loads(out.read_text(encoding="utf-8")) == full_tier
    assert "problems" in capsys.readouterr().out


def test_checker_reports_cross_layer_refusals(tmp_path):
    full, small = tmp_path / "full", tmp_path / "small"
    full.mkdir()
    small.mkdir()
    (full / "pep-0001.rst").write_text("the spam rule", encoding="utf-8")
    (full / "pep-0002.rst").write_text("the eggs rule", encoding="utf-8")
    (small / "pep-0001.rst").write_text("the spam rule", encoding="utf-8")
    rows = [
        {"id": "a", "kind": "concept", "tier": "full", "question": "?",
         "evidence": ["spam"], "source": "pep-0001.rst:1"},
        {"id": "b", "kind": "concept", "tier": "full", "question": "?",
         "evidence": ["eggs"], "source": "pep-0002.rst:1"},
    ]
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "cross.json"

    _load(CHECKER).main([
        "--gold", str(gold), "--corpus", str(full), "--small", str(small), "--out", str(out),
    ])

    assert json.loads(out.read_text(encoding="utf-8")) == ["b"]


CHECKER_ARGS = {"--gold": "g", "--corpus": "c", "--small": "s", "--out": "o"}
SCORER_ARGS = {"--gold": "g", "--layer": "peps_small", "--out": "o"}
MISSING = [(CHECKER, CHECKER_ARGS, flag) for flag in CHECKER_ARGS] + [
    (SCORER, SCORER_ARGS, flag) for flag in SCORER_ARGS
]


@pytest.mark.parametrize("script", [CHECKER, SCORER], ids=lambda p: p.stem)
def test_no_arguments_prints_usage(script):
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert done.returncode != 0
    assert "usage:" in done.stderr
    assert done.stdout == ""


@pytest.mark.parametrize(
    ("script", "arguments", "flag"), MISSING, ids=lambda v: getattr(v, "stem", None)
)
def test_a_missing_argument_prints_usage(script, arguments, flag, capsys):
    argv = [item for key, value in arguments.items() if key != flag for item in (key, value)]
    with patch("stratakb.mcp_server.build_server") as build_server, pytest.raises(SystemExit) as e:
        _load(script).main(argv)
    assert e.value.code != 0
    err = capsys.readouterr().err
    assert "usage:" in err
    assert flag in err
    build_server.assert_not_called()


@pytest.mark.parametrize("script", [CHECKER, SCORER], ids=lambda p: p.stem)
def test_output_under_the_gold_folder_is_refused(script, tmp_path, capsys):
    target = GOLD_DIR / "batch1" / "cross_refusals.json"
    if script == CHECKER:
        full, small = _stub_corpus(tmp_path)
        argv = ["--corpus", str(full), "--small", str(small)]
    else:
        argv = ["--layer", "peps_small"]
    before = _snapshot()
    with pytest.raises(SystemExit) as e:
        _load(script).main(
            ["--gold", str(GOLD_DIR / "batch1" / "gold.jsonl"), "--out", str(target), *argv]
        )
    assert e.value.code != 0
    assert "frozen" in capsys.readouterr().err
    assert _snapshot() == before


def test_checker_refuses_a_folder_without_peps(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    full, _small = _stub_corpus(tmp_path)
    with pytest.raises(SystemExit) as e:
        _load(CHECKER).main([
            "--gold", str(GOLD_DIR / "batch1" / "gold.jsonl"),
            "--corpus", str(full),
            "--small", str(empty),
            "--out", str(tmp_path / "cross.json"),
        ])
    assert e.value.code != 0
    assert "--small" in capsys.readouterr().err


@pytest.fixture
def scorer_input(tmp_path):
    """A small gold file and its cross-refusals in one folder; yields that folder."""
    rows = [
        {"id": "s1", "kind": "concept", "tier": "small", "question": "small?",
         "evidence": ["needle"], "sampled": True},
        {"id": "f1", "kind": "concept", "tier": "full", "question": "other layer?",
         "evidence": ["needle"]},
        {"id": "f2", "kind": "identifier", "tier": "full", "question": "in both layers?",
         "evidence": ["needle"]},
        {"id": "r1", "kind": "refusal", "group": "off_domain", "question": "refuse?",
         "absent": ["nothing"]},
    ]
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "gold.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (folder / "cross_refusals.json").write_text('["f1"]', encoding="utf-8")
    return folder


def _score(folder, layer, out, *extra):
    """Run the scorer on a stub server; returns the questions it asked and the layers."""

    async def call_tool(name, arguments):
        assert name == "strata_ask"
        calls.append((arguments["question"], arguments["layers"]))
        found = arguments["question"] != "refuse?"
        return {
            "answer_status": "found" if found else "not_found",
            "passages": [{"text": "a  needle", "citation": {"file": "a.rst"}}] if found else [],
        }

    calls = []
    server = MagicMock(call_tool=AsyncMock(side_effect=call_tool))
    with (
        patch("stratakb.mcp_server.build_server", return_value=server),
        patch("stratakb.core.logging.configure_logging"),
    ):
        _load(SCORER).main(
            ["--gold", str(folder / "gold.jsonl"), "--layer", layer, "--out", str(out), *extra]
        )
    return calls


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        ("peps_small", {"s1": "answer", "f1": "refuse", "r1": "refuse"}),
        ("peps_full", {"s1": "answer", "f1": "answer", "f2": "answer", "r1": "refuse"}),
    ],
)
def test_scorer_expects_per_layer(layer, expected, scorer_input, tmp_path, capsys):
    out = tmp_path / "out.json"
    calls = _score(scorer_input, layer, out)

    assert {layers[0] for _, layers in calls} == {layer}
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["layer"] == layer
    assert {row["id"]: row["want"] for row in result["rows"]} == expected
    answered = next(row for row in result["rows"] if row["id"] == "s1")
    assert (answered["rank"], answered["status"]) == (1, "found")
    assert "answer:all" in capsys.readouterr().out


def test_scorer_writes_only_its_output_file(scorer_input, tmp_path):
    out = tmp_path / "scores" / "out.json"
    out.parent.mkdir()
    before = {path: path.read_bytes() for path in scorer_input.iterdir()}

    _score(scorer_input, "peps_small", out)

    assert {path: path.read_bytes() for path in scorer_input.iterdir()} == before
    assert list(out.parent.iterdir()) == [out]


def test_scorer_reads_a_cross_refusals_file_it_is_given(scorer_input, tmp_path):
    other = tmp_path / "other.json"
    other.write_text('["f2"]', encoding="utf-8")
    out = tmp_path / "out.json"

    _score(scorer_input, "peps_small", out, "--cross-refusals", str(other))

    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    assert {row["id"] for row in rows} == {"s1", "f2", "r1"}


def test_scorer_refuses_a_missing_output_folder_before_any_question(scorer_input, tmp_path, capsys):
    with (
        patch("stratakb.mcp_server.build_server") as build_server,
        pytest.raises(SystemExit) as e,
    ):
        _load(SCORER).main([
            "--gold", str(scorer_input / "gold.jsonl"),
            "--layer", "peps_small",
            "--out", str(tmp_path / "nowhere" / "out.json"),
        ])
    assert e.value.code != 0
    assert "--out" in capsys.readouterr().err
    build_server.assert_not_called()
