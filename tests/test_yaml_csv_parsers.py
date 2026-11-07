"""Tests for YAML/OpenAPI and CSV parsers.

Tests cover:
- OpenAPI 3.x detection and operation extraction
- AsyncAPI 3.x detection and channel extraction
- Plain YAML fallback
- CSV parsing of a reason code list
- Error handling for bad files
- Metadata fields (service_domain, operation_type, api_version)
- Integration with ingestion pipeline (FORMATS)
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_OPENAPI = FIXTURES_DIR / "test_openapi.yaml"
TEST_ASYNCAPI = FIXTURES_DIR / "test_asyncapi.yaml"
TEST_CSV = FIXTURES_DIR / "reason_codes.csv"


# ==========================================
# OpenAPI parser tests
# ==========================================


class TestOpenAPIParser:
    def test_detects_openapi_format(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        assert result.error is None
        assert result.total_pages > 0

    def test_extracts_operations_as_pages(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        # 3 operations + 1 summary = 4 pages
        assert result.total_pages == 4

    def test_summary_page_present(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        summary = result.pages[0]
        assert summary.page_number == 0
        assert "Customer Order" in summary.text
        assert "API Overview" in summary.text
        assert "14.0.0" in summary.text

    def test_operation_page_has_method_and_path(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        # First operation page (after summary)
        initiate_page = result.pages[1]
        assert "POST" in initiate_page.text
        assert "/CustomerOrder/Initiate" in initiate_page.text
        assert "Initiate" in initiate_page.text

    def test_operation_has_summary_and_description(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        initiate_page = result.pages[1]
        assert "InCR Initiate A new Customer Order" in initiate_page.text

    def test_operation_has_parameters(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        # Retrieve operation has a path parameter
        retrieve_page = result.pages[2]
        assert "customerorderid" in retrieve_page.text
        assert "path" in retrieve_page.text

    def test_operation_has_request_body(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        initiate_page = result.pages[1]
        assert "Request Body" in initiate_page.text
        assert "accountType" in initiate_page.text

    def test_operation_has_tags(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        initiate_page = result.pages[1]
        assert "CR - CustomerOrderFacility" in initiate_page.text

    def test_summary_lists_operation_count(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        summary = result.pages[0]
        assert "Operations (3)" in summary.text

    def test_service_domain_name_in_text(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_OPENAPI)
        for page in result.pages:
            assert "Customer Order" in page.text


# ==========================================
# AsyncAPI parser tests
# ==========================================


class TestAsyncAPIParser:
    def test_detects_asyncapi_format(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_ASYNCAPI)
        assert result.error is None
        assert result.total_pages > 0

    def test_extracts_channels(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_ASYNCAPI)
        # 3 channels + 1 operation + 1 summary = 5 pages
        assert result.total_pages == 5

    def test_summary_page_has_asyncapi_info(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_ASYNCAPI)
        summary = result.pages[0]
        assert "AsyncAPI Overview" in summary.text
        assert "Customer Order" in summary.text
        assert "14.0.0" in summary.text

    def test_channel_page_has_address(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_ASYNCAPI)
        # First channel page
        ch_page = result.pages[1]
        assert "Channel:" in ch_page.text
        assert "CustomerOrderFacility/Created" in ch_page.text

    def test_channel_has_messages(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_ASYNCAPI)
        ch_page = result.pages[1]
        assert "Messages:" in ch_page.text

    def test_operations_extracted(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(TEST_ASYNCAPI)
        # The operation page should be after the 3 channel pages
        op_page = result.pages[4]
        assert "Operation:" in op_page.text
        assert "receive" in op_page.text


def _write_openapi(path: Path, schema: dict) -> Path:
    """Write an OpenAPI file with one operation whose request body is schema."""
    import yaml

    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Cut API", "version": "1.0"},
        "paths": {"/items": {"post": {
            "operationId": "createItem",
            "requestBody": {"content": {"application/json": {"schema": schema}}},
            "responses": {"200": {"description": "OK"}},
        }}},
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _props(count: int, prefix: str = "p") -> dict:
    return {f"{prefix}{i}": {"type": "string"} for i in range(count)}


class TestOpenAPISchemaCut:
    def test_schema_over_limit_flagged_with_count(self, tmp_path):
        from structlog.testing import capture_logs

        from stratakb.knowledge.parsers.document import ParserFlag
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        path = _write_openapi(tmp_path / "big.yaml", {"type": "object", "properties": _props(20)})
        with capture_logs() as logs:
            result = parse_yaml(path)
        assert result.error is None
        assert result.flag == ParserFlag("truncated", 5)
        assert "... and 5 more properties" in result.pages[1].text
        parsed = [e for e in logs if e["event"] == "openapi_parsed"]
        assert parsed[0]["properties_left_out"] == 5

    def test_cuts_summed_over_file(self, tmp_path):
        from stratakb.knowledge.parsers.document import ParserFlag
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        nested = {"properties": _props(17, "n")}  # no type: its summary is listed
        props = {"child": nested, **_props(19)}
        path = _write_openapi(tmp_path / "nested.yaml", {"type": "object", "properties": props})
        result = parse_yaml(path)
        # 20 top-level properties (5 cut) plus 17 nested ones (2 cut)
        assert result.flag == ParserFlag("truncated", 7)

    def test_nested_schema_with_type_not_counted(self, tmp_path):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        # The parent property names its type, so the nested summary is never listed
        nested = {"type": "object", "properties": _props(30, "n")}
        path = _write_openapi(
            tmp_path / "typed.yaml",
            {"type": "object", "properties": {"child": nested, "name": {"type": "string"}}},
        )
        result = parse_yaml(path)
        assert result.error is None
        assert result.flag is None
        assert "child (object)" in result.pages[1].text

    def test_schemas_that_fit_not_flagged(self, tmp_path):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        path = _write_openapi(tmp_path / "fit.yaml", {"type": "object", "properties": _props(15)})
        assert parse_yaml(path).flag is None
        assert parse_yaml(TEST_OPENAPI).flag is None


# ==========================================
# YAML error handling tests
# ==========================================


class TestYAMLErrors:
    def test_nonexistent_file(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml("/nonexistent/file.yaml")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_non_yaml_file(self):
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        result = parse_yaml(__file__)  # .py file
        assert result.error is not None
        assert "Not a YAML" in result.error

    def test_plain_yaml_fallback(self):
        """A YAML file without openapi/asyncapi key should still parse."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("key: value\nlist:\n  - item1\n  - item2\n")
            f.flush()

            from stratakb.knowledge.parsers.yaml_parser import parse_yaml

            result = parse_yaml(f.name)
            assert result.error is None
            assert result.total_pages == 1
            assert "key: value" in result.pages[0].text

        Path(f.name).unlink()


# ==========================================
# CSV parser tests
# ==========================================


def _write_rows(path: Path, count: int) -> Path:
    """Write a CSV with a header and count data rows."""
    lines = ["Code,Name"] + [f"C{i},Name {i}" for i in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestCSVParser:
    def test_parse_code_list(self):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        result = parse_csv(TEST_CSV)
        assert result.error is None
        assert result.total_pages == 3  # 3 data rows

    def test_rows_have_structured_text(self):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        result = parse_csv(TEST_CSV)
        first_page = result.pages[0]
        assert "AC01" in first_page.text
        assert "IncorrectAccountNumber" in first_page.text
        assert "invalid or missing" in first_page.text

    def test_column_headers_as_labels(self):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        result = parse_csv(TEST_CSV)
        first_page = result.pages[0]
        assert "**Code:**" in first_page.text
        assert "**Description:**" in first_page.text

    def test_title_extracted(self):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        result = parse_csv(TEST_CSV)
        # Title should be from the "Name" column
        first_page = result.pages[0]
        assert first_page.text.startswith("# IncorrectAccountNumber")

    def test_small_csv_not_truncated(self):
        from structlog.testing import capture_logs

        from stratakb.knowledge.parsers.csv_parser import parse_csv

        with capture_logs() as logs:
            result = parse_csv(TEST_CSV)
        assert result.flag is None
        assert not [e for e in logs if e["event"] == "csv_truncated"]

    def test_exactly_max_rows_not_truncated(self, tmp_path):
        from structlog.testing import capture_logs

        from stratakb.knowledge.parsers.csv_parser import MAX_ROWS, parse_csv

        path = _write_rows(tmp_path / "limit.csv", MAX_ROWS)
        with capture_logs() as logs:
            result = parse_csv(path)
        assert result.error is None
        assert result.total_pages == MAX_ROWS
        assert result.flag is None
        assert not [e for e in logs if e["event"] == "csv_truncated"]

    def test_rows_past_max_counted_and_logged(self, tmp_path):
        from structlog.testing import capture_logs

        from stratakb.knowledge.parsers.csv_parser import MAX_ROWS, parse_csv
        from stratakb.knowledge.parsers.document import ParserFlag

        path = _write_rows(tmp_path / "big.csv", MAX_ROWS + 5)
        with capture_logs() as logs:
            result = parse_csv(path)
        assert result.error is None
        assert result.total_pages == MAX_ROWS
        assert result.flag == ParserFlag("truncated", 5)
        truncated = [e for e in logs if e["event"] == "csv_truncated"]
        assert len(truncated) == 1
        assert truncated[0]["log_level"] == "warning"
        assert truncated[0]["rows_read"] == MAX_ROWS
        assert truncated[0]["rows_left_out"] == 5

    def test_short_row_keeps_its_values(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "short.csv"
        path.write_text("a,b,c\n1,2\n3,4,5\n", encoding="utf-8")
        result = parse_csv(path)
        assert result.error is None
        assert result.total_pages == 2
        first = result.pages[0].text
        assert "**a:** 1" in first
        assert "**b:** 2" in first
        assert "**c:**" not in first
        assert "**c:** 5" in result.pages[1].text

    def test_long_row_keeps_extra_values(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "long.csv"
        path.write_text("a,b\n1,2,3,4\n5,6\n", encoding="utf-8")
        result = parse_csv(path)
        assert result.error is None
        assert result.total_pages == 2
        first = result.pages[0].text
        assert "**a:** 1" in first
        assert "**b:** 2" in first
        assert "**Column 3:** 3" in first
        assert "**Column 4:** 4" in first
        assert "Column" not in result.pages[1].text

    def test_empty_header_labelled_by_position(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "empty_header.csv"
        path.write_text("a,,c\n1,2,3\n", encoding="utf-8")
        result = parse_csv(path)
        assert result.error is None
        assert result.total_pages == 1
        lines = result.pages[0].text.splitlines()
        assert lines == ["**a:** 1", "**Column 2:** 2", "**c:** 3"]
        assert not [line for line in lines if line.startswith("**:**")]

    def test_empty_header_in_long_row(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "empty_header_long.csv"
        path.write_text("a,,c\n1,2,3,4\n", encoding="utf-8")
        result = parse_csv(path)
        lines = result.pages[0].text.splitlines()
        assert lines == ["**a:** 1", "**Column 2:** 2", "**c:** 3", "**Column 4:** 4"]

    def test_repeated_header_keeps_both_values(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "repeated.csv"
        path.write_text("name,value,name\nx,1,y\n", encoding="utf-8")
        result = parse_csv(path)
        assert result.error is None
        lines = result.pages[0].text.splitlines()
        assert lines.count("**name:** x") == 1
        assert lines.count("**name (2):** y") == 1
        assert lines.count("**value:** 1") == 1

    def test_repeated_title_column_uses_first(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "repeated_title.csv"
        path.write_text("Name,Name\nx,y\n", encoding="utf-8")
        result = parse_csv(path)
        text = result.pages[0].text
        assert text.startswith("# x\n")
        assert "**Name:** x" in text
        assert "**Name (2):** y" in text

    def test_title_falls_back_per_row(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "fallback.csv"
        path.write_text("Title,Name,Code\n,,C1\nT,,C2\n", encoding="utf-8")
        result = parse_csv(path)
        assert result.pages[0].text.startswith("# C1\n")
        assert result.pages[1].text.startswith("# T\n")

    def test_title_columns_once_per_header(self):
        from stratakb.knowledge.parsers.csv_parser import _title_columns

        assert _title_columns(["Code", "Name"]) == [1, 0]
        assert _title_columns(["Name", "Name (2)", "Other"]) == [0]
        assert _title_columns(["Other"]) == []

    def test_header_whitespace_stripped(self, tmp_path):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        path = tmp_path / "spaced.csv"
        path.write_text("Code, Name\nA1,Foo\n", encoding="utf-8")
        result = parse_csv(path)
        text = result.pages[0].text
        assert text.startswith("# Foo\n")
        assert "**Name:** Foo" in text
        assert "** Name:**" not in text

    def test_blank_lines_skipped(self, tmp_path):
        from unittest.mock import patch

        from stratakb.knowledge.parsers.csv_parser import parse_csv
        from stratakb.knowledge.parsers.document import ParserFlag

        path = tmp_path / "blank.csv"
        path.write_text("a,b\n1,2\n\n3,4\n", encoding="utf-8")
        result = parse_csv(path)
        assert result.error is None
        assert result.total_pages == 2
        assert [p.page_number for p in result.pages] == [1, 2]
        assert result.flag is None

        path.write_text("a,b\n1,2\n\n\n3,4\n", encoding="utf-8")
        with patch("stratakb.knowledge.parsers.csv_parser.MAX_ROWS", 1):
            result = parse_csv(path)
        assert result.total_pages == 1
        assert result.flag == ParserFlag("truncated", 1)

    def test_nonexistent_csv(self):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        result = parse_csv("/nonexistent/file.csv")
        assert result.error is not None

    def test_non_csv_file(self):
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        result = parse_csv(__file__)
        assert result.error is not None
        assert "Not a CSV" in result.error

    def test_empty_csv(self):
        import tempfile

        from stratakb.knowledge.parsers.csv_parser import parse_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("col1,col2\n")  # Header only, no data
            f.flush()

            result = parse_csv(f.name)
            assert result.error is None
            assert result.pages == []
            assert result.total_pages == 0

        Path(f.name).unlink()

    def test_csv_without_header(self):
        import tempfile

        from stratakb.knowledge.parsers.csv_parser import parse_csv

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("")  # 0 bytes: no header row at all
            f.flush()

            result = parse_csv(f.name)
            assert result.error is None
            assert result.pages == []
            assert result.total_pages == 0

        Path(f.name).unlink()


# ==========================================
# Ingestion pipeline integration
# ==========================================


class TestPipelineIntegration:
    def test_yaml_registered_in_formats(self):
        from stratakb.knowledge.parsers.registry import FORMATS

        assert ".yaml" in FORMATS
        assert ".yml" in FORMATS

    def test_csv_registered_in_formats(self):
        from stratakb.knowledge.parsers.registry import FORMATS

        assert ".csv" in FORMATS

    def test_discover_yaml_files(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(FIXTURES_DIR)
        yaml_files = [f for f in files if f.suffix in (".yaml", ".yml")]
        assert len(yaml_files) >= 2  # openapi + asyncapi

    def test_discover_csv_files(self):
        from stratakb.knowledge.ingestion import _discover_files

        files = _discover_files(FIXTURES_DIR)
        csv_files = [f for f in files if f.suffix == ".csv"]
        assert len(csv_files) >= 1

    def test_parse_file_dispatches_yaml(self):
        from stratakb.knowledge.ingestion import _parse_file

        result = _parse_file(TEST_OPENAPI)
        assert result.error is None
        assert result.total_pages > 0

    def test_chunk_document_sets_yaml_doc_type(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.yaml_parser import parse_yaml

        doc = parse_yaml(TEST_OPENAPI)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=2000, chunk_overlap=100)
        assert len(chunks) > 0
        assert chunks[0].doc_type == "yaml"

    def test_chunk_document_sets_csv_doc_type(self):
        from stratakb.knowledge.ingestion import _chunk_document
        from stratakb.knowledge.parsers.csv_parser import parse_csv

        doc = parse_csv(TEST_CSV)
        chunks = _chunk_document(doc, layer_name="test", chunk_size=2000, chunk_overlap=100)
        assert len(chunks) > 0
        assert chunks[0].doc_type == "csv"
