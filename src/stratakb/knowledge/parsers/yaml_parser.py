"""YAML parser for OpenAPI 3.x and AsyncAPI 3.x specifications.

Extracts API operations from OpenAPI and AsyncAPI specs, creating one chunk
per operation with structured metadata for semantic search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from stratakb.knowledge.parsers.document import ParsedDocument, ParsedPage, ParserFlag

logger = structlog.get_logger()

# Properties listed per schema; the rest are counted and the file is flagged
MAX_SCHEMA_PROPERTIES = 15


def parse_yaml(file_path: str | Path) -> ParsedDocument:
    """Parse an OpenAPI or AsyncAPI YAML file into searchable chunks.

    Each API operation becomes a separate page/chunk with metadata.
    A summary page is prepended for the overall spec.

    Args:
        file_path: Path to the YAML file.

    Returns:
        ParsedDocument with one page per operation + summary page.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    if file_path.suffix.lower() not in (".yaml", ".yml"):
        return ParsedDocument(source_file=source, error=f"Not a YAML file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return ParsedDocument(source_file=source, error=f"YAML parse error: {e}")

    if not isinstance(spec, dict):
        return ParsedDocument(source_file=source, error="YAML file does not contain a mapping")

    # Detect spec type
    if "openapi" in spec:
        return _parse_openapi(spec, source)
    elif "asyncapi" in spec:
        return _parse_asyncapi(spec, source)
    else:
        # Plain YAML — treat as a single document
        return _parse_plain_yaml(spec, source)


def _extract_info(spec: dict) -> tuple[str, str, str]:
    """Extract title, description, version from info block."""
    info = spec.get("info", {})
    title = info.get("title", "Unknown")
    description = info.get("description", "")
    version = info.get("version", "")
    return title, description, version


def _schema_summary(schema: dict | None, depth: int = 0) -> tuple[str, int]:
    """Generate a human-readable summary of a JSON schema.

    Returns the summary text and the number of properties left out of it
    (those past MAX_SCHEMA_PROPERTIES, nested summaries included).
    """
    if not schema or depth > 2:
        return "", 0

    if "$ref" in schema:
        ref = schema["$ref"]
        name = ref.split("/")[-1]
        return name, 0

    schema_type = schema.get("type", "object")
    props = schema.get("properties", {})
    if not props:
        return schema_type, 0

    lines = []
    left_out = 0
    for prop_name, prop_schema in list(props.items())[:MAX_SCHEMA_PROPERTIES]:
        # Summarize the nested schema only when its text is used, so its cut
        # properties are counted only when they would have been listed
        prop_type = prop_schema.get("type")
        if prop_type is None:
            prop_type, nested_cut = _schema_summary(prop_schema, depth + 1)
            left_out += nested_cut
        desc = prop_schema.get("description", "")
        if desc:
            lines.append(f"  - {prop_name} ({prop_type}): {desc}")
        else:
            lines.append(f"  - {prop_name} ({prop_type})")

    if len(props) > MAX_SCHEMA_PROPERTIES:
        cut = len(props) - MAX_SCHEMA_PROPERTIES
        lines.append(f"  ... and {cut} more properties")
        left_out += cut

    return "\n".join(lines), left_out


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve a $ref pointer within the spec. Returns empty dict on failure."""
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    obj = spec
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part, {})
        else:
            return {}
    return obj if isinstance(obj, dict) else {}


def _parse_openapi(spec: dict, source: str) -> ParsedDocument:
    """Parse OpenAPI 3.x spec into operation-level chunks."""
    title, description, version = _extract_info(spec)
    pages: list[ParsedPage] = []
    page_num = 1

    # Collect operation names for summary
    operation_names: list[str] = []
    # Schema properties left out of the summaries, over the whole file
    properties_left_out = 0

    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method in ("get", "post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if not operation or not isinstance(operation, dict):
                continue

            op_id = operation.get("operationId", f"{method.upper()} {path}")
            summary = operation.get("summary", "")
            op_desc = operation.get("description", "")
            tags = operation.get("tags", [])

            operation_names.append(op_id)

            # Build operation text
            lines = [
                f"# {title} — {op_id}",
                f"**Method:** {method.upper()} {path}",
            ]

            if tags:
                lines.append(f"**Tags:** {', '.join(tags)}")

            if summary:
                lines.append(f"**Summary:** {summary}")

            if op_desc and op_desc != summary:
                lines.append(f"**Description:** {op_desc}")

            # Parameters
            params = operation.get("parameters", [])
            if params:
                lines.append("\n**Parameters:**")
                for param in params:
                    if "$ref" in param:
                        param = _resolve_ref(spec, param["$ref"])
                    p_name = param.get("name", "?")
                    p_in = param.get("in", "?")
                    p_desc = param.get("description", "")
                    p_required = param.get("required", False)
                    req_str = " (required)" if p_required else ""
                    lines.append(f"  - {p_name} (in: {p_in}){req_str}: {p_desc}")

            # Request body
            req_body = operation.get("requestBody", {})
            if "$ref" in req_body:
                req_body = _resolve_ref(spec, req_body["$ref"])
            if req_body:
                content = req_body.get("content", {})
                for media_type, media_schema in content.items():
                    schema = media_schema.get("schema", {})
                    schema_text, cut = _schema_summary(schema)
                    properties_left_out += cut
                    if schema_text:
                        lines.append(f"\n**Request Body** ({media_type}):\n{schema_text}")

            # Response (just 200)
            responses = operation.get("responses", {})
            success_resp = responses.get("200", responses.get("201", {}))
            if "$ref" in success_resp:
                success_resp = _resolve_ref(spec, success_resp["$ref"])
            if success_resp:
                resp_desc = success_resp.get("description", "")
                if resp_desc:
                    lines.append(f"\n**Success Response:** {resp_desc}")
                content = success_resp.get("content", {})
                for media_type, media_schema in content.items():
                    schema = media_schema.get("schema", {})
                    schema_text, cut = _schema_summary(schema)
                    properties_left_out += cut
                    if schema_text:
                        lines.append(f"**Response Schema** ({media_type}):\n{schema_text}")

            page_text = "\n".join(lines)

            pages.append(ParsedPage(
                text=page_text,
                page_number=page_num,
                source_file=source,
            ))
            page_num += 1

    # Summary page (prepend)
    op_list = ", ".join(operation_names[:20])
    if len(operation_names) > 20:
        op_list += f", ... ({len(operation_names)} total)"

    summary_text = (
        f"# {title} — API Overview\n"
        f"**API Version:** {version}\n"
        f"**Description:** {description}\n"
        f"**Operations ({len(operation_names)}):** {op_list}\n"
        f"**Paths:** {len(paths)}"
    )

    summary_page = ParsedPage(
        text=summary_text,
        page_number=0,
        source_file=source,
    )

    all_pages = [summary_page] + pages

    logger.info(
        "openapi_parsed",
        file=source,
        service_domain=title,
        version=version,
        operations=len(operation_names),
        properties_left_out=properties_left_out,
    )

    return ParsedDocument(
        source_file=source,
        pages=all_pages,
        total_pages=len(all_pages),
        flag=ParserFlag("truncated", properties_left_out) if properties_left_out else None,
    )


def _parse_asyncapi(spec: dict, source: str) -> ParsedDocument:
    """Parse AsyncAPI 3.x spec into channel-level chunks."""
    title, description, version = _extract_info(spec)
    pages: list[ParsedPage] = []
    page_num = 1

    channel_names: list[str] = []

    channels = spec.get("channels", {})
    for channel_name, channel_def in channels.items():
        if not isinstance(channel_def, dict):
            continue

        channel_names.append(channel_name)
        address = channel_def.get("address", channel_name)
        ch_desc = channel_def.get("description", "")

        lines = [
            f"# {title} — Channel: {channel_name}",
            f"**Address:** {address}",
        ]

        if ch_desc:
            lines.append(f"**Description:** {ch_desc}")

        # Messages
        messages = channel_def.get("messages", {})
        if messages:
            lines.append("\n**Messages:**")
            for msg_name, msg_def in messages.items():
                if "$ref" in msg_def:
                    ref_name = msg_def["$ref"].split("/")[-1]
                    lines.append(f"  - {msg_name}: {ref_name}")
                else:
                    lines.append(f"  - {msg_name}")

        page_text = "\n".join(lines)
        pages.append(ParsedPage(
            text=page_text,
            page_number=page_num,
            source_file=source,
        ))
        page_num += 1

    # Operations (AsyncAPI 3.x)
    operations = spec.get("operations", {})
    for op_name, op_def in operations.items():
        if not isinstance(op_def, dict):
            continue

        action = op_def.get("action", "")
        channel_ref = op_def.get("channel", {})
        op_desc = op_def.get("description", op_def.get("summary", ""))

        lines = [
            f"# {title} — Operation: {op_name}",
            f"**Action:** {action}",
        ]

        if isinstance(channel_ref, dict) and "$ref" in channel_ref:
            lines.append(f"**Channel:** {channel_ref['$ref'].split('/')[-1]}")
        elif isinstance(channel_ref, str):
            lines.append(f"**Channel:** {channel_ref}")

        if op_desc:
            lines.append(f"**Description:** {op_desc}")

        page_text = "\n".join(lines)
        pages.append(ParsedPage(
            text=page_text,
            page_number=page_num,
            source_file=source,
        ))
        page_num += 1

    # Summary page
    ch_list = ", ".join(channel_names[:20])
    if len(channel_names) > 20:
        ch_list += f", ... ({len(channel_names)} total)"

    summary_text = (
        f"# {title} — AsyncAPI Overview\n"
        f"**API Version:** {version}\n"
        f"**Description:** {description}\n"
        f"**Channels ({len(channel_names)}):** {ch_list}"
    )

    summary_page = ParsedPage(
        text=summary_text,
        page_number=0,
        source_file=source,
    )

    all_pages = [summary_page] + pages

    logger.info(
        "asyncapi_parsed",
        file=source,
        service_domain=title,
        version=version,
        channels=len(channel_names),
    )

    return ParsedDocument(
        source_file=source,
        pages=all_pages,
        total_pages=len(all_pages),
    )


def _parse_plain_yaml(spec: dict, source: str) -> ParsedDocument:
    """Parse a generic YAML file as a single document."""
    text = yaml.dump(spec, default_flow_style=False, allow_unicode=True)

    pages = [ParsedPage(
        text=text,
        page_number=1,
        source_file=source,
    )]

    logger.info("yaml_parsed", file=source, type="plain")

    return ParsedDocument(
        source_file=source,
        pages=pages,
        total_pages=1,
    )
