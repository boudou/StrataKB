"""XML/XSD parser for message schemas (e.g. ISO 20022) and generic XML documents.

Extracts element definitions, complex types, constraints, and descriptions
from XSD schema files. Each complex type or element group becomes a page.
Generic XML files are parsed into a structured text representation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import structlog

from stratakb.knowledge.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = structlog.get_logger()

# Common XSD namespace
XSD_NS = "http://www.w3.org/2001/XMLSchema"
_NS = {"xs": XSD_NS}


def parse_xml(file_path: str | Path) -> ParsedDocument:
    """Parse an XML or XSD file into searchable text chunks.

    For XSD files: extracts complex types, elements, and groups as pages.
    For generic XML: produces a structured text dump.

    Args:
        file_path: Path to the XML/XSD file.

    Returns:
        ParsedDocument with one page per type/element group.
    """
    file_path = Path(file_path)
    source = file_path.name

    if not file_path.exists():
        return ParsedDocument(source_file=source, error=f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in (".xml", ".xsd", ".xmi", ".xmo"):
        return ParsedDocument(source_file=source, error=f"Not an XML/XSD file: {file_path}")

    try:
        tree = ET.parse(str(file_path))
        root = tree.getroot()
    except ET.ParseError as e:
        return ParsedDocument(source_file=source, error=f"XML parse error: {e}")

    # Detect XSD schema
    root_tag = _strip_ns(root.tag)
    is_xsd = root_tag == "schema" or root.tag == f"{{{XSD_NS}}}schema"

    if is_xsd:
        return _parse_xsd(root, source)
    else:
        return _parse_generic_xml(root, source)


def _parse_xsd(root: ET.Element, source: str) -> ParsedDocument:
    """Parse an XSD schema, extracting complex types, elements, and groups."""
    pages: list[ParsedPage] = []
    page_num = 0

    # Extract schema-level info for summary
    target_ns = root.get("targetNamespace", "")
    schema_info_parts = [f"# XSD Schema: {source}"]
    if target_ns:
        schema_info_parts.append(f"**Target Namespace:** {target_ns}")

    # Count items for summary
    complex_types = root.findall("xs:complexType[@name]", _NS)
    simple_types = root.findall("xs:simpleType[@name]", _NS)
    top_elements = root.findall("xs:element[@name]", _NS)
    groups = root.findall("xs:group[@name]", _NS)

    # Also search without namespace prefix for non-prefixed XSD
    if not complex_types:
        complex_types = [el for el in root if _strip_ns(el.tag) == "complexType" and el.get("name")]
    if not simple_types:
        simple_types = [el for el in root if _strip_ns(el.tag) == "simpleType" and el.get("name")]
    if not top_elements:
        top_elements = [el for el in root if _strip_ns(el.tag) == "element" and el.get("name")]
    if not groups:
        groups = [el for el in root if _strip_ns(el.tag) == "group" and el.get("name")]

    schema_info_parts.append(f"**Complex Types ({len(complex_types)})** | "
                             f"**Simple Types ({len(simple_types)})** | "
                             f"**Elements ({len(top_elements)})** | "
                             f"**Groups ({len(groups)})**")

    # Summary page
    pages.append(ParsedPage(
        text="\n".join(schema_info_parts),
        page_number=page_num,
        source_file=source,
    ))
    page_num += 1

    # Complex types
    for ct in complex_types:
        text = _complex_type_to_text(ct, source)
        if text.strip():
            pages.append(ParsedPage(text=text, page_number=page_num, source_file=source))
            page_num += 1

    # Simple types (restrictions/enumerations)
    for st in simple_types:
        text = _simple_type_to_text(st, source)
        if text.strip():
            pages.append(ParsedPage(text=text, page_number=page_num, source_file=source))
            page_num += 1

    # Top-level elements with inline complex types
    for el in top_elements:
        text = _element_to_text(el, source)
        if text.strip():
            pages.append(ParsedPage(text=text, page_number=page_num, source_file=source))
            page_num += 1

    # Groups
    for grp in groups:
        text = _group_to_text(grp, source)
        if text.strip():
            pages.append(ParsedPage(text=text, page_number=page_num, source_file=source))
            page_num += 1

    logger.info(
        "xsd_parsed",
        file=source,
        complex_types=len(complex_types),
        simple_types=len(simple_types),
        elements=len(top_elements),
        groups=len(groups),
        pages=len(pages),
    )

    return ParsedDocument(source_file=source, pages=pages, total_pages=len(pages))


def _complex_type_to_text(ct: ET.Element, source: str) -> str:
    """Convert a complexType element to structured text."""
    name = ct.get("name", "anonymous")
    lines = [f"## ComplexType: {name}", f"**Source:** {source}"]

    # Documentation/annotation
    doc = _get_annotation(ct)
    if doc:
        lines.append(f"**Description:** {doc}")

    # Extract child elements from sequence/all/choice
    children = _get_child_elements(ct)
    if children:
        lines.append("\n**Elements:**")
        for child in children:
            lines.append(child)

    # Attributes
    attrs = _get_attributes(ct)
    if attrs:
        lines.append("\n**Attributes:**")
        for attr in attrs:
            lines.append(attr)

    return "\n".join(lines)


def _simple_type_to_text(st: ET.Element, source: str) -> str:
    """Convert a simpleType element to structured text."""
    name = st.get("name", "anonymous")
    lines = [f"## SimpleType: {name}", f"**Source:** {source}"]

    doc = _get_annotation(st)
    if doc:
        lines.append(f"**Description:** {doc}")

    # Restriction base and facets
    restriction = _find_child(st, "restriction")
    if restriction is not None:
        base = restriction.get("base", "")
        if base:
            lines.append(f"**Base Type:** {_strip_prefix(base)}")

        # Enumerations
        enums = [e.get("value", "") for e in _find_children(restriction, "enumeration")]
        if enums:
            lines.append(f"**Allowed Values:** {', '.join(enums)}")

        # Facets
        for facet_name in ("minLength", "maxLength", "minInclusive", "maxInclusive",
                           "pattern", "totalDigits", "fractionDigits"):
            facet = _find_child(restriction, facet_name)
            if facet is not None:
                lines.append(f"**{facet_name}:** {facet.get('value', '')}")

    return "\n".join(lines)


def _element_to_text(el: ET.Element, source: str) -> str:
    """Convert a top-level element definition to structured text."""
    name = el.get("name", "anonymous")
    el_type = el.get("type", "")
    lines = [f"## Element: {name}", f"**Source:** {source}"]

    doc = _get_annotation(el)
    if doc:
        lines.append(f"**Description:** {doc}")

    if el_type:
        lines.append(f"**Type:** {_strip_prefix(el_type)}")

    # Inline complex type
    inline_ct = _find_child(el, "complexType")
    if inline_ct is not None:
        children = _get_child_elements(inline_ct)
        if children:
            lines.append("\n**Elements:**")
            for child in children:
                lines.append(child)

    return "\n".join(lines)


def _group_to_text(grp: ET.Element, source: str) -> str:
    """Convert a group definition to structured text."""
    name = grp.get("name", "anonymous")
    lines = [f"## Group: {name}", f"**Source:** {source}"]

    doc = _get_annotation(grp)
    if doc:
        lines.append(f"**Description:** {doc}")

    children = _get_child_elements(grp)
    if children:
        lines.append("\n**Elements:**")
        for child in children:
            lines.append(child)

    return "\n".join(lines)


def _get_child_elements(parent: ET.Element) -> list[str]:
    """Extract child element declarations from sequence/all/choice."""
    results = []
    for container_tag in ("sequence", "all", "choice"):
        container = _find_child(parent, container_tag)
        if container is not None:
            for elem in _find_children(container, "element"):
                name = elem.get("name", elem.get("ref", "?"))
                name = _strip_prefix(name)
                el_type = elem.get("type", "")
                el_type = _strip_prefix(el_type) if el_type else ""
                min_occ = elem.get("minOccurs", "")
                max_occ = elem.get("maxOccurs", "")

                parts = [f"- `{name}`"]
                if el_type:
                    parts.append(f"({el_type})")
                constraints = []
                if min_occ:
                    constraints.append(f"min={min_occ}")
                if max_occ:
                    constraints.append(f"max={max_occ}")
                if constraints:
                    parts.append(f"[{', '.join(constraints)}]")

                doc = _get_annotation(elem)
                if doc:
                    parts.append(f"— {doc}")

                results.append(" ".join(parts))
    return results


def _get_attributes(parent: ET.Element) -> list[str]:
    """Extract attribute declarations."""
    results = []
    for attr in _find_children(parent, "attribute"):
        name = attr.get("name", "?")
        attr_type = _strip_prefix(attr.get("type", ""))
        use = attr.get("use", "optional")
        parts = [f"- `@{name}`"]
        if attr_type:
            parts.append(f"({attr_type})")
        parts.append(f"[{use}]")
        doc = _get_annotation(attr)
        if doc:
            parts.append(f"— {doc}")
        results.append(" ".join(parts))
    return results


def _get_annotation(el: ET.Element) -> str:
    """Extract documentation text from xs:annotation/xs:documentation."""
    ann = _find_child(el, "annotation")
    if ann is not None:
        doc = _find_child(ann, "documentation")
        if doc is not None and doc.text:
            return doc.text.strip()
    return ""


def _find_child(parent: ET.Element, local_name: str) -> ET.Element | None:
    """Find first child with given local name, handling namespaces."""
    # Try with XSD namespace
    child = parent.find(f"xs:{local_name}", _NS)
    if child is not None:
        return child
    # Try without namespace
    for ch in parent:
        if _strip_ns(ch.tag) == local_name:
            return ch
    return None


def _find_children(parent: ET.Element, local_name: str) -> list[ET.Element]:
    """Find all children with given local name, handling namespaces."""
    children = parent.findall(f"xs:{local_name}", _NS)
    if children:
        return children
    return [ch for ch in parent if _strip_ns(ch.tag) == local_name]


def _strip_ns(tag: str) -> str:
    """Strip namespace URI from a tag: {uri}local -> local."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _strip_prefix(name: str) -> str:
    """Strip namespace prefix: xs:string -> string."""
    if ":" in name:
        return name.split(":", 1)[1]
    return name


def _parse_generic_xml(root: ET.Element, source: str) -> ParsedDocument:
    """Parse a generic XML file into a single text page."""
    lines = [f"# XML Document: {source}"]
    _element_tree_to_text(root, lines, depth=0)

    text = "\n".join(lines)
    page = ParsedPage(text=text, page_number=1, source_file=source)

    logger.info("xml_parsed", file=source, elements=len(lines) - 1)

    return ParsedDocument(source_file=source, pages=[page], total_pages=1)


def _element_tree_to_text(el: ET.Element, lines: list[str], depth: int) -> None:
    """Recursively convert an XML element tree to structured text."""
    indent = "  " * depth
    tag = _strip_ns(el.tag)

    attrs = " ".join(f'{k}="{v}"' for k, v in el.attrib.items())
    header = f"{indent}**{tag}**" + (f" ({attrs})" if attrs else "")
    lines.append(header)

    if el.text and el.text.strip():
        lines.append(f"{indent}  {el.text.strip()}")

    for child in el:
        _element_tree_to_text(child, lines, depth + 1)
