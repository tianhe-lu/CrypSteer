"""Java AST adapter used by constant-sensitive detectors.

The constant-sensitive detector family requires ``javalang`` for Java parsing.
Offset-preserving source-span tables are built only after the javalang parse
succeeds, because detector findings still need exact Java substrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crypto_steering.validation.java.lightweight_ast import JavaMethodNode, parse_method_nodes
from crypto_steering.validation.java.source import find_matching_delimiter, strip_java_comments
from crypto_steering.validation.java.static_analysis import Assignment, Parameter, extract_assignments


@dataclass(frozen=True)
class JavaAstContext:
    source: str
    parser_name: str
    methods: tuple[JavaMethodNode, ...]
    assignments: tuple[Assignment, ...]
    external_tree: Any | None = None
    parse_error: str = ""

    @property
    def parse_success(self) -> bool:
        return self.external_tree is not None and not self.parse_error


def build_java_ast_context(source: str) -> JavaAstContext:
    """Build the shared Java analysis context from a required javalang parse."""

    stripped = strip_java_comments(source)
    tree, parse_error = _parse_with_javalang(stripped)
    if tree is None:
        return JavaAstContext(
            source=stripped,
            parser_name="javalang+lightweight-fallback",
            methods=parse_method_nodes(stripped),
            assignments=extract_assignments(stripped),
            external_tree=None,
            parse_error=parse_error,
        )
    return JavaAstContext(
        source=stripped,
        parser_name="javalang",
        methods=_method_nodes_from_javalang(stripped, tree),
        assignments=extract_assignments(stripped),
        external_tree=tree,
    )


def _parse_with_javalang(source: str) -> tuple[Any | None, str]:
    try:
        import javalang  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("javalang is required for constant-sensitive detectors") from exc
    try:
        return javalang.parse.parse(source), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _method_nodes_from_javalang(source: str, tree: Any) -> tuple[JavaMethodNode, ...]:
    try:
        import javalang  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import checked earlier
        raise RuntimeError("javalang is required for constant-sensitive detectors") from exc

    line_offsets = _line_offsets(source)
    methods: list[JavaMethodNode] = []
    declarations = list(tree.filter(javalang.tree.MethodDeclaration))
    declarations.extend(tree.filter(javalang.tree.ConstructorDeclaration))
    for _path, node in declarations:
        if node.position is None:
            continue
        if getattr(node, "body", None) is None:
            continue
        node_start = _offset_from_position(line_offsets, node.position)
        body_start = source.find("{", node_start)
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_end == -1:
            continue
        methods.append(
            JavaMethodNode(
                name=node.name,
                params=tuple(_parameter_from_javalang(param) for param in getattr(node, "parameters", ())),
                start=_member_header_start(source, body_start),
                body_start=body_start,
                body_end=body_end,
            )
        )
    return tuple(sorted(methods, key=lambda method: method.body_start))


def _parameter_from_javalang(parameter: Any) -> Parameter:
    type_name = _type_name_from_javalang(parameter.type)
    if getattr(parameter, "varargs", False):
        type_name = f"{type_name}[]"
    return Parameter(name=parameter.name, type_name=type_name)


def _type_name_from_javalang(type_node: Any) -> str:
    parts: list[str] = []
    current = type_node
    while current is not None:
        name = getattr(current, "name", "")
        if name:
            parts.append(str(name))
        current = getattr(current, "sub_type", None)
    dimensions = getattr(type_node, "dimensions", ()) or ()
    return ".".join(parts) + "[]" * len(dimensions)


def _line_offsets(source: str) -> tuple[int, ...]:
    offsets = [0]
    for index, char in enumerate(source):
        if char == "\n":
            offsets.append(index + 1)
    return tuple(offsets)


def _offset_from_position(line_offsets: tuple[int, ...], position: Any) -> int:
    line_index = max(0, int(position.line) - 1)
    if line_index >= len(line_offsets):
        return line_offsets[-1]
    return line_offsets[line_index] + max(0, int(position.column) - 1)


def _member_header_start(source: str, body_start: int) -> int:
    return max(
        source.rfind(";", 0, body_start),
        source.rfind("{", 0, body_start),
        source.rfind("}", 0, body_start),
    ) + 1
