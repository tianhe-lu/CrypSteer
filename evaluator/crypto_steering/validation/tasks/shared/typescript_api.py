"""Resolve TypeScript calls bound to a Node module namespace."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.source.index import CallExpression, SourceAnalyzer


_IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"


@dataclass(frozen=True)
class TypeScriptCryptoCall:
    """A source call paired with its canonical Node crypto operation."""

    operation: str
    call: CallExpression


def node_crypto_calls(
    analyzer: SourceAnalyzer,
    *operations: str,
) -> tuple[TypeScriptCryptoCall, ...]:
    """Return calls proven to refer to the requested Node crypto exports.

    Generated programs often bind a function from a crypto namespace through
    assignment or destructuring. Bare calls count only when such a binding
    exists, so an unrelated local function with the same name is not mistaken
    for a Node API call.
    """

    if analyzer.language != "typescript" or not operations:
        return ()
    wanted = set(operations)
    aliases = node_crypto_function_aliases(analyzer, *operations)
    namespaces = _node_crypto_namespaces(analyzer)
    query_names = tuple(sorted(wanted | set(aliases)))
    matches: list[TypeScriptCryptoCall] = []
    for call in analyzer.calls(*query_names):
        compact = re.sub(r"\s+", "", call.callee)
        if "." in compact:
            receiver, operation = compact.rsplit(".", 1)
            if receiver in namespaces and operation in wanted:
                matches.append(TypeScriptCryptoCall(operation, call))
            continue
        operation = aliases.get(call.simple_name)
        if operation in wanted:
            matches.append(TypeScriptCryptoCall(operation, call))
    return tuple(matches)


def node_crypto_function_aliases(
    analyzer: SourceAnalyzer,
    *operations: str,
) -> dict[str, str]:
    """Map local function names to canonical Node crypto export names."""

    if analyzer.language != "typescript" or not operations:
        return {}
    wanted = set(operations)
    aliases = {
        alias: imported
        for imported, alias in getattr(analyzer, "import_aliases", {}).items()
        if imported in wanted
    }
    namespaces = _node_crypto_namespaces(analyzer)
    if not namespaces:
        return aliases

    masked = getattr(analyzer, "masked", analyzer.source)
    namespace_pattern = "|".join(
        re.escape(namespace)
        for namespace in sorted(namespaces, key=len, reverse=True)
    )
    operation_pattern = "|".join(
        re.escape(operation)
        for operation in sorted(wanted, key=len, reverse=True)
    )

    for match in re.finditer(
        rf"\b(?:const|let|var)\s+(?P<alias>{_IDENTIFIER})\s*=\s*"
        rf"(?P<namespace>{namespace_pattern})\s*\.\s*"
        rf"(?P<operation>{operation_pattern})\b",
        masked,
    ):
        aliases[match.group("alias")] = match.group("operation")

    for match in re.finditer(
        rf"\b(?:const|let|var)\s*\{{(?P<items>[^{{}}]*)\}}\s*=\s*"
        rf"(?P<namespace>{namespace_pattern})\b",
        masked,
    ):
        for raw_item in match.group("items").split(","):
            item = raw_item.strip()
            if not item:
                continue
            parts = re.split(r"\s*:\s*", item, maxsplit=1)
            operation = parts[0].strip()
            alias = parts[-1].strip()
            if operation in wanted and re.fullmatch(_IDENTIFIER, alias):
                aliases[alias] = operation

    changed = True
    while changed:
        changed = False
        for match in re.finditer(
            rf"\b(?:const|let|var)\s+(?P<alias>{_IDENTIFIER})\s*=\s*"
            rf"(?P<source>{_IDENTIFIER})\s*;",
            masked,
        ):
            source_name = match.group("source")
            alias = match.group("alias")
            if source_name in aliases and alias not in aliases:
                aliases[alias] = aliases[source_name]
                changed = True
    return aliases


def _node_crypto_namespaces(analyzer: SourceAnalyzer) -> frozenset[str]:
    return frozenset(
        set(getattr(analyzer, "crypto_namespace_aliases", ())) | {"crypto"}
    )
