"""Lightweight Java source utilities used by dataset scripts.

These helpers are intentionally conservative. They are not a full Java parser,
but they handle the common shapes produced by code-generation models well
enough for extraction, static checks, and targeted M8 password rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ArgumentSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class CallSite:
    name: str
    start: int
    end: int
    open_paren: int
    close_paren: int
    arguments: tuple[ArgumentSpan, ...]


@dataclass(frozen=True)
class MethodBody:
    name: str
    params: tuple[str, ...]
    start: int
    body_start: int
    body_end: int
    text: str


def extract_java_source(raw_text: str) -> str:
    """Extract Java code from a model response and remove package declarations."""

    java_fences = re.findall(r"```java\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    generic_fences = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*(.*?)```", raw_text, flags=re.DOTALL)
    if java_fences:
        source = java_fences[-1]
    elif generic_fences:
        source = generic_fences[-1]
    else:
        source = _slice_to_java_source(raw_text)
    source = source.strip()
    source = re.sub(r"^\s*package\s+[\w.]+\s*;\s*", "", source, flags=re.MULTILINE)
    return source.strip() + "\n"


def _slice_to_java_source(raw_text: str) -> str:
    class_match = re.search(r"\b(?:public\s+)?class\s+[A-Za-z_]\w*\b", raw_text)
    if not class_match:
        return raw_text
    start = raw_text.rfind("\n", 0, class_match.start()) + 1
    return raw_text[start:]


def public_class_name(source: str) -> str | None:
    match = re.search(r"\bpublic\s+class\s+([A-Za-z_]\w*)\b", source)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+([A-Za-z_]\w*)\b", source)
    if match:
        return match.group(1)
    return None


def strip_java_comments(source: str) -> str:
    """Blank Java comments while preserving string/char literals and offsets."""

    result: list[str] = []
    i = 0
    in_string = False
    in_char = False
    escaped = False
    while i < len(source):
        char = source[i]
        next_char = source[i + 1] if i + 1 < len(source) else ""
        if in_string or in_char:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "'":
            in_char = True
            result.append(char)
            i += 1
            continue
        if char == "/" and next_char == "/":
            result.append(" ")
            result.append(" ")
            i += 2
            while i < len(source) and source[i] != "\n":
                result.append(" ")
                i += 1
            if i < len(source):
                result.append("\n")
                i += 1
            continue
        if char == "/" and next_char == "*":
            result.append(" ")
            result.append(" ")
            i += 2
            closed = False
            while i + 1 < len(source) and not (source[i] == "*" and source[i + 1] == "/"):
                result.append("\n" if source[i] == "\n" else " ")
                i += 1
            if i + 1 < len(source):
                result.append(" ")
                result.append(" ")
                i += 2
                closed = True
            if not closed:
                while i < len(source):
                    result.append("\n" if source[i] == "\n" else " ")
                    i += 1
            continue

        result.append(char)
        i += 1
    return "".join(result)


def mask_java_comments_and_literals(source: str) -> str:
    """Blank comments and string/char literals while preserving offsets."""

    source = strip_java_comments(source)
    result: list[str] = []
    i = 0
    in_string = False
    in_char = False
    escaped = False
    while i < len(source):
        char = source[i]
        if in_string or in_char:
            result.append("\n" if char == "\n" else " ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(" ")
            i += 1
            continue
        if char == "'":
            in_char = True
            result.append(" ")
            i += 1
            continue

        result.append(char)
        i += 1
    return "".join(result)


def find_matching_delimiter(source: str, open_index: int, open_char: str = "(", close_char: str = ")") -> int:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    i = open_index
    while i < len(source):
        char = source[i]
        next_char = source[i + 1] if i + 1 < len(source) else ""
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            i += 1
            continue

        if char == "/" and next_char == "/":
            i += 2
            while i < len(source) and source[i] != "\n":
                i += 1
            continue
        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(source) and not (source[i] == "*" and source[i + 1] == "/"):
                i += 1
            i += 2 if i + 1 < len(source) else 0
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_arguments_with_spans(source: str, open_paren: int, close_paren: int) -> tuple[ArgumentSpan, ...]:
    args_start = open_paren + 1
    raw = source[args_start:close_paren]
    args: list[ArgumentSpan] = []
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    arg_start = 0

    for offset, char in enumerate(raw):
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            continue

        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            _append_argument_span(args, raw, args_start, arg_start, offset)
            arg_start = offset + 1

    _append_argument_span(args, raw, args_start, arg_start, len(raw))
    return tuple(arg for arg in args if arg.text)


def _append_argument_span(args: list[ArgumentSpan], raw: str, absolute_start: int, start: int, end: int) -> None:
    text = raw[start:end]
    leading = len(text) - len(text.lstrip())
    trailing = len(text.rstrip())
    cleaned = text.strip()
    if not cleaned:
        return
    args.append(
        ArgumentSpan(
            text=cleaned,
            start=absolute_start + start + leading,
            end=absolute_start + start + trailing,
        )
    )


def find_calls(source: str, call_regex: str, name: str) -> tuple[CallSite, ...]:
    calls: list[CallSite] = []
    search_source = mask_java_comments_and_literals(source)
    for match in re.finditer(call_regex, search_source, flags=re.MULTILINE):
        open_paren = search_source.find("(", match.start(), match.end() + 1)
        if open_paren == -1:
            continue
        close_paren = find_matching_delimiter(source, open_paren)
        if close_paren == -1:
            continue
        calls.append(
            CallSite(
                name=name,
                start=match.start(),
                end=close_paren + 1,
                open_paren=open_paren,
                close_paren=close_paren,
                arguments=split_arguments_with_spans(source, open_paren, close_paren),
            )
        )
    return tuple(calls)


def iter_methods(source: str) -> tuple[MethodBody, ...]:
    pattern = re.compile(
        r"(?:public|private|protected|static|final|synchronized|native|\s)+"
        r"[\w.<>\[\], ?]+\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*"
        r"\((?P<params>[^)]*)\)\s*(?:throws\s+[\w.$<>\[\], ?]+\s*)?\{",
        flags=re.MULTILINE,
    )
    methods: list[MethodBody] = []
    for match in pattern.finditer(source):
        body_start = source.find("{", match.start(), match.end())
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_end == -1:
            continue
        methods.append(
            MethodBody(
                name=match.group("name"),
                params=parse_parameter_names(match.group("params")),
                start=match.start(),
                body_start=body_start,
                body_end=body_end,
                text=source[body_start + 1 : body_end],
            )
        )
    return tuple(methods)


def parse_parameter_names(params_text: str) -> tuple[str, ...]:
    params: list[str] = []
    fake_source = f"f({params_text})"
    open_paren = fake_source.find("(")
    close_paren = fake_source.rfind(")")
    for arg in split_arguments_with_spans(fake_source, open_paren, close_paren):
        param = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", arg.text)
        param = param.replace("final ", "").replace("...", "[]")
        parts = param.strip().split()
        if not parts:
            continue
        name = parts[-1].replace("[]", "").strip()
        if re.match(r"^[A-Za-z_]\w*$", name):
            params.append(name)
    return tuple(params)


def params_in_scope(source: str, index: int) -> tuple[str, ...]:
    for method in iter_methods(source):
        if method.body_start <= index <= method.body_end:
            return method.params
    return tuple(param for method in iter_methods(source) for param in method.params)


def statement_start(source: str, index: int) -> int:
    search_source = mask_java_comments_and_literals(source)
    candidates = [
        search_source.rfind(";", 0, index),
        search_source.rfind("{", 0, index),
        search_source.rfind("}", 0, index),
    ]
    start = max(candidates) + 1
    while start < len(source) and source[start] in " \t\r\n":
        start += 1
    return start


def line_indentation(source: str, index: int) -> str:
    line_start = source.rfind("\n", 0, index) + 1
    match = re.match(r"[ \t]*", source[line_start:index])
    return match.group(0) if match else ""


def insert_before_final_class_brace(source: str, text: str) -> str:
    index = source.rfind("}")
    if index == -1:
        raise ValueError("cannot insert helper: no class closing brace found")
    prefix = source[:index].rstrip()
    suffix = source[index:]
    return f"{prefix}\n\n{text.rstrip()}\n{suffix}"
