"""Pinned, fail-soft language frontends for multilingual dataset artifacts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable

from crypto_steering.validation.paths import project_root, resolve_project_path
from crypto_steering.validation.frontends.models import FrontendRuntimeConfig
from crypto_steering.validation.models import (
    CompileOutcome,
    ProgramInput,
    Diagnostic,
    ImportFinding,
    ImportOutcome,
)
from crypto_steering.progress import manual_progress, progress


_TYPESCRIPT_IMPORT_RE = re.compile(
    r"(?:\bfrom\s*|\brequire\s*\(\s*|\bimport\s*\(\s*|\bimport\s*)"
    r"(?P<quote>['\"])(?P<module>[^'\"]+)(?P=quote)",
)
_CSHARP_USING_RE = re.compile(
    r"(?m)^\s*(?:global\s+)?using\s+(?:static\s+)?"
    r"(?:[A-Za-z_]\w*\s*=\s*)?(?:global::)?"
    r"(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;"
)
_JAVA_IMPORT_RE = re.compile(
    r"(?m)^\s*import\s+(?:static\s+)?(?P<module>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)"
    r"(?:\.\*)?\s*;"
)

# Node's built-in module names accepted with or without the node: prefix.
_NODE_BUILTINS = frozenset(
    {
        "assert",
        "assert/strict",
        "async_hooks",
        "buffer",
        "child_process",
        "cluster",
        "console",
        "constants",
        "crypto",
        "dgram",
        "diagnostics_channel",
        "dns",
        "dns/promises",
        "domain",
        "events",
        "fs",
        "fs/promises",
        "http",
        "http2",
        "https",
        "module",
        "net",
        "os",
        "path",
        "path/posix",
        "path/win32",
        "perf_hooks",
        "process",
        "punycode",
        "querystring",
        "readline",
        "readline/promises",
        "repl",
        "stream",
        "stream/consumers",
        "stream/promises",
        "stream/web",
        "string_decoder",
        "sys",
        "timers",
        "timers/promises",
        "tls",
        "trace_events",
        "tty",
        "url",
        "util",
        "util/types",
        "v8",
        "vm",
        "wasi",
        "worker_threads",
        "zlib",
    }
)


@dataclass(frozen=True)
class FrontendAudit:
    compile_by_program: dict[str, CompileOutcome]
    imports_by_program: dict[str, ImportOutcome]
    errors: tuple[dict[str, Any], ...]


def validate_frontends(
    config: FrontendRuntimeConfig,
    artifacts: Iterable[ProgramInput],
    *,
    work_dir: Path,
    logs_dir: Path,
) -> FrontendAudit:
    """Validate every artifact, retaining a result even after tool failures."""

    grouped: dict[str, list[ProgramInput]] = {
        "java": [],
        "typescript": [],
        "csharp": [],
        "python": [],
    }
    artifacts_by_program: dict[str, ProgramInput] = {}
    for artifact in artifacts:
        previous = artifacts_by_program.get(artifact.program_id)
        if previous is not None:
            raise RuntimeError(
                "frontend artifact program_id must be unique: "
                f"{artifact.program_id} maps to both "
                f"{previous.source_path} and {artifact.source_path}"
            )
        artifacts_by_program[artifact.program_id] = artifact
        grouped[artifact.language].append(artifact)

    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    compile_results: dict[str, CompileOutcome] = {}
    import_results: dict[str, ImportOutcome] = {}
    errors: list[dict[str, Any]] = []

    for language, members in grouped.items():
        for artifact in progress(
            members,
            total=len(members),
            desc=f"{language} import checks",
            unit="program",
            dynamic_ncols=True,
            leave=False,
        ):
            try:
                import_results[artifact.program_id] = validate_imports(
                    artifact.language,
                    artifact.source,
                    config,
                )
            except Exception as error:
                import_results[artifact.program_id] = ImportOutcome(
                    attempted=False,
                    success=False,
                    reason_code="analysis_error",
                )
                errors.append(
                    _error_record(
                        config,
                        language=language,
                        source_path=str(artifact.source_path),
                        error_type=type(error).__name__,
                        error_message=str(error),
                        phase="import_analysis",
                        program_id=artifact.program_id,
                    )
                )
        if not members:
            continue
        try:
            if language == "python":
                environment_error = _python_environment_error(config)
                if environment_error:
                    language_results = {
                        artifact.program_id: _python_environment_failure(
                            environment_error
                        )
                        for artifact in members
                    }
                else:
                    language_results = {}
                    for artifact in progress(
                        members,
                        total=len(members),
                        desc="python frontend",
                        unit="program",
                        dynamic_ncols=True,
                        leave=False,
                    ):
                        language_results[artifact.program_id] = _validate_python(
                            artifact,
                            config,
                        )
            else:
                language_results = _validate_external_language(
                    config,
                    language,
                    members,
                    work_dir=work_dir / language,
                    log_path=logs_dir / f"frontend_{language}.log",
                )
        except Exception as error:  # Infrastructure failure remains fail-soft.
            errors.append(
                _error_record(
                    config,
                    language=language,
                    source_path="",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    phase="frontend_batch",
                )
            )
            language_results = {}

        for artifact in members:
            result = language_results.get(artifact.program_id)
            if result is None:
                result = CompileOutcome(
                    attempted=False,
                    success=False,
                    parse_success=False,
                    frontend=f"{language}_frontend",
                    reason_code="analysis_error",
                    stderr="frontend batch produced no result for this artifact",
                )
                errors.append(
                    _error_record(
                        config,
                        language=language,
                        source_path=str(artifact.source_path),
                        error_type="MissingFrontendResult",
                        error_message="frontend batch produced no result",
                        phase="frontend_result_join",
                        program_id=artifact.program_id,
                    )
                )
            compile_results[artifact.program_id] = result

    return FrontendAudit(
        compile_by_program=compile_results,
        imports_by_program=import_results,
        errors=tuple(errors),
    )


def validate_imports(
    language: str,
    source: str,
    config: FrontendRuntimeConfig,
) -> ImportOutcome:
    try:
        modules = _source_imports(language, source)
    except SyntaxError:
        return ImportOutcome(
            attempted=True,
            success=False,
            reason_code="syntax_error_prevented_import_analysis",
        )
    findings: list[ImportFinding] = []
    for module in modules:
        approved, resolved = _import_status(language, module, config)
        findings.append(
            ImportFinding(
                module=module,
                imported_name=None,
                approved=approved,
                resolved=resolved,
                reason_code=(
                    "approved_platform_dependency"
                    if approved and resolved
                    else "unresolved_approved_dependency"
                    if approved
                    else "unsupported_dependency"
                ),
            )
        )
    if language == "python":
        for module, imported_name in _python_unresolved_imported_names(source):
            findings.append(
                ImportFinding(
                    module=module,
                    imported_name=imported_name,
                    approved=True,
                    resolved=False,
                    reason_code="unresolved_approved_dependency",
                )
            )
        for module, imported_name in _python_private_cryptography_imports(source):
            findings.append(
                ImportFinding(
                    module=module,
                    imported_name=imported_name,
                    approved=False,
                    resolved=False,
                    reason_code="unsupported_private_dependency",
                )
            )
    success = all(item.approved and item.resolved for item in findings)
    return ImportOutcome(
        attempted=True,
        success=success,
        findings=tuple(findings),
        reason_code=(
            "imports_approved"
            if success
            else "unsupported_dependency"
            if any(not item.approved for item in findings)
            else "unresolved_approved_dependency"
        ),
    )


def _validate_python(
    artifact: ProgramInput,
    config: FrontendRuntimeConfig,
) -> CompileOutcome:
    diagnostics: list[Diagnostic] = []
    expected_version = str(
        config.toolchains["python"].raw.get("python_version", "3.11")
    )
    expected_parts = tuple(int(part) for part in expected_version.split(".")[:2])
    if tuple(sys.version_info[:2]) != expected_parts:
        message = (
            f"configured CPython {expected_version}.x is required, found "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        return CompileOutcome(
            attempted=False,
            success=False,
            parse_success=False,
            frontend="cpython_compile_3_11",
            stderr=message,
            diagnostics=(
                Diagnostic(
                    code="PYTHON_VERSION_MISMATCH",
                    severity="error",
                    message=message,
                ),
            ),
            reason_code="analysis_error",
        )
    try:
        compile(
            artifact.source,
            str(artifact.source_path),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
    except SyntaxError as error:
        diagnostics.append(
            Diagnostic(
                code="PY_SYNTAX_ERROR",
                severity="error",
                message=str(error.msg),
                line=error.lineno,
                column=error.offset,
                end_line=error.end_lineno,
                end_column=error.end_offset,
            )
        )
        return CompileOutcome(
            attempted=True,
            success=False,
            parse_success=False,
            frontend="cpython_compile_3_11",
            exit_code=1,
            diagnostics=tuple(diagnostics),
            reason_code="syntax_error",
        )
    except Exception as error:
        diagnostics.append(
            Diagnostic(
                code="PY_COMPILE_ERROR",
                severity="error",
                message=f"{type(error).__name__}: {error}",
            )
        )
        return CompileOutcome(
            attempted=True,
            success=False,
            parse_success=False,
            frontend="cpython_compile_3_11",
            exit_code=1,
            diagnostics=tuple(diagnostics),
            reason_code="analysis_error",
        )
    return CompileOutcome(
        attempted=True,
        success=True,
        parse_success=True,
        frontend="cpython_compile_3_11",
        exit_code=0,
        diagnostics=(),
        reason_code="compile_success",
    )


def _python_environment_error(
    config: FrontendRuntimeConfig,
) -> str | None:
    raw = config.toolchains["python"].raw
    expected_python = str(raw.get("python_version", "3.11"))
    expected_parts = tuple(int(part) for part in expected_python.split(".")[:2])
    if tuple(sys.version_info[:2]) != expected_parts:
        return (
            f"configured CPython {expected_python}.x is required, found "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
    expected_cryptography = str(raw.get("cryptography_version", ""))
    if expected_cryptography:
        try:
            actual_cryptography = importlib.metadata.version("cryptography")
        except importlib.metadata.PackageNotFoundError:
            return "configured cryptography package is not installed"
        if actual_cryptography != expected_cryptography:
            return (
                f"configured cryptography {expected_cryptography} is required, "
                f"found {actual_cryptography}"
            )
    return None


def _python_environment_failure(message: str) -> CompileOutcome:
    return CompileOutcome(
        attempted=False,
        success=False,
        parse_success=False,
        frontend="cpython_compile_3_11",
        stderr=message,
        diagnostics=(
            Diagnostic(
                code="PYTHON_ENVIRONMENT_MISMATCH",
                severity="error",
                message=message,
            ),
        ),
        reason_code="analysis_error",
    )


def _validate_external_language(
    config: FrontendRuntimeConfig,
    language: str,
    artifacts: list[ProgramInput],
    *,
    work_dir: Path,
    log_path: Path,
) -> dict[str, CompileOutcome]:
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "frontend_input.tsv"
    output_path = work_dir / "frontend_output.jsonl"
    source_root = _frontend_source_root(language, artifacts)
    _write_frontend_input(input_path, artifacts, language=language)
    command = _frontend_command(
        config,
        language,
        input_path=input_path,
        output_path=output_path,
        work_dir=work_dir,
        source_root=source_root,
    )
    return_code = _run_and_tee(
        command,
        log_path=log_path,
        cwd=_frontend_working_directory(config, language),
        progress_total=len(artifacts),
        progress_description=f"{language} frontend",
        progress_value=lambda line: _frontend_progress_value(language, line),
    )
    raw_results = _read_external_results(output_path, artifacts)
    results: dict[str, CompileOutcome] = {}
    for program_id, raw in raw_results.items():
        results[program_id] = _external_outcome(raw, command)
    if return_code != 0:
        # Preserve any complete records written before a batch-level failure.
        for artifact in artifacts:
            results.setdefault(
                artifact.program_id,
                CompileOutcome(
                    attempted=False,
                    success=False,
                    parse_success=False,
                    frontend=f"{language}_frontend",
                    command=tuple(command),
                    exit_code=return_code,
                    stderr=f"frontend process exited with status {return_code}",
                    reason_code="analysis_error",
                ),
            )
    return results


def _frontend_command(
    config: FrontendRuntimeConfig,
    language: str,
    *,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    source_root: Path,
) -> list[str]:
    raw = config.toolchains[language].raw
    root = project_root()
    if language == "java":
        javac = str(raw.get("javac_command", "javac"))
        java = str(raw.get("java_command", "java"))
        release = str(raw.get("release", "21"))
        helper_source = resolve_project_path(
            str(raw.get("compiler_script", "scripts/tooling/java/ValidateDatasetTree.java")),
            root,
        )
        classes = work_dir / "helper_classes"
        classes.mkdir(parents=True, exist_ok=True)
        compile_command = [
            javac,
            "--release",
            release,
            "-d",
            str(classes),
            str(helper_source),
        ]
        compile_log = work_dir / "helper_compile.log"
        compile_status = _run_and_tee(compile_command, log_path=compile_log)
        if compile_status != 0:
            raise RuntimeError(
                f"Java frontend helper compilation failed with status {compile_status}"
            )
        return [
            java,
            "-cp",
            str(classes),
            "ValidateDatasetTree",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--release",
            release,
            "--progress-every",
            "500",
        ]
    if language == "typescript":
        node = str(raw.get("node_command", "node"))
        script = resolve_project_path(str(raw["compiler_script"]), root)
        tsconfig = resolve_project_path(str(raw["tsconfig_path"]), root)
        launcher = [sys.executable] if script.suffix == ".py" else [node]
        return [
            *launcher,
            str(script),
            "--input-root",
            str(source_root),
            "--output-jsonl",
            str(output_path),
            "--work-root",
            str(work_dir / "typescript_batches"),
            "--base-tsconfig",
            str(tsconfig),
            "--file-list",
            str(input_path),
            "--batch-size",
            str(raw.get("batch_size", 250)),
            "--progress-every",
            "500",
        ]
    if language == "csharp":
        dotnet = _resolve_dotnet_command(str(raw.get("dotnet_command", "dotnet")))
        project = resolve_project_path(str(raw["tooling_project"]), root)
        return [
            dotnet,
            "run",
            "--project",
            str(project),
            "--configuration",
            "Release",
            "--no-restore",
            "--",
            "validate-tree",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--language-version",
            str(raw.get("language_version", "14.0")),
            "--progress-every",
            "500",
        ]
    raise ValueError(f"no external frontend for language {language}")


def _frontend_source_root(
    language: str,
    artifacts: Iterable[ProgramInput],
) -> Path:
    """Derive the language root from the artifacts being validated.

    Validation is also used for staged and cleaned candidate trees. Using the
    dataset root from the base config would make those candidates appear to be
    outside the TypeScript input tree.
    """

    roots: set[Path] = set()
    for artifact in artifacts:
        relative = Path(artifact.relative_path)
        if len(relative.parts) < 2 or relative.parts[0] != language:
            raise ValueError(
                f"artifact relative path does not begin with {language}: "
                f"{artifact.relative_path}"
            )
        within_language = Path(*relative.parts[1:])
        source_path = artifact.source_path.resolve()
        root = source_path.parents[len(within_language.parts) - 1]
        if (root / within_language).resolve() != source_path:
            raise ValueError(
                "artifact source path and relative path identify different trees: "
                f"{artifact.source_path} vs {artifact.relative_path}"
            )
        roots.add(root)
    if len(roots) != 1:
        raise ValueError(
            f"{language} frontend artifacts span {len(roots)} source roots"
        )
    return next(iter(roots))


def _resolve_dotnet_command(configured: str) -> str:
    """Find a user-local SDK when ``dotnet`` is absent from this process PATH."""

    expanded = Path(configured).expanduser()
    if expanded.parent != Path("."):
        return str(expanded)
    discovered = shutil.which(configured)
    if discovered is not None or configured != "dotnet":
        return discovered or configured

    candidates: list[Path] = []
    dotnet_root = os.environ.get("DOTNET_ROOT")
    if dotnet_root:
        candidates.append(Path(dotnet_root).expanduser() / "dotnet")
    candidates.append(Path.home() / ".dotnet" / "dotnet")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return configured


def _write_frontend_input(
    path: Path,
    artifacts: Iterable[ProgramInput],
    *,
    language: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for artifact in artifacts:
            if language == "typescript":
                handle.write(f"{artifact.source_path.resolve()}\n")
                continue
            if "\t" in artifact.program_id or "\t" in str(artifact.source_path):
                raise ValueError("frontend input fields must not contain tabs")
            handle.write(f"{artifact.program_id}\t{artifact.source_path.resolve()}\n")


def _frontend_working_directory(
    config: FrontendRuntimeConfig,
    language: str,
) -> Path:
    if language == "csharp":
        project = resolve_project_path(
            str(config.toolchains[language].raw["tooling_project"]),
            project_root(),
        )
        return project.parent
    return project_root()


def _run_and_tee(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    progress_total: int | None = None,
    progress_description: str | None = None,
    progress_value: Callable[[str], int | None] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with manual_progress(
        total=progress_total,
        desc=progress_description,
        unit="program",
        dynamic_ncols=True,
        leave=False,
    ) as progress_bar:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                completed = progress_value(line) if progress_value else None
                if completed is not None and progress_bar.enabled:
                    progress_bar.update_to(
                        min(completed, progress_total or completed)
                    )
                else:
                    print(line, end="", flush=True)
            return process.wait()


def _frontend_progress_value(language: str, line: str) -> int | None:
    patterns = {
        "java": r"Java frontend (?:progress|completed):\s*(\d+)",
        "typescript": r"TypeScript validation progress:\s*(\d+)(?:/\d+)?",
        "csharp": r"C# frontend progress:\s*(\d+)",
    }
    pattern = patterns.get(language)
    if pattern is None:
        return None
    match = re.search(pattern, line)
    return int(match.group(1)) if match else None


def _read_external_results(
    path: Path,
    artifacts: Iterable[ProgramInput],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_relative_path = {
        artifact.relative_path: artifact.program_id for artifact in artifacts
    }
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid frontend JSONL at {path}:{line_number}: {error}"
                ) from error
            program_id = row.get("program_id")
            if not program_id and isinstance(row.get("relative_path"), str):
                program_id = by_relative_path.get(str(row["relative_path"]))
            if not isinstance(program_id, str) or not program_id:
                raise ValueError(
                    f"frontend row lacks program_id at {path}:{line_number}"
                )
            if program_id in results:
                raise ValueError(f"duplicate frontend result for {program_id}")
            results[program_id] = row
    return results


def _external_outcome(raw: dict[str, Any], command: list[str]) -> CompileOutcome:
    diagnostics = tuple(
        Diagnostic(
            code=str(item.get("code", "UNKNOWN")),
            severity=str(item.get("severity", "error")),
            message=str(item.get("message", "")),
            line=_optional_int(item.get("line")),
            column=_optional_int(item.get("column")),
            end_line=_optional_int(item.get("end_line")),
            end_column=_optional_int(item.get("end_column")),
        )
        for item in raw.get("diagnostics", [])
        if isinstance(item, dict)
    )
    infrastructure_error = raw.get("infrastructure_error")
    success = bool(raw.get("success", False)) and not infrastructure_error
    parse_success = bool(raw.get("parse_success", False))
    return CompileOutcome(
        attempted=bool(raw.get("attempted", True)),
        success=success,
        parse_success=parse_success,
        frontend=str(raw.get("frontend", "external_frontend")),
        command=tuple(command),
        exit_code=_optional_int(raw.get("exit_code")),
        stderr=str(infrastructure_error or ""),
        diagnostics=diagnostics,
        reason_code=_compile_reason(
            success=success,
            parse_success=parse_success,
            diagnostics=diagnostics,
            infrastructure_error=str(infrastructure_error or ""),
        ),
    )


def _compile_reason(
    *,
    success: bool,
    parse_success: bool,
    diagnostics: tuple[Diagnostic, ...],
    infrastructure_error: str,
) -> str:
    if success:
        return "compile_success"
    if infrastructure_error:
        return "analysis_error"
    if not parse_success:
        return "syntax_error"
    codes = " ".join(item.code.lower() for item in diagnostics)
    messages = " ".join(item.message.lower() for item in diagnostics)
    combined = f"{codes} {messages}"
    if any(
        marker in combined
        for marker in (
            "ts2307",
            "cs0234",
            "package does not exist",
            "module not found",
            "cannot find module",
        )
    ):
        return "module_or_namespace_error"
    if any(
        marker in combined
        for marker in (
            "cant.resolve",
            "ts2304",
            "ts2552",
            "cs0103",
            "cs0246",
            "cannot find symbol",
            "could not be found",
            "cannot find name",
        )
    ):
        return "unresolved_symbol"
    if any(
        marker in combined
        for marker in (
            "incompatible.types",
            "cant.apply.symbol",
            "ts2322",
            "ts2345",
            "ts2554",
            "cs1501",
            "cs1503",
            "cannot be applied",
            "not assignable",
        )
    ):
        return "type_or_overload_error"
    return "other_compile_error"


def _source_imports(language: str, source: str) -> tuple[str, ...]:
    if language == "typescript":
        return tuple(
            sorted({match.group("module") for match in _TYPESCRIPT_IMPORT_RE.finditer(source)})
        )
    if language == "csharp":
        return tuple(
            sorted({match.group("module") for match in _CSHARP_USING_RE.finditer(source)})
        )
    if language == "java":
        return tuple(
            sorted({match.group("module") for match in _JAVA_IMPORT_RE.finditer(source)})
        )
    if language == "python":
        tree = ast.parse(source)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    modules.add("." * node.level + (node.module or ""))
                elif node.module:
                    modules.add(node.module)
        return tuple(sorted(modules))
    raise ValueError(f"unsupported language {language}")


def _python_private_cryptography_imports(
    source: str,
) -> tuple[tuple[str, str | None], ...]:
    tree = ast.parse(source)
    findings: set[tuple[str, str | None]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "cryptography" and any(
                    part.startswith("_") for part in parts[1:]
                ):
                    findings.add((alias.name, None))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".", 1)[0] == "cryptography"
        ):
            module = node.module or "cryptography"
            module_is_private = any(
                part.startswith("_") for part in module.split(".")[1:]
            )
            for alias in node.names:
                if module_is_private or alias.name.startswith("_"):
                    findings.add((module, alias.name))
    return tuple(sorted(findings, key=lambda item: (item[0], item[1] or "")))


def _python_unresolved_imported_names(
    source: str,
) -> tuple[tuple[str, str], ...]:
    """Return unavailable names imported from approved Python modules."""

    tree = ast.parse(source)
    findings: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
        ):
            continue
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                continue
            if not _python_imported_name_resolves(node.module, alias.name):
                findings.add((node.module, alias.name))
    return tuple(sorted(findings))


@lru_cache(maxsize=None)
def _python_imported_name_resolves(module: str, imported_name: str) -> bool:
    candidate = f"{module}.{imported_name}"
    if _python_module_resolves(candidate):
        return True
    try:
        imported_module = importlib.import_module(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    return hasattr(imported_module, imported_name)


def _import_status(
    language: str,
    module: str,
    config: FrontendRuntimeConfig,
) -> tuple[bool, bool]:
    if language == "java":
        approved = module == "java" or module.startswith(("java.", "javax."))
        return approved, approved
    if language == "csharp":
        approved = module == "System" or module.startswith("System.")
        return approved, approved
    if language == "typescript":
        normalized = module[5:] if module.startswith("node:") else module
        approved = normalized in _NODE_BUILTINS
        return approved, approved
    if language == "python":
        if module.startswith("."):
            return False, False
        root = module.split(".", 1)[0]
        approved = config.toolchains["python"].raw.get(
            "approved_import_roots",
            ["cryptography"],
        )
        permitted = root in sys.stdlib_module_names or root in {
            str(item) for item in approved
        }
        if not permitted:
            return False, False
        resolved = _python_module_resolves(module)
        return True, resolved
    return False, False


@lru_cache(maxsize=None)
def _python_module_resolves(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # ``os.path`` is an importable stdlib alias even though ``os`` is not a
        # package and therefore has no nested module spec.
        return module == "os.path"


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error_record(
    config: FrontendRuntimeConfig,
    *,
    language: str,
    source_path: str,
    error_type: str,
    error_message: str,
    phase: str,
    program_id: str | None = None,
) -> dict[str, Any]:
    return {
        "version": config.version,
        "source_config_path": config.source_config_path,
        "phase": phase,
        "program_id": program_id,
        "language": language,
        "source_path": source_path,
        "error_type": error_type,
        "error_message": error_message,
    }
