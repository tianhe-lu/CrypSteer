"""Validate isolated TypeScript dataset programs with the pinned native compiler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any


DIAGNOSTIC_RE = re.compile(
    r"^(.*)\((\d+),(\d+)\):\s+"
    r"(error|warning|message|suggestion)\s+TS(\d+):\s+(.*)$"
)
SUMMARY_RE = re.compile(r"^Found \d+ errors?(?: in .*?)?\.?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--base-tsconfig", type=Path, required=True)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_path = args.output_jsonl.resolve()
    work_root = args.work_root.resolve()
    base_tsconfig_path = args.base_tsconfig.resolve()
    require_directory(input_root)
    require_outside_input(input_root, output_path)
    require_outside_input(input_root, work_root)
    require_positive(args.batch_size, "--batch-size")
    require_positive(args.progress_every, "--progress-every")

    tooling_root = Path(__file__).resolve().parent
    native_tsc = resolve_native_tsc(tooling_root)
    type_roots = tooling_root / "node_modules" / "@types"
    package_path = tooling_root / "node_modules" / "typescript" / "package.json"
    if not base_tsconfig_path.is_file():
        raise RuntimeError(f"base TypeScript config is missing: {base_tsconfig_path}")
    if not package_path.is_file():
        raise RuntimeError(f"pinned TypeScript package is missing: {package_path}")

    base_tsconfig = read_mapping(base_tsconfig_path)
    compiler_options = base_tsconfig.get("compilerOptions")
    if not isinstance(compiler_options, dict):
        raise RuntimeError(
            f"base TypeScript config has no compilerOptions: {base_tsconfig_path}"
        )
    package = read_mapping(package_path)
    typescript_version = str(package.get("version", "unknown"))
    frontend = f"typescript-{typescript_version}-native-node-next-strict"

    source_paths = (
        read_file_list(args.file_list.resolve(), input_root)
        if args.file_list
        else collect_sources(input_root)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for offset in range(0, len(source_paths), args.batch_size):
            batch = source_paths[offset : offset + args.batch_size]
            batch_root = work_root / f"batch_{offset:08d}"
            try:
                records = validate_batch(
                    batch,
                    batch_root=batch_root,
                    input_root=input_root,
                    compiler_options=compiler_options,
                    type_roots=type_roots,
                    native_tsc=native_tsc,
                    frontend=frontend,
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                records = [
                    infrastructure_record(
                        source_path,
                        input_root=input_root,
                        native_tsc=native_tsc,
                        frontend=frontend,
                        message=message,
                    )
                    for source_path in batch
                ]
            finally:
                shutil.rmtree(batch_root, ignore_errors=True)

            for record in records:
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                completed += 1
                if completed % args.progress_every == 0 or completed == len(source_paths):
                    print(
                        f"  TypeScript validation progress: {completed}/{len(source_paths)}",
                        flush=True,
                    )

    print(f"Validated {len(source_paths)} TypeScript sources into {output_path}")


def validate_batch(
    batch: list[Path],
    *,
    batch_root: Path,
    input_root: Path,
    compiler_options: dict[str, Any],
    type_roots: Path,
    native_tsc: Path,
    frontend: str,
) -> list[dict[str, Any]]:
    if batch_root.exists():
        shutil.rmtree(batch_root)
    source_root = batch_root / "sources"
    source_root.mkdir(parents=True)
    (batch_root / "package.json").write_text(
        json.dumps({"private": True, "type": "module"}, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary_to_original: dict[Path, Path] = {}
    temporary_files: list[Path] = []
    for index, original in enumerate(batch):
        temporary = source_root / f"{index:04d}_{original.name}"
        shutil.copyfile(original, temporary)
        temporary = temporary.resolve()
        temporary_files.append(temporary)
        temporary_to_original[temporary] = original

    invocation = invoke_typescript_compiler(
        temporary_files,
        tsconfig_path=batch_root / "tsconfig.json",
        batch_root=batch_root,
        compiler_options=compiler_options,
        type_roots=type_roots,
        native_tsc=native_tsc,
    )
    diagnostics_by_original, infrastructure_messages = parse_diagnostics(
        f"{invocation.stdout}\n{invocation.stderr}",
        temporary_to_original=temporary_to_original,
        diagnostic_root=batch_root,
    )
    if invocation.returncode != 0 and not diagnostics_by_original:
        infrastructure_messages.append(
            f"TypeScript compiler exited {invocation.returncode} without a source diagnostic"
        )

    parse_failed = {
        original
        for original, diagnostics in diagnostics_by_original.items()
        if any(is_parse_error(item) for item in diagnostics)
    }
    semantic_files = [
        temporary
        for temporary in temporary_files
        if temporary_to_original[temporary] not in parse_failed
    ]
    if parse_failed and semantic_files:
        semantic_invocation = invoke_typescript_compiler(
            semantic_files,
            tsconfig_path=batch_root / "tsconfig.semantic.json",
            batch_root=batch_root,
            compiler_options=compiler_options,
            type_roots=type_roots,
            native_tsc=native_tsc,
        )
        semantic_diagnostics, semantic_infrastructure = parse_diagnostics(
            f"{semantic_invocation.stdout}\n{semantic_invocation.stderr}",
            temporary_to_original=temporary_to_original,
            diagnostic_root=batch_root,
        )
        if semantic_invocation.returncode != 0 and not semantic_diagnostics:
            semantic_infrastructure.append(
                "TypeScript semantic compiler pass exited "
                f"{semantic_invocation.returncode} without a source diagnostic"
            )
        diagnostics_by_original = {
            original: diagnostics
            for original, diagnostics in diagnostics_by_original.items()
            if original in parse_failed
        }
        diagnostics_by_original.update(semantic_diagnostics)
        infrastructure_messages.extend(semantic_infrastructure)

    records: list[dict[str, Any]] = []
    for original in batch:
        diagnostics = list(diagnostics_by_original.get(original, ()))
        diagnostics.extend(
            {
                "code": "typescript_batch_infrastructure",
                "severity": "error",
                "message": message,
                "line": None,
                "column": None,
                "end_line": None,
                "end_column": None,
            }
            for message in infrastructure_messages
        )
        errors = [item for item in diagnostics if item["severity"] == "error"]
        parse_success = not any(is_parse_error(item) for item in errors)
        records.append(
            {
                "relative_path": relative_record_path(original, input_root),
                "attempted": True,
                "success": not errors,
                "parse_success": parse_success,
                "frontend": frontend,
                "command": [
                    str(native_tsc),
                    "--project",
                    "<batch-tsconfig>",
                    "--pretty",
                    "false",
                ],
                "exit_code": invocation.returncode,
                "stdout": "",
                "stderr": "",
                "diagnostics": diagnostics,
                "infrastructure_error": (
                    "\n".join(infrastructure_messages)
                    if infrastructure_messages
                    else None
                ),
                "reason_code": (
                    "typescript_typecheck_passed"
                    if not errors
                    else "typescript_typecheck_failed"
                ),
            }
        )
    return records


def invoke_typescript_compiler(
    source_paths: list[Path],
    *,
    tsconfig_path: Path,
    batch_root: Path,
    compiler_options: dict[str, Any],
    type_roots: Path,
    native_tsc: Path,
) -> subprocess.CompletedProcess[str]:
    tsconfig_path.write_text(
        json.dumps(
            {
                "compilerOptions": {
                    **compiler_options,
                    "typeRoots": [str(type_roots.resolve())],
                    "noEmit": True,
                    "moduleDetection": "force",
                },
                "files": [str(path) for path in source_paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [
            str(native_tsc),
            "--project",
            str(tsconfig_path.resolve()),
            "--pretty",
            "false",
        ],
        cwd=batch_root,
        check=False,
        capture_output=True,
        text=True,
    )


def parse_diagnostics(
    text: str,
    *,
    temporary_to_original: dict[Path, Path],
    diagnostic_root: Path,
) -> tuple[dict[Path, list[dict[str, Any]]], list[str]]:
    by_original: dict[Path, list[dict[str, Any]]] = {}
    infrastructure_messages: list[str] = []
    last_record: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or SUMMARY_RE.fullmatch(line.strip()):
            continue
        match = DIAGNOSTIC_RE.fullmatch(line)
        if match is None:
            if line[:1].isspace() and last_record is not None:
                last_record["message"] += f"\n{line.strip()}"
                continue
            infrastructure_messages.append(line)
            last_record = None
            continue
        reported = Path(match.group(1))
        temporary = (
            reported.resolve()
            if reported.is_absolute()
            else (diagnostic_root / reported).resolve()
        )
        original = temporary_to_original.get(temporary)
        if original is None:
            infrastructure_messages.append(line)
            last_record = None
            continue
        record: dict[str, Any] = {
            "code": f"TS{match.group(5)}",
            "severity": match.group(4).lower(),
            "message": match.group(6),
            "line": int(match.group(2)),
            "column": int(match.group(3)),
            "end_line": None,
            "end_column": None,
        }
        by_original.setdefault(original, []).append(record)
        last_record = record
    return by_original, infrastructure_messages


def infrastructure_record(
    source_path: Path,
    *,
    input_root: Path,
    native_tsc: Path,
    frontend: str,
    message: str,
) -> dict[str, Any]:
    return {
        "relative_path": relative_record_path(source_path, input_root),
        "attempted": False,
        "success": False,
        "parse_success": False,
        "frontend": frontend,
        "command": [
            str(native_tsc),
            "--project",
            "<batch-tsconfig>",
            "--pretty",
            "false",
        ],
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "diagnostics": [],
        "infrastructure_error": message,
        "reason_code": "typescript_batch_infrastructure_error",
    }


def resolve_native_tsc(tooling_root: Path) -> Path:
    system = platform.system().lower()
    platform_name = {"windows": "win32", "darwin": "darwin", "linux": "linux"}.get(
        system,
        system,
    )
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    executable = "tsc.exe" if os.name == "nt" else "tsc"
    candidate = (
        tooling_root
        / "node_modules"
        / "@typescript"
        / f"typescript-{platform_name}-{architecture}"
        / "lib"
        / executable
    )
    if not candidate.is_file():
        raise RuntimeError(f"pinned native TypeScript compiler is missing: {candidate}")
    return candidate.resolve()


def read_file_list(path: Path, root: Path) -> list[Path]:
    records = sorted(
        {
            Path(line.strip()).resolve()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    for source_path in records:
        require_source_within_root(source_path, root)
    return records


def collect_sources(root: Path) -> list[Path]:
    records = sorted(path.resolve() for path in root.rglob("source.ts"))
    for source_path in records:
        require_source_within_root(source_path, root)
    return records


def require_source_within_root(source_path: Path, root: Path) -> None:
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"TypeScript source is outside input root: {source_path}") from error
    if source_path.suffix.lower() != ".ts" or not source_path.is_file():
        raise RuntimeError(f"TypeScript source is invalid: {source_path}")


def require_outside_input(input_root: Path, candidate: Path) -> None:
    try:
        candidate.relative_to(input_root)
    except ValueError:
        return
    raise RuntimeError(f"validation output must be outside the dataset: {candidate}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"directory does not exist: {path}")


def require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")


def read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def relative_record_path(source_path: Path, input_root: Path) -> str:
    return f"typescript/{source_path.relative_to(input_root).as_posix()}"


def is_parse_error(diagnostic: dict[str, Any]) -> bool:
    code = str(diagnostic.get("code", ""))
    if not code.startswith("TS") or not code[2:].isdigit():
        return False
    number = int(code[2:])
    return 1000 <= number < 2000


if __name__ == "__main__":
    main()
