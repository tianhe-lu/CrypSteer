import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";


const options = parseArgs(process.argv.slice(2));
const inputRoot = path.resolve(required(options, "--input-root"));
const outputJsonl = path.resolve(required(options, "--output-jsonl"));
const workRoot = path.resolve(required(options, "--work-root"));
const baseTsconfigPath = path.resolve(required(options, "--base-tsconfig"));
const fileListPath = options.get("--file-list");
const batchSize = positiveInteger(options.get("--batch-size") ?? "250", "--batch-size");
const progressEvery = positiveInteger(
    options.get("--progress-every") ?? "500",
    "--progress-every",
);

requireDirectory(inputRoot);
requireOutsideInput(inputRoot, outputJsonl);
requireOutsideInput(inputRoot, workRoot);

const scriptRoot = path.dirname(fileURLToPath(import.meta.url));
const tscPath = path.join(scriptRoot, "node_modules", "typescript", "bin", "tsc");
const nativeTscPath = path.join(
    scriptRoot,
    "node_modules",
    "@typescript",
    `typescript-${process.platform}-${process.arch}`,
    "lib",
    process.platform === "win32" ? "tsc.exe" : "tsc",
);
const typeRoots = path.join(scriptRoot, "node_modules", "@types");
const typescriptPackagePath = path.join(
    scriptRoot,
    "node_modules",
    "typescript",
    "package.json",
);
const toolingPackagePath = path.join(scriptRoot, "package.json");
if (!fs.statSync(tscPath, { throwIfNoEntry: false })?.isFile()) {
    throw new Error(`Pinned TypeScript compiler is missing: ${tscPath}`);
}
if (!fs.statSync(nativeTscPath, { throwIfNoEntry: false })?.isFile()) {
    throw new Error(`Pinned native TypeScript compiler is missing: ${nativeTscPath}`);
}
if (!fs.statSync(baseTsconfigPath, { throwIfNoEntry: false })?.isFile()) {
    throw new Error(`Base TypeScript config is missing: ${baseTsconfigPath}`);
}
const baseTsconfig = JSON.parse(fs.readFileSync(baseTsconfigPath, "utf8"));
if (
    baseTsconfig === null
    || typeof baseTsconfig !== "object"
    || baseTsconfig.compilerOptions === null
    || typeof baseTsconfig.compilerOptions !== "object"
) {
    throw new Error(`Base TypeScript config has no compilerOptions: ${baseTsconfigPath}`);
}
const typescriptPackage = JSON.parse(
    fs.readFileSync(typescriptPackagePath, "utf8"),
);
const typescriptVersion = String(typescriptPackage.version ?? "unknown");
const toolingPackage = JSON.parse(fs.readFileSync(toolingPackagePath, "utf8"));
const expectedNodeVersion = String(toolingPackage.engines?.node ?? "");
if (!expectedNodeVersion || process.versions.node !== expectedNodeVersion) {
    throw new Error(
        `Pinned Node ${expectedNodeVersion || "<missing>"} is required, found ${process.versions.node}`,
    );
}
const frontendName = `typescript-${typescriptVersion}-node-${process.versions.node}-node-next-strict`;

const sourcePaths = fileListPath
    ? readFileList(path.resolve(fileListPath), inputRoot)
    : collectSources(inputRoot);
fs.mkdirSync(path.dirname(outputJsonl), { recursive: true });
fs.mkdirSync(workRoot, { recursive: true });
const output = fs.createWriteStream(outputJsonl, { encoding: "utf8" });

let completed = 0;
for (let offset = 0; offset < sourcePaths.length; offset += batchSize) {
    const batch = sourcePaths.slice(offset, offset + batchSize);
    const batchRoot = path.join(workRoot, `batch_${String(offset).padStart(8, "0")}`);
    let batchRecords;
    try {
        batchRecords = validateBatch(batch, batchRoot);
    } catch (error) {
        const message = `${error?.name ?? "Error"}: ${error?.message ?? String(error)}`;
        batchRecords = batch.map((original) => ({
            relative_path: `typescript/${normalizePath(path.relative(inputRoot, original))}`,
            attempted: false,
            success: false,
            parse_success: false,
            frontend: frontendName,
            command: [nativeTscPath, "--project", "<batch-tsconfig>", "--pretty", "false"],
            exit_code: null,
            stdout: "",
            stderr: "",
            diagnostics: [],
            infrastructure_error: message,
            reason_code: "typescript_batch_infrastructure_error",
        }));
    } finally {
        try {
            fs.rmSync(batchRoot, { recursive: true, force: true });
        } catch (error) {
            process.stderr.write(
                `  TypeScript validation cleanup warning for ${batchRoot}: ${error.message}\n`,
            );
        }
    }

    for (const record of batchRecords) {
        output.write(JSON.stringify(record) + "\n");
        completed += 1;
        if (completed % progressEvery === 0 || completed === sourcePaths.length) {
            process.stderr.write(
                `  TypeScript validation progress: ${completed}/${sourcePaths.length}\n`,
            );
        }
    }
}

await new Promise((resolve, reject) => {
    output.end(resolve);
    output.on("error", reject);
});
process.stdout.write(
    `Validated ${sourcePaths.length} TypeScript sources into ${outputJsonl}\n`,
);


function validateBatch(batch, batchRoot) {
    const sourceRoot = path.join(batchRoot, "sources");
    fs.rmSync(batchRoot, { recursive: true, force: true });
    fs.mkdirSync(sourceRoot, { recursive: true });
    fs.writeFileSync(
        path.join(batchRoot, "package.json"),
        JSON.stringify({ private: true, type: "module" }, null, 2) + "\n",
        "utf8",
    );

    const temporaryToOriginal = new Map();
    const originalToRelative = new Map();
    const temporaryFiles = [];
    for (let index = 0; index < batch.length; index += 1) {
        const original = batch[index];
        const temporary = path.join(
            sourceRoot,
            `${String(index).padStart(4, "0")}_${path.basename(original)}`,
        );
        fs.copyFileSync(original, temporary);
        temporaryFiles.push(temporary);
        temporaryToOriginal.set(normalizePath(temporary), original);
        originalToRelative.set(original, normalizePath(path.relative(inputRoot, original)));
    }

    const tsconfigPath = path.join(batchRoot, "tsconfig.json");
    fs.writeFileSync(
        tsconfigPath,
        JSON.stringify(
            {
                compilerOptions: {
                    ...baseTsconfig.compilerOptions,
                    typeRoots: [typeRoots],
                    noEmit: true,
                    moduleDetection: "force",
                },
                files: temporaryFiles,
            },
            null,
            2,
        ) + "\n",
        "utf8",
    );

    const invocation = spawnSync(
        nativeTscPath,
        ["--project", tsconfigPath, "--pretty", "false"],
        {
            cwd: batchRoot,
            encoding: "utf8",
            maxBuffer: 128 * 1024 * 1024,
        },
    );
    const parsed = parseDiagnostics(
        `${invocation.stdout ?? ""}\n${invocation.stderr ?? ""}`,
        temporaryToOriginal,
        batchRoot,
    );
    const infrastructureMessages = [...parsed.infrastructureMessages];
    if (invocation.error) {
        infrastructureMessages.push(
            `${invocation.error.name}: ${invocation.error.message}`,
        );
    }
    if (invocation.status !== 0 && parsed.byOriginal.size === 0) {
        infrastructureMessages.push(
            `TypeScript compiler exited ${String(invocation.status)} without a source diagnostic`,
        );
    }

    return batch.map((original) => {
        const diagnostics = parsed.byOriginal.get(original) ?? [];
        const globalDiagnostics = infrastructureMessages.map((message) => ({
            code: "typescript_batch_infrastructure",
            severity: "error",
            message,
            line: null,
            column: null,
            end_line: null,
            end_column: null,
        }));
        const allDiagnostics = [...diagnostics, ...globalDiagnostics];
        const errors = allDiagnostics.filter((item) => item.severity === "error");
        const parseSuccess = !errors.some((item) => {
            const numeric = Number(String(item.code).replace(/^TS/u, ""));
            return Number.isInteger(numeric) && numeric >= 1000 && numeric < 2000;
        });
        return {
            relative_path: `typescript/${originalToRelative.get(original)}`,
            attempted: true,
            success: errors.length === 0,
            parse_success: parseSuccess,
            frontend: frontendName,
            command: [nativeTscPath, "--project", "<batch-tsconfig>", "--pretty", "false"],
            exit_code: invocation.status,
            stdout: "",
            stderr: "",
            diagnostics: allDiagnostics,
            infrastructure_error: infrastructureMessages.length > 0
                ? infrastructureMessages.join("\n")
                : null,
            reason_code: errors.length === 0
                ? "typescript_typecheck_passed"
                : "typescript_typecheck_failed",
        };
    });
}


function parseDiagnostics(text, temporaryToOriginal, diagnosticRoot) {
    const byOriginal = new Map();
    const infrastructureMessages = [];
    const lines = text.split(/\r?\n/u);
    const pattern = /^(.*)\((\d+),(\d+)\):\s+(error|warning|message|suggestion)\s+TS(\d+):\s+(.*)$/u;
    let lastRecord = null;
    for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line) {
            continue;
        }
        if (/^Found \d+ errors?(?: in .*?)?\.?$/u.test(line.trim())) {
            continue;
        }
        const match = pattern.exec(line);
        if (!match) {
            if (/^\s+/u.test(line) && lastRecord !== null) {
                lastRecord.message += `\n${line.trim()}`;
                continue;
            }
            infrastructureMessages.push(line);
            lastRecord = null;
            continue;
        }
        const reportedPath = match[1];
        const temporary = normalizePath(
            path.isAbsolute(reportedPath)
                ? path.resolve(reportedPath)
                : path.resolve(diagnosticRoot, reportedPath),
        );
        const original = temporaryToOriginal.get(temporary);
        if (!original) {
            infrastructureMessages.push(line);
            lastRecord = null;
            continue;
        }
        const record = {
            code: `TS${match[5]}`,
            severity: match[4].toLowerCase(),
            message: match[6],
            line: Number(match[2]),
            column: Number(match[3]),
            end_line: null,
            end_column: null,
        };
        const records = byOriginal.get(original) ?? [];
        records.push(record);
        byOriginal.set(original, records);
        lastRecord = record;
    }
    return { byOriginal, infrastructureMessages };
}


function readFileList(listPath, root) {
    const records = fs.readFileSync(listPath, "utf8")
        .split(/\r?\n/u)
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => path.resolve(item));
    for (const sourcePath of records) {
        requireSourceWithinRoot(sourcePath, root);
    }
    return [...new Set(records)].sort();
}


function collectSources(root) {
    const records = [];
    const pending = [root];
    while (pending.length > 0) {
        const current = pending.pop();
        for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
            const candidate = path.join(current, entry.name);
            if (entry.isDirectory()) {
                pending.push(candidate);
            } else if (entry.isFile() && entry.name === "source.ts") {
                records.push(candidate);
            }
        }
    }
    return records.sort();
}


function requireSourceWithinRoot(sourcePath, root) {
    const relative = path.relative(root, sourcePath);
    if (
        relative === ""
        || relative === ".."
        || relative.startsWith(`..${path.sep}`)
        || path.isAbsolute(relative)
        || path.basename(sourcePath) !== "source.ts"
    ) {
        throw new Error(`TypeScript source is outside input root: ${sourcePath}`);
    }
    if (!fs.statSync(sourcePath, { throwIfNoEntry: false })?.isFile()) {
        throw new Error(`TypeScript source does not exist: ${sourcePath}`);
    }
}


function requireOutsideInput(input, candidate) {
    const relative = path.relative(input, candidate);
    if (relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== "..")) {
        throw new Error(`Validation output must be outside the dataset: ${candidate}`);
    }
}


function requireDirectory(candidate) {
    if (!fs.statSync(candidate, { throwIfNoEntry: false })?.isDirectory()) {
        throw new Error(`Directory does not exist: ${candidate}`);
    }
}


function parseArgs(args) {
    const records = new Map();
    for (let index = 0; index < args.length; index += 2) {
        const key = args[index];
        const value = args[index + 1];
        if (!key?.startsWith("--") || value === undefined) {
            throw new Error(`Expected --key value arguments, found ${String(key)}`);
        }
        records.set(key, value);
    }
    return records;
}


function required(optionsMap, key) {
    const value = optionsMap.get(key);
    if (!value) {
        throw new Error(`Missing required argument ${key}`);
    }
    return value;
}


function positiveInteger(value, name) {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error(`${name} must be a positive integer`);
    }
    return parsed;
}


function normalizePath(value) {
    return value.split(path.sep).join("/");
}
