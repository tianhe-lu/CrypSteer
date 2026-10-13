import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Compiles dataset sources independently in one JVM.
 *
 * Input is a UTF-8 TSV file containing program_id and absolute source_path.
 * Output is JSONL. A source-level failure is always emitted as a record and
 * never aborts later files.
 */
public final class ValidateDatasetTree {
    private static final Pattern PACKAGE_PATTERN = Pattern.compile(
        "(?m)^\\s*package\\s+([A-Za-z_$][\\w$]*(?:\\.[A-Za-z_$][\\w$]*)*)\\s*;"
    );
    private static final Pattern PUBLIC_TYPE_PATTERN = Pattern.compile(
        "(?m)\\bpublic\\s+(?:abstract\\s+|final\\s+|sealed\\s+|non-sealed\\s+)?"
            + "(?:class|interface|enum|record)\\s+([A-Za-z_$][\\w$]*)"
    );
    private static final Pattern TYPE_PATTERN = Pattern.compile(
        "(?m)\\b(?:class|interface|enum|record)\\s+([A-Za-z_$][\\w$]*)"
    );
    private static final List<String> SYNTAX_DIAGNOSTIC_FRAGMENTS = List.of(
        "expected",
        "premature.eof",
        "reached.end.of.file.while.parsing",
        "illegal.start",
        "not.stmt",
        "unclosed",
        "invalid.meth.decl.ret.type.req",
        "orphaned",
        "catch.without.try",
        "finally.without.try",
        "else.without.if",
        "try.without.catch.finally.or.resource.decls"
    );

    private ValidateDatasetTree() {}

    public static void main(String[] args) throws Exception {
        Map<String, String> options = parseArgs(args);
        Path input = requiredPath(options, "--input");
        Path output = requiredPath(options, "--output");
        String release = options.getOrDefault("--release", "21");
        int progressEvery = Integer.parseInt(
            options.getOrDefault("--progress-every", "500")
        );

        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new IllegalStateException(
                "The Java compiler is unavailable; run this tool with a JDK, not a JRE."
            );
        }

        Files.createDirectories(output.toAbsolutePath().getParent());
        try (
            StandardJavaFileManager standardFileManager = compiler.getStandardFileManager(
                null,
                Locale.ROOT,
                StandardCharsets.UTF_8
            );
            BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8);
            BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)
        ) {
            String line;
            int completed = 0;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                String[] fields = line.split("\\t", 2);
                if (fields.length != 2) {
                    throw new IllegalArgumentException("Malformed compiler input row: " + line);
                }
                String programId = fields[0];
                Path sourcePath = Path.of(fields[1]);
                writer.write(
                    compileOne(
                        compiler,
                        standardFileManager,
                        programId,
                        sourcePath,
                        release
                    )
                );
                writer.newLine();
                completed += 1;
                if (progressEvery > 0 && completed % progressEvery == 0) {
                    System.err.println("  Java frontend progress: " + completed);
                }
            }
            System.err.println("  Java frontend completed: " + completed);
        }
    }

    private static String compileOne(
        JavaCompiler compiler,
        StandardJavaFileManager standardFileManager,
        String programId,
        Path sourcePath,
        String release
    ) {
        try {
            String source = Files.readString(sourcePath, StandardCharsets.UTF_8);
            String binaryName = binaryName(source);
            DiagnosticCollector<JavaFileObject> collector = new DiagnosticCollector<>();
            JavaFileObject sourceObject = new StringSourceObject(binaryName, source);
            JavaFileManager fileManager = new MemoryFileManager(standardFileManager);
            List<String> compilerOptions = List.of(
                "--release",
                release,
                "-proc:none",
                "-Xlint:none",
                "-encoding",
                "UTF-8"
            );
            boolean success = Boolean.TRUE.equals(
                compiler.getTask(
                    null,
                    fileManager,
                    collector,
                    compilerOptions,
                    null,
                    List.of(sourceObject)
                ).call()
            );
            List<Diagnostic<? extends JavaFileObject>> diagnostics = collector.getDiagnostics();
            boolean parseSuccess = diagnostics.stream()
                .filter(item -> item.getKind() == Diagnostic.Kind.ERROR)
                .noneMatch(ValidateDatasetTree::isSyntaxDiagnostic);
            return resultJson(
                programId,
                sourcePath,
                success,
                parseSuccess,
                diagnostics,
                null
            );
        } catch (Throwable error) {
            return resultJson(
                programId,
                sourcePath,
                false,
                false,
                List.of(),
                error.getClass().getName() + ": " + String.valueOf(error.getMessage())
            );
        }
    }

    private static boolean isSyntaxDiagnostic(
        Diagnostic<? extends JavaFileObject> diagnostic
    ) {
        String code = diagnostic.getCode().toLowerCase(Locale.ROOT);
        return SYNTAX_DIAGNOSTIC_FRAGMENTS.stream().anyMatch(code::contains);
    }

    private static String binaryName(String source) {
        Matcher packageMatcher = PACKAGE_PATTERN.matcher(source);
        String packageName = packageMatcher.find() ? packageMatcher.group(1) : "";
        Matcher publicMatcher = PUBLIC_TYPE_PATTERN.matcher(source);
        String typeName;
        if (publicMatcher.find()) {
            typeName = publicMatcher.group(1);
        } else {
            Matcher typeMatcher = TYPE_PATTERN.matcher(source);
            typeName = typeMatcher.find() ? typeMatcher.group(1) : "Main";
        }
        return packageName.isEmpty() ? typeName : packageName + "." + typeName;
    }

    private static String resultJson(
        String programId,
        Path sourcePath,
        boolean success,
        boolean parseSuccess,
        List<Diagnostic<? extends JavaFileObject>> diagnostics,
        String infrastructureError
    ) {
        StringBuilder result = new StringBuilder();
        result.append('{');
        field(result, "program_id", programId).append(',');
        field(result, "source_path", sourcePath.toString()).append(',');
        result.append("\"attempted\":true,");
        result.append("\"success\":").append(success).append(',');
        result.append("\"parse_success\":").append(parseSuccess).append(',');
        result.append("\"frontend\":\"javac_java_compiler_api\",");
        result.append("\"exit_code\":").append(success ? 0 : 1).append(',');
        result.append("\"diagnostics\":[");
        for (int index = 0; index < diagnostics.size(); index += 1) {
            if (index > 0) {
                result.append(',');
            }
            Diagnostic<? extends JavaFileObject> diagnostic = diagnostics.get(index);
            result.append('{');
            field(result, "code", diagnostic.getCode()).append(',');
            field(result, "severity", diagnostic.getKind().name().toLowerCase(Locale.ROOT)).append(',');
            field(result, "message", diagnostic.getMessage(Locale.ROOT)).append(',');
            numberField(result, "line", diagnostic.getLineNumber()).append(',');
            numberField(result, "column", diagnostic.getColumnNumber());
            result.append('}');
        }
        result.append("],");
        if (infrastructureError == null) {
            result.append("\"infrastructure_error\":null");
        } else {
            field(result, "infrastructure_error", infrastructureError);
        }
        result.append('}');
        return result.toString();
    }

    private static StringBuilder field(StringBuilder target, String name, String value) {
        return target
            .append('"')
            .append(jsonEscape(name))
            .append("\":\"")
            .append(jsonEscape(value))
            .append('"');
    }

    private static StringBuilder numberField(StringBuilder target, String name, long value) {
        target.append('"').append(jsonEscape(name)).append("\":");
        if (value < 0) {
            return target.append("null");
        }
        return target.append(value);
    }

    private static String jsonEscape(String value) {
        StringBuilder escaped = new StringBuilder();
        for (int index = 0; index < value.length(); index += 1) {
            char character = value.charAt(index);
            switch (character) {
                case '"' -> escaped.append("\\\"");
                case '\\' -> escaped.append("\\\\");
                case '\b' -> escaped.append("\\b");
                case '\f' -> escaped.append("\\f");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (character < 0x20) {
                        escaped.append(String.format(Locale.ROOT, "\\u%04x", (int) character));
                    } else {
                        escaped.append(character);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> options = new HashMap<>();
        for (int index = 0; index < args.length; index += 2) {
            if (index + 1 >= args.length || !args[index].startsWith("--")) {
                throw new IllegalArgumentException("Expected --name value arguments.");
            }
            options.put(args[index], args[index + 1]);
        }
        return options;
    }

    private static Path requiredPath(Map<String, String> options, String name) {
        String value = options.get(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing required option " + name);
        }
        return Path.of(value);
    }

    private static final class StringSourceObject extends SimpleJavaFileObject {
        private final String source;

        private StringSourceObject(String binaryName, String source) {
            super(
                URI.create("string:///" + binaryName.replace('.', '/') + Kind.SOURCE.extension),
                Kind.SOURCE
            );
            this.source = source;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return source;
        }
    }

    private static final class ByteCodeObject extends SimpleJavaFileObject {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        private ByteCodeObject(String binaryName, Kind kind) {
            super(URI.create("bytes:///" + binaryName.replace('.', '/') + kind.extension), kind);
        }

        @Override
        public ByteArrayOutputStream openOutputStream() {
            return bytes;
        }
    }

    private static final class MemoryFileManager
        extends ForwardingJavaFileManager<StandardJavaFileManager> {
        private final List<ByteCodeObject> outputs = new ArrayList<>();

        private MemoryFileManager(StandardJavaFileManager delegate) {
            super(delegate);
        }

        @Override
        public JavaFileObject getJavaFileForOutput(
            Location location,
            String className,
            JavaFileObject.Kind kind,
            FileObject sibling
        ) throws IOException {
            ByteCodeObject output = new ByteCodeObject(className, kind);
            outputs.add(output);
            return output;
        }
    }
}
