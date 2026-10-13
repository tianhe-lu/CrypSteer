using System;
using System.Collections.Immutable;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Emit;

internal static class SourceTreeValidator
{
    public static void Run(string[] args)
    {
        Dictionary<string, string> options = ParseArgs(args);
        string inputPath = Required(options, "--input");
        string outputPath = Required(options, "--output");
        int progressEvery = int.Parse(
            options.GetValueOrDefault("--progress-every", "500")
        );
        LanguageVersion languageVersion = ParseLanguageVersion(
            options.GetValueOrDefault("--language-version", "14.0")
        );

        ImmutableArray<MetadataReference> references = PlatformReferences();
        string? outputDirectory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        using StreamReader reader = new(inputPath, Encoding.UTF8);
        using StreamWriter writer = new(outputPath, false, new UTF8Encoding(false));
        int completed = 0;
        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }
            int separator = line.IndexOf('\t');
            if (separator <= 0 || separator == line.Length - 1)
            {
                throw new InvalidOperationException(
                    $"Malformed compiler input row: {line}"
                );
            }
            string programId = line[..separator];
            string sourcePath = Path.GetFullPath(line[(separator + 1)..]);
            FrontendRecord result = ValidateOne(
                programId,
                sourcePath,
                languageVersion,
                references
            );
            writer.WriteLine(JsonSerializer.Serialize(result));
            completed += 1;
            if (
                progressEvery > 0
                && (completed % progressEvery == 0 || reader.EndOfStream)
            )
            {
                Console.Error.WriteLine($"  C# frontend progress: {completed}");
            }
        }
    }

    private static FrontendRecord ValidateOne(
        string programId,
        string sourcePath,
        LanguageVersion languageVersion,
        ImmutableArray<MetadataReference> references
    )
    {
        try
        {
            string source = File.ReadAllText(sourcePath, Encoding.UTF8);
            CSharpSyntaxTree syntaxTree = (CSharpSyntaxTree)CSharpSyntaxTree.ParseText(
                source,
                new CSharpParseOptions(languageVersion),
                sourcePath,
                Encoding.UTF8
            );
            ImmutableArray<Diagnostic> syntaxDiagnostics = syntaxTree
                .GetDiagnostics()
                .Where(item => item.Severity == DiagnosticSeverity.Error)
                .ToImmutableArray();
            bool hasTopLevelStatements = syntaxTree
                .GetCompilationUnitRoot()
                .Members
                .Any(member => member is GlobalStatementSyntax);
            CSharpCompilation compilation = CSharpCompilation.Create(
                $"DatasetArtifact_{completedAssemblySuffix(programId)}",
                new[] { syntaxTree },
                references,
                new CSharpCompilationOptions(
                    hasTopLevelStatements
                        ? OutputKind.ConsoleApplication
                        : OutputKind.DynamicallyLinkedLibrary,
                    optimizationLevel: OptimizationLevel.Release,
                    nullableContextOptions: NullableContextOptions.Enable,
                    allowUnsafe: false,
                    deterministic: true
                )
            );
            using Stream emittedAssembly = Stream.Null;
            EmitResult emitResult = compilation.Emit(emittedAssembly);
            ImmutableArray<Diagnostic> diagnostics = emitResult.Diagnostics
                .Where(item => item.Severity != DiagnosticSeverity.Hidden)
                .Distinct(DiagnosticIdentityComparer.Instance)
                .ToImmutableArray();
            bool success = !diagnostics.Any(
                item => item.Severity == DiagnosticSeverity.Error
            );
            return new FrontendRecord(
                programId,
                sourcePath,
                true,
                success,
                syntaxDiagnostics.Length == 0,
                "roslyn_csharp_compilation",
                success ? 0 : 1,
                diagnostics.Select(ToRecord).ToArray(),
                null
            );
        }
        catch (Exception error)
        {
            return new FrontendRecord(
                programId,
                sourcePath,
                true,
                false,
                false,
                "roslyn_csharp_compilation",
                1,
                Array.Empty<DiagnosticRecord>(),
                $"{error.GetType().FullName}: {error.Message}"
            );
        }
    }

    private static DiagnosticRecord ToRecord(Diagnostic diagnostic)
    {
        FileLinePositionSpan location = diagnostic.Location.GetLineSpan();
        int? line = null;
        int? column = null;
        int? endLine = null;
        int? endColumn = null;
        if (diagnostic.Location.IsInSource)
        {
            line = location.StartLinePosition.Line + 1;
            column = location.StartLinePosition.Character + 1;
            endLine = location.EndLinePosition.Line + 1;
            endColumn = location.EndLinePosition.Character + 1;
        }
        return new DiagnosticRecord(
            diagnostic.Id,
            diagnostic.Severity.ToString().ToLowerInvariant(),
            diagnostic.GetMessage(),
            line,
            column,
            endLine,
            endColumn
        );
    }

    private static ImmutableArray<MetadataReference> PlatformReferences()
    {
        string trustedPlatformAssemblies =
            AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string
            ?? throw new InvalidOperationException(
                "TRUSTED_PLATFORM_ASSEMBLIES is unavailable."
            );
        return trustedPlatformAssemblies
            .Split(Path.PathSeparator)
            .Select(path => (MetadataReference)MetadataReference.CreateFromFile(path))
            .ToImmutableArray();
    }

    private static LanguageVersion ParseLanguageVersion(string value)
    {
        return value switch
        {
            "14" or "14.0" or "CSharp14" => LanguageVersion.CSharp14,
            "latest" or "Latest" => LanguageVersion.Latest,
            _ => throw new InvalidOperationException(
                $"Unsupported configured C# language version: {value}"
            ),
        };
    }

    private static string completedAssemblySuffix(string programId)
    {
        byte[] digest = System.Security.Cryptography.SHA256.HashData(
            Encoding.UTF8.GetBytes(programId)
        );
        return Convert.ToHexString(digest.AsSpan(0, 8));
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        if (args.Length % 2 != 0)
        {
            throw new InvalidOperationException("Expected --name value arguments.");
        }
        Dictionary<string, string> result = new(StringComparer.Ordinal);
        for (int index = 0; index < args.Length; index += 2)
        {
            if (!args[index].StartsWith("--", StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Expected an option name, found {args[index]}"
                );
            }
            result[args[index]] = args[index + 1];
        }
        return result;
    }

    private static string Required(Dictionary<string, string> options, string name)
    {
        if (!options.TryGetValue(name, out string? value) || string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException($"Missing required option {name}");
        }
        return value;
    }

    private sealed class DiagnosticIdentityComparer : IEqualityComparer<Diagnostic>
    {
        public static readonly DiagnosticIdentityComparer Instance = new();

        public bool Equals(Diagnostic? left, Diagnostic? right)
        {
            if (ReferenceEquals(left, right))
            {
                return true;
            }
            if (left is null || right is null)
            {
                return false;
            }
            return left.Id == right.Id
                && left.Severity == right.Severity
                && left.Location.SourceSpan == right.Location.SourceSpan
                && left.GetMessage() == right.GetMessage();
        }

        public int GetHashCode(Diagnostic diagnostic)
        {
            return HashCode.Combine(
                diagnostic.Id,
                diagnostic.Severity,
                diagnostic.Location.SourceSpan,
                diagnostic.GetMessage()
            );
        }
    }

    private sealed record DiagnosticRecord(
        string code,
        string severity,
        string message,
        int? line,
        int? column,
        int? end_line,
        int? end_column
    );

    private sealed record FrontendRecord(
        string program_id,
        string source_path,
        bool attempted,
        bool success,
        bool parse_success,
        string frontend,
        int exit_code,
        DiagnosticRecord[] diagnostics,
        string? infrastructure_error
    );
}
