using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

internal static class SourceTreeFormatter
{
    private sealed record ReservedIdentifierRepair(
        string Name,
        Regex Declaration,
        SyntaxKind KeywordKind,
        string OldIdentifier,
        string ReplacementIdentifier,
        int ExpectedTokenCount
    );

    private sealed record ExactTextRepair(
        string Name,
        string OldText,
        string NewText,
        int ExpectedOccurrences
    );

    private static readonly ReservedIdentifierRepair[] IdentifierRepairs =
    {
        new(
            Name: "memory_stream_out",
            Declaration: new Regex(
                @"(?m)^[ \t]*MemoryStream[ \t]+(?<identifier>out)[ \t]*="
                    + @"[ \t]*new[ \t]+MemoryStream"
                    + @"[ \t]*\([^;\r\n]*\)[ \t]*;[ \t]*\r?$",
                RegexOptions.CultureInvariant
            ),
            KeywordKind: SyntaxKind.OutKeyword,
            OldIdentifier: "out",
            ReplacementIdentifier: "outputStream",
            ExpectedTokenCount: 4
        ),
        new(
            Name: "char_array_out",
            Declaration: new Regex(
                @"(?m)^[ \t]*char\[\][ \t]+(?<identifier>out)[ \t]*="
                    + @"[^;\r\n]+;[ \t]*\r?$",
                RegexOptions.CultureInvariant
            ),
            KeywordKind: SyntaxKind.OutKeyword,
            OldIdentifier: "out",
            ReplacementIdentifier: "outputChars",
            ExpectedTokenCount: 4
        ),
        new(
            Name: "byte_array_params",
            Declaration: new Regex(
                @"(?m)^[ \t]*byte\[\][ \t]+(?<identifier>params)[ \t]*="
                    + @"[^;\r\n]+;[ \t]*\r?$",
                RegexOptions.CultureInvariant
            ),
            KeywordKind: SyntaxKind.ParamsKeyword,
            OldIdentifier: "params",
            ReplacementIdentifier: "parameters",
            ExpectedTokenCount: 2
        ),
    };

    private static readonly ExactTextRepair[] TextRepairs =
    {
        new(
            Name: "console_in_parenthesized_member_access",
            OldText: "Console.(In.ReadLine() ?? string.Empty)",
            NewText: "(Console.In.ReadLine() ?? string.Empty)",
            ExpectedOccurrences: 1
        ),
    };

    public static void Run(string[] args)
    {
        var options = ParseArgs(args);
        var inputRoot = Path.GetFullPath(options["--input-root"]);
        var outputRoot = Path.GetFullPath(options["--output-root"]);

        if (!Directory.Exists(inputRoot))
        {
            throw new DirectoryNotFoundException(
                $"C# formatter input root does not exist: {inputRoot}"
            );
        }
        var relativeOutput = Path.GetRelativePath(inputRoot, outputRoot);
        var parentPrefix = $"..{Path.DirectorySeparatorChar}";
        if (
            relativeOutput != ".."
            && !relativeOutput.StartsWith(
                parentPrefix,
                StringComparison.Ordinal
            )
        )
        {
            throw new InvalidOperationException(
                "C# formatter output root must be outside its input root."
            );
        }

        var sourcePaths = Directory
            .EnumerateFiles(inputRoot, "source.cs", SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();

        var repairedArtifacts = 0;
        var repairUnitsByKind = IdentifierRepairs
            .Select(repair => repair.Name)
            .Concat(TextRepairs.Select(repair => repair.Name))
            .ToDictionary(
                name => name,
                _ => 0,
                StringComparer.Ordinal
            );
        var failures = new List<string>();
        for (var index = 0; index < sourcePaths.Length; index++)
        {
            var sourcePath = sourcePaths[index];
            var relativePath = Path.GetRelativePath(inputRoot, sourcePath);
            var outputPath = Path.Combine(outputRoot, relativePath);
            try
            {
                var source = File.ReadAllText(sourcePath);
                var formatted = FormatSource(
                    source,
                    sourcePath,
                    out var sourceRepairedTokens
                );
                if (sourceRepairedTokens.Count > 0)
                {
                    repairedArtifacts++;
                    foreach (var (name, count) in sourceRepairedTokens)
                    {
                        repairUnitsByKind[name] += count;
                    }
                }

                Directory.CreateDirectory(
                    Path.GetDirectoryName(outputPath)
                    ?? throw new InvalidOperationException(
                        $"Output path has no directory: {outputPath}"
                    )
                );
                File.WriteAllText(outputPath, formatted);
            }
            catch (Exception error)
            {
                failures.Add(
                    $"{relativePath}:{Environment.NewLine}{error.Message}"
                );
            }

            var completed = index + 1;
            if (completed % 500 == 0 || completed == sourcePaths.Length)
            {
                Console.Error.WriteLine(
                    $"  C# formatting progress: {completed}/{sourcePaths.Length}"
                );
            }
        }

        if (failures.Count > 0)
        {
            throw new InvalidOperationException(
                $"C# formatter rejected {failures.Count} source artifact(s):"
                    + Environment.NewLine
                    + string.Join(
                        Environment.NewLine + Environment.NewLine,
                        failures.Take(50)
                    )
                    + (
                        failures.Count > 50
                            ? Environment.NewLine
                                + $"... {failures.Count - 50} more omitted"
                            : string.Empty
                    )
            );
        }

        Console.WriteLine(
            $"Formatted {sourcePaths.Length} C# syntax trees into {outputRoot}"
        );
        Console.WriteLine(
            "Applied generated-source pre-parse repairs in "
                + $"{repairedArtifacts} artifacts: "
                + string.Join(
                    ", ",
                    repairUnitsByKind.Select(item =>
                        $"{item.Key}={item.Value}"
                    )
                )
        );
    }

    private static string FormatSource(
        string source,
        string sourcePath,
        out Dictionary<string, int> repairedTokens
    )
    {
        var parseOptions = new CSharpParseOptions(LanguageVersion.CSharp14);
        var sourceForFormatting = source;
        repairedTokens = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var repair in TextRepairs)
        {
            sourceForFormatting = RepairKnownExactText(
                sourceForFormatting,
                sourcePath,
                repair,
                out var repairCount
            );
            if (repairCount > 0)
            {
                repairedTokens[repair.Name] = repairCount;
            }
        }
        foreach (var repair in IdentifierRepairs)
        {
            sourceForFormatting = RepairKnownReservedIdentifier(
                sourceForFormatting,
                sourcePath,
                parseOptions,
                repair,
                out var repairTokenCount
            );
            if (repairTokenCount > 0)
            {
                repairedTokens[repair.Name] = repairTokenCount;
            }
        }
        var syntaxTree = CSharpSyntaxTree.ParseText(
            sourceForFormatting,
            parseOptions
        );
        RequireNoSyntaxErrors(syntaxTree, sourcePath, "before formatting");

        var root = syntaxTree.GetRoot();
        var tokensBefore = TokenFingerprints(root);
        var triviaBefore = SignificantTriviaTexts(root);
        var formatted = root
            .NormalizeWhitespace(
                indentation: "    ",
                eol: "\n",
                elasticTrivia: false
            )
            .ToFullString()
            .TrimEnd()
            + "\n";

        var formattedTree = CSharpSyntaxTree.ParseText(formatted, parseOptions);
        RequireNoSyntaxErrors(
            formattedTree,
            sourcePath,
            "after formatting"
        );
        var formattedRoot = formattedTree.GetRoot();

        if (!tokensBefore.SequenceEqual(TokenFingerprints(formattedRoot)))
        {
            throw new InvalidOperationException(
                $"Roslyn formatting changed C# tokens: {sourcePath}"
            );
        }
        if (!triviaBefore.SequenceEqual(SignificantTriviaTexts(formattedRoot)))
        {
            throw new InvalidOperationException(
                "Roslyn formatting changed C# comments, directives, or "
                    + $"other significant trivia: {sourcePath}"
            );
        }

        var secondPass = formattedRoot
            .NormalizeWhitespace(
                indentation: "    ",
                eol: "\n",
                elasticTrivia: false
            )
            .ToFullString()
            .TrimEnd()
            + "\n";
        if (!string.Equals(formatted, secondPass, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Roslyn formatting is not idempotent: {sourcePath}"
            );
        }
        return formatted;
    }

    private static string RepairKnownExactText(
        string source,
        string sourcePath,
        ExactTextRepair repair,
        out int repairedOccurrences
    )
    {
        repairedOccurrences = CountOccurrences(source, repair.OldText);
        if (repairedOccurrences == 0)
        {
            return source;
        }
        if (repairedOccurrences != repair.ExpectedOccurrences)
        {
            throw new InvalidOperationException(
                $"Expected {repair.ExpectedOccurrences} {repair.Name} "
                    + $"occurrence(s) in {sourcePath}, found "
                    + repairedOccurrences
            );
        }
        return source.Replace(
            repair.OldText,
            repair.NewText,
            StringComparison.Ordinal
        );
    }

    private static int CountOccurrences(string source, string value)
    {
        var count = 0;
        var start = 0;
        while (
            (start = source.IndexOf(
                value,
                start,
                StringComparison.Ordinal
            )) >= 0
        )
        {
            count++;
            start += value.Length;
        }
        return count;
    }

    private static string RepairKnownReservedIdentifier(
        string source,
        string sourcePath,
        CSharpParseOptions parseOptions,
        ReservedIdentifierRepair repair,
        out int repairedTokens
    )
    {
        repairedTokens = 0;
        var declarations = repair.Declaration.Matches(source);
        if (declarations.Count == 0)
        {
            return source;
        }
        if (declarations.Count != 1)
        {
            throw new InvalidOperationException(
                $"Expected one {repair.Name} declaration in {sourcePath}, "
                    + $"found {declarations.Count}"
            );
        }

        var lexicalTokens = SyntaxFactory
            .ParseTokens(
                source,
                options: parseOptions
            )
            .ToArray();
        if (
            lexicalTokens.Any(token =>
                token.IsKind(SyntaxKind.IdentifierToken)
                && token.ValueText == repair.ReplacementIdentifier
            )
        )
        {
            throw new InvalidOperationException(
                $"Cannot apply {repair.Name} because "
                    + $"{repair.ReplacementIdentifier} already exists in "
                    + sourcePath
            );
        }

        var identifierTokens = lexicalTokens
            .Where(token => token.IsKind(repair.KeywordKind))
            .OrderBy(token => token.SpanStart)
            .ToArray();
        var textualTokenCount = Regex.Matches(
            source,
            $@"\b{Regex.Escape(repair.OldIdentifier)}\b",
            RegexOptions.CultureInvariant
        ).Count;
        var declarationStart = declarations[0]
            .Groups["identifier"]
            .Index;
        if (
            identifierTokens.Length != repair.ExpectedTokenCount
            || textualTokenCount != repair.ExpectedTokenCount
            || identifierTokens.Count(token =>
                token.SpanStart == declarationStart
            )
                != 1
        )
        {
            throw new InvalidOperationException(
                $"Unsupported {repair.Name} token layout in {sourcePath}: "
                    + $"lexical={identifierTokens.Length}, "
                    + $"textual={textualTokenCount}, "
                    + $"expected={repair.ExpectedTokenCount}"
            );
        }

        var repaired = new StringBuilder(source);
        foreach (var token in identifierTokens.Reverse())
        {
            repaired.Remove(token.SpanStart, token.Span.Length);
            repaired.Insert(
                token.SpanStart,
                repair.ReplacementIdentifier
            );
        }
        repairedTokens = identifierTokens.Length;
        return repaired.ToString();
    }

    private static string[] TokenFingerprints(SyntaxNode root)
    {
        return root
            .DescendantTokens(descendIntoTrivia: true)
            .Select(token =>
                token.RawKind
                    + "\0"
                    + token.Text
                    + "\0"
                    + token.ValueText
                    + "\0"
                    + token.IsMissing
            )
            .ToArray();
    }

    private static string[] SignificantTriviaTexts(SyntaxNode root)
    {
        return root
            .DescendantTrivia(descendIntoTrivia: true)
            .Where(trivia => !IsFormattingTrivia(trivia))
            .Select(trivia =>
                trivia.RawKind
                    + "\0"
                    + trivia.ToFullString().Trim()
            )
            .ToArray();
    }

    private static bool IsFormattingTrivia(SyntaxTrivia trivia)
    {
        return trivia.IsKind(SyntaxKind.WhitespaceTrivia)
            || trivia.IsKind(SyntaxKind.EndOfLineTrivia);
    }

    private static void RequireNoSyntaxErrors(
        SyntaxTree tree,
        string sourcePath,
        string phase
    )
    {
        var errors = tree
            .GetDiagnostics()
            .Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)
            .Take(20)
            .ToArray();
        if (errors.Length == 0)
        {
            return;
        }
        throw new InvalidOperationException(
            $"C# syntax errors {phase} in {sourcePath}:"
            + Environment.NewLine
            + string.Join(
                Environment.NewLine,
                errors.Select(error => error.ToString())
            )
        );
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        var options = new Dictionary<string, string>(
            StringComparer.Ordinal
        );
        for (var index = 0; index < args.Length; index += 2)
        {
            if (index + 1 >= args.Length)
            {
                throw new ArgumentException(
                    $"Missing value for C# formatter argument: {args[index]}"
                );
            }
            options[args[index]] = args[index + 1];
        }
        foreach (var required in new[] { "--input-root", "--output-root" })
        {
            if (!options.ContainsKey(required))
            {
                throw new ArgumentException(
                    $"Missing required C# formatter argument: {required}"
                );
            }
        }
        return options;
    }
}
