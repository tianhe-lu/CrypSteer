using System;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

if (args.Length > 0 && args[0] == "format-tree")
{
    SourceTreeFormatter.Run(args.Skip(1).ToArray());
    return;
}

if (args.Length > 0 && args[0] == "validate-tree")
{
    SourceTreeValidator.Run(args.Skip(1).ToArray());
    return;
}

const string source = """
    using System;
    using System.Security.Cryptography;

    public static class CryptoApiSmoke
    {
        public static byte[] RandomBytes(int length)
        {
            return RandomNumberGenerator.GetBytes(length);
        }

        public static byte[] Hash(ReadOnlySpan<byte> input)
        {
            return SHA256.HashData(input);
        }
    }
    """;

var syntaxTree = CSharpSyntaxTree.ParseText(
    source,
    new CSharpParseOptions(LanguageVersion.CSharp14)
);

var trustedPlatformAssemblies =
    AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string
    ?? throw new InvalidOperationException(
        "TRUSTED_PLATFORM_ASSEMBLIES is unavailable."
    );

var references = trustedPlatformAssemblies
    .Split(Path.PathSeparator)
    .Select(path => MetadataReference.CreateFromFile(path));

var compilation = CSharpCompilation.Create(
    assemblyName: "JcaApiMisuse.CSharpTooling.Smoke",
    syntaxTrees: new[] { syntaxTree },
    references: references,
    options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
);

var errors = compilation
    .GetDiagnostics()
    .Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)
    .ToArray();

if (errors.Length > 0)
{
    throw new InvalidOperationException(
        "Roslyn compilation failed:"
        + Environment.NewLine
        + string.Join(Environment.NewLine, errors.Select(error => error.ToString()))
    );
}

var semanticModel = compilation.GetSemanticModel(syntaxTree);
var root = syntaxTree.GetRoot();
var hashInvocation = root
    .DescendantNodes()
    .OfType<InvocationExpressionSyntax>()
    .Single(invocation => invocation.Expression.ToString() == "SHA256.HashData");

var hashSymbol = semanticModel.GetSymbolInfo(hashInvocation).Symbol as IMethodSymbol;
var containingType = hashSymbol?.ContainingType.ToDisplayString();

if (containingType != "System.Security.Cryptography.SHA256")
{
    throw new InvalidOperationException(
        $"Expected SHA256.HashData to resolve to the platform API, found {containingType ?? "<unresolved>"}."
    );
}

Console.WriteLine("C# toolchain smoke passed.");
Console.WriteLine($"Runtime: {Environment.Version}");
Console.WriteLine($"Resolved API: {hashSymbol!.ToDisplayString()}");
