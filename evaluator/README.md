# Evaluator

This evaluator checks target security and structural validity without executing generated programs.

- `crypto_steering/validation/detectors/m01.py`–`m10.py`: one detector per misuse category.
- `crypto_steering/validation/tasks/f01/`–`f09/`: one `validate` entry in each `__init__.py`. 
- `../scripts/config/evaluation.json`: evaluation settings. Tooling paths are relative to this `evaluator/` directory.

Use CPython 3.11.x, JDK 21, Node.js 24.18.0, npm 11.16.0, and .NET SDK 10.0.302.
The included lock files select TypeScript 7.0.2, Node types 24.13.3, and Roslyn 5.6.0; C# uses language version 14.

With the Python environment active, run these setup commands from the artifact root:

```bash
python -m pip install -r requirements.txt
npm ci --prefix evaluator/scripts/tooling/typescript
(cd evaluator/scripts/tooling/csharp && dotnet restore --locked-mode)
```

Keep `javac`, `java`, `node`, and `dotnet` available on `PATH`. Java tooling is compiled during evaluation. 
See [script usage](../scripts/README.md) for the evaluation command and output paths.
