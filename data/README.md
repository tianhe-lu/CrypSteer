# Data

The dataset files are released under the [MIT License](../LICENSE). See [Citation](../README.md#citation) for the accompanying paper.

Each JSONL line contains one program, generation prompt, or instruction pair.

| File | Records | Contents |
| --- | ---: | --- |
| `programs.jsonl` | 96,000 | 12,000 original families, four structures, two labels. |
| `combinations.jsonl` | 4,096 | 1,024 security-choice combinations per language. |
| `heatmap_programs.jsonl` | 80 | 40 secure/misuse pairs for code heatmaps. |
| `generation_prompts.jsonl` | 200 | Five fixed generation prompts per language and misuse category. |
| `surgical_prompt_pairs.jsonl` | 144 | Paired secure/misuse instructions for Surgical Repair. |

Languages: `java`, `typescript`, `csharp`, `python`.

## Fields

| Field | Meaning |
| --- | --- |
| `id` | Unique program ID. |
| `original_id` | Family ID shared by four secure/misuse pairs. |
| `pair_id` | Identifies a secure/misuse program or instruction pair. |
| `language` | Programming language. |
| `misuse_id` | Target misuse category, M01–M10. |
| `task_id` | Task ID, such as `F6_T3`. |
| `structure` | `variable_inline`, `variable_helper`, `callsite_inline`, or `callsite_helper`. |
| `label` | `secure` or `misuse` under the target-category policy. |
| `secure_bits` | Ten-character bit string, ordered M01–M10 from left to right. |
| `secure_count` | Number of secure choices, from 0 to 10. |
| `code` | Full source text, including imports. |
| `prompt_id` | Unique generation prompt ID. |
| `task_prompt` | Requested programming task. |
| `generation_prompt` | Complete user message, including language-specific instructions. |
| `secure_input`, `misuse_input` | Complete contrasting instructions, including comment delimiters. |

Each file contains the fields relevant to its records.
`variable` places the value in a variable; `callsite` places it at the API call.
`inline` uses a direct expression; `helper` uses a helper return value.

## Labels and pairing

Binary encoding: 1 = secure, 0 = misuse.
In `combinations.jsonl`, `1111111111` is all-secure and `0000000000` is all-misuse.
`1000000000` makes only M01 secure.
The paired-program files use the strings `secure` and `misuse`.

Match counterparts using `pair_id`.
Keep all eight programs with the same `original_id` together when splitting data.
SCS-Code uses the 48,000 pairs in `programs.jsonl`; step 04b constructs its A/B inputs.
