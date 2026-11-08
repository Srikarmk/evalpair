# evalpair

<!-- mcp-name: io.github.Srikarmk/evalpair -->

Run an LLM eval suite against a model, cache every call, and compare two runs item by item: did the new
model, prompt or quantization actually change anything, or is the difference noise?

Most eval tools print two scores side by side. That reads a 2-point gap on 160 questions as a result, when
it usually isn't one. evalpair pairs the two runs by item, puts a bootstrap interval on the difference, counts
the items that got worse and better, and can measure a **noise floor**: the same model asked the same
questions twice. In my own studies the same prompt sent twice at temperature 0 changed up to 3% of answers,
so an answer-change rate only means something above that.

It's a CLI and an MCP server, so an agent can run an eval and read the verdict itself. It only calls the
model server you point it at, and keeps everything in one local folder.

## Quick start

```bash
uvx evalpair suites
uvx evalpair run gsm8k-160 -m llama3.1:8b --repeat --label llama
uvx evalpair run gsm8k-160 -m qwen3:8b --think off --label qwen3
uvx evalpair compare llama qwen3
```

`run` prints the score with a 95% interval and the first failures, unparsed and truncated replies first
(those are usually the parser's or the token budget's fault, not the model's). `compare` prints both scores,
the paired difference with its interval, how many items got worse and better, the noise floor, and ends with
a plain-language verdict: whether the score difference is distinguishable from zero, and whether the answers
moved more than the same model asked twice.

A real comparison: Ollama's default 4-bit Llama 3.1 8B against its 2-bit build, on an M5 Pro laptop
(`--num-ctx 4096`; about 40 minutes for both runs):

```
$ evalpair compare llama-q4 llama-q2
A            llama-q4: 84.4% [79.4%, 90.0%]
B            llama-q2: 60.6% [53.1%, 68.1%]
B − A        -23.8 pp [-31.9 pp, -16.2 pp]
changed      40.6% [33.1%, 48.1%] of 160 items (43 worse in B, 5 better)
floor (A)    0.0% [0.0%, 0.0%]; excess +40.6 pp [+33.1 pp, +48.1 pp]

B scores lower than A: -23.8 pp [-31.9 pp, -16.2 pp], paired over 160 items. Answers changed on 40.6% of
items (43 got worse, 5 better). The noise floor (the same model asked twice) is 0.0% for A, so the change
is beyond noise.
  worse in B:  gsm8k-0259  '168' → '52'
  ...
```

Both scores are the ones the quantization study measured for these builds on the same items.

Interrupt a run and run it again: every reply is cached, so it picks up where it stopped.

## As an MCP server

```bash
claude mcp add evalpair -- uvx evalpair mcp
```

or in any MCP client's config: `{"command": "uvx", "args": ["evalpair", "mcp"]}`.

| Tool | What it does |
|---|---|
| `list_suites` | Suites you can run: name, item count, checker, what it measures |
| `run_eval` | Run a suite against one model and save the run (calls the model; cached replies are reused) |
| `list_runs` | Saved runs, newest first |
| `show_run` | A run's score with its interval, its noise floor, and the end of failed replies |
| `compare_runs` | Two runs paired by item: difference with an interval, flips each way, change vs. noise floor, examples |

Everything except `run_eval` only reads saved runs.

## Backends

- **Ollama** (default): `--backend ollama`, server at `http://localhost:11434`. The run records the model's
  digest, so you can tell later exactly which build was tested. `--think on|off` for models that think,
  `--num-ctx` for the context window.
- **Any OpenAI-compatible server**: `--backend openai --base-url http://localhost:8000/v1` for vLLM,
  llama.cpp's server or LM Studio, or a hosted API. A bearer token is read from `EVALPAIR_API_KEY` (or
  `OPENAI_API_KEY`) and never written to disk.

Temperature defaults to 0 and the seed to 7. Both are part of the cache key, along with the server URL,
model, messages, token budget and thinking setting.

## Suites

Three are bundled. They're the item sets from my Measuring LLM Systems studies, with those studies' prompts
word for word, so a run can be checked against the published numbers:

| Suite | Items | Checker | Source |
|---|---|---|---|
| `gsm8k-160` | 160 | `number` | GSM8K test (MIT) |
| `mmlu-hs-160` | 160 | `choice`, stratified by subject | MMLU, four high-school subjects (MIT) |
| `e2e-extract-150` | 150 | `json_fields` | E2E NLG restaurant records (CC BY-SA 4.0) |

To make your own, create a folder with two files, in `~/.evalpair/suites/<name>/` or anywhere you then pass
by path:

```toml
# suite.toml
description = "Capitals, answered in one word"
checker = "exact"
max_tokens = 64
system = "Answer in one word."           # optional
user = "What is the capital of {country}?"
group = "region"                         # optional: stratify the bootstrap by this item field
```

```jsonl
{"id": "fr", "country": "France", "region": "Europe", "gold": "Paris"}
{"id": "pe", "country": "Peru", "region": "Americas", "gold": ["Lima", "Ciudad de los Reyes"]}
```

`{field}` in a prompt is filled from the item; a list of strings becomes lettered options (`A. …`), anything
else is shown as indented JSON. Use `{{` and `}}` for literal braces.

| Checker | Scores | Gold |
|---|---|---|
| `number` | the number after "Answer:", else `\boxed{}`, else the last number | a number |
| `choice` | the letter after "Answer:", with fallbacks (letters set by `checker_options.letters`) | a letter |
| `exact` | the text after "Answer:", or the last line, normalised | a string or list of strings |
| `contains` | whether a gold string appears as whole words | a string or list |
| `regex` | the last match of `checker_options.pattern` (first group) | a string or list |
| `json_fields` | the share of the gold object's fields the reply's JSON gets right (null must be null) | an object |
| `chrf` | character n-gram F-score (sacreBLEU's chrF) against a reference text, for free text such as commit messages or summaries; `checker_options.line = "first"` compares first lines only | a string or list |

Checkers are deterministic code: no LLM grades another LLM here. The `number`, `choice` and `json_fields`
parsers come from the studies, and re-scoring all 9,870 replies from the quantization study's main run with them gives
the study's own scores on every one (`scripts/check_against_studies.py`).

## How the numbers are computed

- **Unit of resampling: the item.** 2,000 bootstrap draws, percentile intervals, seeded, so the same runs
  always give the same interval. With a `group`, items are resampled within each group.
- **Comparisons are paired** over the items both runs answered, using the first pass of each. If the two
  runs used different versions of a suite (the hash of its spec and items differs), the comparison says so.
- **Noise floor**: with `--repeat`, every item is sent a second time as a separate call (its own cache entry).
  The floor is the share of items whose parsed answer changed between the passes. A comparison calls a change
  "beyond noise" only if, per item, (changed between runs) minus (changed between the passes) has an interval
  above zero for every run that has a floor.
- Items are taken in file order, so `--limit 50` is the same 50 items every time and limited runs still pair up.

## Where things live

`$EVALPAIR_HOME`, or `~/.evalpair`:

```
cache.sqlite           every reply, keyed by request
runs/<run>/meta.json   suite and its hash, model, digest, backend, options, counts, timings
runs/<run>/rows.jsonl  one line per item and pass: reply, parsed answer, score, tokens
suites/<name>/         your suites
```

## Limits

- One model per run, chat-style prompts, one turn. No tool use, multi-turn or constrained decoding yet.
- Deterministic checkers only, so open-ended answers need a suite whose answer can be checked by code.
- The noise floor measures run-to-run variation on the same server. It doesn't cover a different
  machine or a server update.

## License

MIT for the code. The bundled suites keep their sources' licenses (MIT for GSM8K and MMLU, CC BY-SA 4.0 for
E2E NLG); each `suite.toml` names its license and source.
