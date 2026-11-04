"""Build the bundled suites from the item set of the Measuring LLM Systems studies.

The items are quant-cliff's main split (data/items.jsonl): 160 GSM8K and 160 MMLU questions, the same seeded
draws as prompt-brittleness, format-tax and groupthink, and format-tax's 150 E2E restaurant records. The
source files are pinned to Hugging Face revisions with sha256 hashes in quant-cliff's config.py. Prompts are
those studies' free-text, reasoning-first prompts, word for word.

    python scripts/make_suites.py ../quant-cliff/data/items.jsonl
"""

import json
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "evalpair", "suites")
GSM8K = 'Solve the following math problem. Reason step by step, then write the final answer on the last line as "Answer: <number>".'
MMLU = ('Answer the following multiple-choice question about {subject_name}. Reason step by step, then write '
        'the letter of the correct option on the last line as "Answer: <letter>".')
EXTRACT = (
    "Convert the restaurant record below into a JSON object with exactly these keys:\n"
    "- name: the venue's name\n"
    '- eat_type: one of "restaurant", "pub", "coffee shop"\n'
    "- food: the type of food served\n"
    '- price_range: one of "cheap", "moderate", "high", "less than £20", "£20-25", "more than £30"\n'
    '- customer_rating: one of "low", "average", "high", "1 out of 5", "3 out of 5", "5 out of 5"\n'
    '- area: one of "riverside", "city centre"\n'
    "- family_friendly: true or false\n"
    "- near: a nearby landmark\n"
    "Use null for any field the record does not give. Reply with only the JSON object.\n\n"
    "Record:\n{record}"
)


def toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # a TOML basic string is a JSON string


def write(name, spec, items):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    lines = []
    for k, v in spec.items():
        lines.append(f"{k} = {v if isinstance(v, int) else toml_str(v)}")
    with open(os.path.join(d, "suite.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(name, len(items))


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    main_ = [r for r in rows if r["split"] == "main"]
    g = [{"id": r["id"], "question": r["question"], "gold": r["gold"]} for r in main_ if r["task"] == "gsm8k"]
    m = [{"id": r["id"], "subject": r["subject"],
          "subject_name": r["subject"].replace("high_school_", "high school ").replace("_", " "),
          "question": r["question"], "options": r["options"], "gold": r["gold"]} for r in main_ if r["task"] == "mmlu"]
    e = [{"id": r["id"], "record": r["record"], "gold": r["gold"]} for r in main_ if r["task"] == "drift"]
    write("gsm8k-160", {
        "description": "Grade-school maths word problems (GSM8K test), reasoning first, graded on the final number.",
        "checker": "number", "max_tokens": 1024, "user": GSM8K + "\n\nProblem: {question}",
        "license": "MIT", "source": "openai/gsm8k (test), revision 740312a"}, g)
    write("mmlu-hs-160", {
        "description": "Multiple choice from four high-school MMLU subjects, reasoning first, graded on the letter.",
        "checker": "choice", "max_tokens": 768, "group": "subject",
        "user": MMLU + "\n\nQuestion: {question}\n\n{options}",
        "license": "MIT", "source": "cais/mmlu (test: high school biology, geography, macroeconomics, psychology), revision c30699e"}, m)
    write("e2e-extract-150", {
        "description": "Restaurant records to a fixed 8-field JSON object (E2E NLG), graded field by field.",
        "checker": "json_fields", "max_tokens": 300, "user": EXTRACT,
        "license": "CC BY-SA 4.0", "source": "tuetschek/e2e_nlg_cleaned (test), revision 6f77a74"}, e)


if __name__ == "__main__":
    main(sys.argv[1])
