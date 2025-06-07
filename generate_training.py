import os
from pathlib import Path
from llm_utils.llm_parsers import process_llm_response, LLMResponse

OUTPUT_DIR = Path("outputs")
TRAINING_OUTPUT = Path("training_data.jsonl")

def parse_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        prompt_start = content.index("[PROMPT]") + len("[PROMPT]")
        response_start = content.index("[RESPONSE]")
    except ValueError:
        return None  # Skip malformed files

    prompt = content[prompt_start:response_start].strip()
    response = content[response_start + len("[RESPONSE]"):].strip()

    result = process_llm_response(response)

    if result == LLMResponse.AFFIRMATIVE:
        label = "1"
    elif result == LLMResponse.NEGATIVE:
        label = "0"
    else:
        return None  # Skip ambiguous or failed responses

    return prompt, response, label

import json

def main():
    with open(TRAINING_OUTPUT, "w", encoding="utf-8") as out_f:
        for file_path in OUTPUT_DIR.glob("*.txt"):
            parsed = parse_file(file_path)
            if parsed:
                prompt, response, label = parsed
                json.dump({"text": response, "label": int(label)}, out_f)
                out_f.write("\n")

if __name__ == "__main__":
    main()