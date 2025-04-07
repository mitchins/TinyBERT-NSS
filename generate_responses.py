from typing import List
import random
import llm
from enum import Enum
import os
from pathlib import Path
import concurrent.futures

# Where to dump the simple text responses
OUTPUT_DIR = 'outputs'
LLM_REQUEST_TIMEOUT = 10  # in seconds

llm_models = [
    'gemma3-4b-it_local',
    'qwen2.5-7b-instruct-mlx_local',
    'llama-3.2-3b-instruct_local',
    'mistral-nemo-instruct-2407_local'
]

class PromptType(Enum):
    SCENE_CHANGE = "scene_change"
    RECOMMENDATION = "recommendation"
    END_OF_PASSAGE = "end_of_passage"
    CLASSIFICATION_CORRECTNESS = "classification_correctness"
    NEW_CHARACTER_INTRODUCED = "new_character_introduced"
    CONTRADICTION_PRESENT = "contradiction_present"
    REASONING_COMPLETE = "reasoning_complete"
    TONE_CONSISTENCY = "tone_consistency"
    INSTRUCTION_FULFILLED = "instruction_fulfilled"
    PARAGRAPH_SPLIT_SUGGESTED = "paragraph_split_suggested"

PROMPT_TEMPLATES = {
    PromptType.SCENE_CHANGE: [
        "Answer yes or no: does the scene change in the following text?\n\n{passage}",
        "In the following text, is there a change of scene or setting?\n\n{passage}",
        "Does this passage represent a break between scenes or narrative beats?\n\n{passage}",
        "Answer [Y/N]: is this a scene transition point?\n\n{passage}",
        "Would a scriptwriter likely treat this as a new scene? Respond with yes or no.\n\n{passage}",
    ],
    PromptType.RECOMMENDATION: [
        "Does the following text contain a recommendation or suggestion?\n\n{passage}",
        "Answer yes or no: is there an explicit recommendation made in the passage below?\n\n{passage}",
        "In the passage below, is the speaker advising a course of action or making a suggestion?\n\n{passage}",
        "Respond [Y/N]: does the following text include any recommendations?\n\n{passage}",
        "Would you say this text offers guidance or makes a recommendation? Answer yes or no.\n\n{passage}",
    ],
    PromptType.END_OF_PASSAGE: [
        "Is the following passage a natural place to end the section?\n\n{passage}",
        "Would this be a reasonable stopping point in the text? Answer yes or no.\n\n{passage}",
        "Does this feel like a conclusion or wrap-up of the current passage?\n\n{passage}",
        "Answer [Y/N]: is this a good place to pause the narrative?\n\n{passage}",
        "Would you consider this the end of a narrative segment? Answer yes or no.\n\n{passage}",
    ],
    PromptType.CLASSIFICATION_CORRECTNESS: [
        "Is the given classification of this text accurate? Answer yes or no.\n\n{passage}",
        "Does the label applied to this passage seem correct? Respond [Y/N].\n\n{passage}",
        "Would you agree with the classification of this content? Yes or no.\n\n{passage}",
        "Based on the passage, does the stated classification hold up? Answer [Y/N].\n\n{passage}",
        "Evaluate the classification of this text. Is it appropriate? Respond yes or no.\n\n{passage}",
    ],
    PromptType.NEW_CHARACTER_INTRODUCED: [
        "Is a new character introduced in the following text? Answer yes or no.\n\n{passage}",
        "Does the passage include the introduction of someone not previously mentioned? Respond [Y/N].\n\n{passage}",
        "Would you say a new character enters the narrative here? Yes or no.\n\n{passage}",
        "Does this section bring in a new individual or figure? Answer [Y/N].\n\n{passage}",
        "Does the passage introduce anyone not yet seen before? Respond yes or no.\n\n{passage}",
    ],
    PromptType.CONTRADICTION_PRESENT: [
        "Does the passage contain a contradiction or conflicting statement? Answer yes or no.\n\n{passage}",
        "Is there an inconsistency or contradiction in the text below? Respond [Y/N].\n\n{passage}",
        "Would you say the speaker contradicts themselves in this passage? Yes or no.\n\n{passage}",
        "Based on the passage, is there any self-contradictory logic or phrasing? Answer [Y/N].\n\n{passage}",
        "Does this text contain conflicting information or ideas? Respond yes or no.\n\n{passage}",
    ],
    PromptType.REASONING_COMPLETE: [
        "Is the reasoning in this passage complete and coherent? Answer yes or no.\n\n{passage}",
        "Would you say the argument or explanation given is self-contained? Respond [Y/N].\n\n{passage}",
        "Does the text offer a clear and logical conclusion based on the reasoning provided? Yes or no.\n\n{passage}",
        "Is the thought process or rationale in this section fully developed? Answer [Y/N].\n\n{passage}",
        "Does this passage demonstrate clear and complete reasoning? Respond yes or no.\n\n{passage}",
    ],
    PromptType.TONE_CONSISTENCY: [
        "Is the tone of this passage consistent with what preceded it? Answer yes or no.\n\n{passage}",
        "Would you say the emotional tone remains steady in this section? Respond [Y/N].\n\n{passage}",
        "Does the tone shift noticeably in the following text? Yes or no.\n\n{passage}",
        "Is the narrative voice or mood stable throughout this excerpt? Answer [Y/N].\n\n{passage}",
        "Does this passage maintain tonal continuity with earlier material? Respond yes or no.\n\n{passage}",
    ],
    PromptType.INSTRUCTION_FULFILLED: [
        "Does this passage fulfill the given instruction or task? Answer yes or no.\n\n{passage}",
        "Would you say the text below achieves what was asked? Respond [Y/N].\n\n{passage}",
        "Based on the content, does it complete the instruction? Yes or no.\n\n{passage}",
        "Is the speaker responding appropriately to the original request? Answer [Y/N].\n\n{passage}",
        "Has the instruction been successfully carried out in the passage? Respond yes or no.\n\n{passage}",
    ],
    PromptType.PARAGRAPH_SPLIT_SUGGESTED: [
        "Should the following paragraph be split for clarity or pacing? Answer yes or no.\n\n{passage}",
        "Would breaking this paragraph into smaller parts improve readability? Respond [Y/N].\n\n{passage}",
        "Is this paragraph too dense or long to stand on its own? Yes or no.\n\n{passage}",
        "Could this text benefit from paragraph separation? Answer [Y/N].\n\n{passage}",
        "Does the paragraph include multiple distinct ideas that suggest a split? Respond yes or no.\n\n{passage}",
    ],
}

def get_prompts(prompt_type: PromptType, passage: str) -> List[str]:
    return [template.format(passage=passage) for template in PROMPT_TEMPLATES[prompt_type]]

def random_prompt(passage: str, prompt_type: PromptType = None) -> str:
    if prompt_type:
        templates = PROMPT_TEMPLATES[prompt_type]
        return random.choice([template.format(passage=passage) for template in templates])
    else:
        all_templates = []
        for templates in PROMPT_TEMPLATES.values():
            all_templates.extend(templates)
        return random.choice([template.format(passage=passage) for template in all_templates])

def get_all_prompts(passage: str, prompt_type: PromptType = None) -> List[str]:
    if prompt_type:
        return [template.format(passage=passage) for template in PROMPT_TEMPLATES[prompt_type]]
    else:
        prompts = []
        for templates in PROMPT_TEMPLATES.values():
            prompts.extend([template.format(passage=passage) for template in templates])
        return prompts

def send_prompt(llm_client, passage, prompt_text):
    return llm_client.prompt(passage, system=prompt_text, temperature=0.2)

def run_all_prompts(samples_dir="samples", output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    for model in llm_models:
        llm_client = llm.get_model(model)

        for filename in os.listdir(samples_dir):
            if not filename.endswith(".txt"):
                continue

            with open(os.path.join(samples_dir, filename), "r") as f:
                raw = f.read()

            # Strip metadata and isolate passage
            if "---" in raw:
                parts = raw.split("---")
                passage = parts[-1].strip()
            else:
                passage = raw.strip()

            sample_id = filename.replace(".txt", "")

            for prompt_type in PromptType:
                for i, prompt_template in enumerate(PROMPT_TEMPLATES[prompt_type]):
                    prompt_text = prompt_template.format(passage=passage)

                    output_path = Path(output_dir) / f"{sample_id}__{prompt_type.value}__{i}__{model}.txt"
                    if output_path.exists():
                        print(f"[SKIP] {output_path} already exists. Skipping generation.")
                        continue

                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(send_prompt, llm_client, passage, prompt_text)
                            response = future.result(timeout=LLM_REQUEST_TIMEOUT)
                            if response and hasattr(response, "text"):
                                result = response.text()
                            else:
                                print(f"[SKIP] {output_path} – no valid response returned.")
                                continue
                    except concurrent.futures.TimeoutError:
                        print(f"[SKIP] {output_path} – prompt timed out after {LLM_REQUEST_TIMEOUT}s.")
                        continue
                    except Exception as e:
                        print(f"[SKIP] {output_path} – LLM request failed: {e}")
                        continue

                    with open(output_path, "w") as out_f:
                        out_f.write(f"[PROMPT]\n{prompt_text}\n\n[RESPONSE]\n{result}")

if __name__ == "__main__":
    run_all_prompts()