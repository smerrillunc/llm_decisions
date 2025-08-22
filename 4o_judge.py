import re
import json
import openai
import argparse
import os

def build_comparison_prompt(reference, response_a, response_b):
    return f"""
You are an impartial evaluator comparing two responses (A and B) to a REFERENCE.

Your ONLY goal is to determine which response most closely matches the **content and tone** of the REFERENCE.

DO reward:
- Similarity in tone, vagueness, informality, or stream-of-consciousness phrasing — **only if these are present in the REFERENCE.**
- Alignment with the key points and main ideas expressed in the REFERENCE.

Return ONLY a JSON object in this format:
{{
  "winner": "A" or "B" or "Tie",
  "justification": "1–2 sentences clearly explaining your choice, consistent with the winner."
}}

Reference:
\"\"\"{reference.strip()}\"\"\"

Response A:
\"\"\"{response_a.strip()}\"\"\"

Response B:
\"\"\"{response_b.strip()}\"\"\"

Output JSON Now:

""".strip()



def build_monologue_prompt_combined(prompt, monologue, model_response):
    instructions = {
        "fit": (
            "Evaluate how well the model's response matches the ground truth monologue in terms of meaning, intent, and content. "
            "It does not need to be word-for-word, but should be semantically and thematically consistent."
        ),
        "style": (
            "Evaluate how closely the model's response matches the tone, voice, and rhetorical style of the monologue. "
            "This includes emotional tone, word choice, cadence, and personality."
        )
    }

    base_prompt = f"""
You are an impartial expert evaluator. Please evaluate the model's response on **two aspects**: meaning/content fit and style/voice.

1. **Fit**: {instructions['fit']}
2. **Style**: {instructions['style']}

Scoring guide for each aspect:
5 = Excellent
4 = Good
3 = Fair
2 = Weak
1 = Poor

Prompt (reverse-engineered from monologue):
\"\"\"{prompt}\"\"\"

Reference Monologue:
\"\"\"{monologue}\"\"\"

Model Response:
\"\"\"{model_response}\"\"\"

Respond ONLY with a JSON object like:
{{
    "fit_score": 4,
    "fit_explanation": "...",
    "style_score": 5,
    "style_explanation": "..."
}}

Now output your JSON Object:
"""

    return base_prompt


def safe_parse_json_block(text: str):
    """
    Extract and parse JSON from a markdown code block.
    Attempts best-effort recovery if JSON is truncated.
    """
    # Remove code fences like ```json ... ```
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip(), flags=re.DOTALL)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try recovery for truncated JSON
        fixed = cleaned

        # If an unterminated string, close it
        if "Unterminated string" in str(e):
            fixed += '"'  # close the last string
        # If missing closing brace, add one
        if not fixed.strip().endswith("}"):
            fixed += "}"

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # As a last resort, try partial parse: keep only valid JSON prefix
            # This cuts off at the last full JSON object boundary
            last_brace = fixed.rfind("}")
            if last_brace != -1:
                try:
                    return json.loads(fixed[:last_brace+1])
                except Exception:
                    pass
            # Couldn’t recover
            
    return None

def get_gpt4o_response(prompt, max_tokens=300, retries=3):
    """
    Query GPT-4o and return its response text.
    Retries up to `retries` times if the JSON parse fails.
    """
    attempt = 0
    while attempt < retries:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1
        )
        text = response.choices[0].message.content
        print(text)
        parsed = safe_parse_json_block(text)
        if parsed is not None:
            return parsed
        else:
            print(f"Attempt {attempt + 1} failed to parse JSON. Retrying...")

        attempt += 1

    # If still failing after retries, raise error
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment using a judge LLM.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSON data file.")

    args = parser.parse_args()
    openai.api_key = 'REMOVED'

    print("Loading data...")
    with open(args.data_file, 'r') as file:
        data = json.load(file)
    
    agents = data.keys()

    total_estimated_cost = 0

    for agent in agents:
        print(f"Processing agent: {agent}")
        for item in data[agent]:
            item['gpt_judgment'] = []
            item['final_comparison'] = {}
            
            prompt = item['prompt'].split('unknownspeaker:')[1].split('<|eot_id|>')[0]
            monologue = item['monologue']
            gpt_response = item['gpt_response']
            model_response = item['model_responses'][0]

            model_score_prompt = build_monologue_prompt_combined(prompt, monologue, model_response)
            gpt_score_prompt = build_monologue_prompt_combined(prompt, monologue, gpt_response)
            compare_prompt = build_comparison_prompt(monologue, model_response, gpt_response)

            
            model_score_result = get_gpt4o_response(model_score_prompt)
            gpt_score_result = get_gpt4o_response(gpt_score_prompt)
            compare_result = get_gpt4o_response(compare_prompt)

            if model_score_result is None or gpt_score_result is None or compare_result is None:
                continue

            item['gpt_judgment'] = [{'response_idx': 0,
                                    'aspect': 'fit',
                                    'score': model_score_result.get('fit_score', 0),
                                    'explanation': model_score_result.get('fit_explanation', '') },
                                    
                                    {'response_idx': 0,
                                    'aspect': 'style',
                                    'score': model_score_result.get('style_score', 0),
                                    'explanation': model_score_result.get('style_explanation', '') },
                                    
                                    {'response_idx': 'gpt',
                                    'aspect': 'fit',
                                    'score': gpt_score_result.get('fit_score', 0),
                                    'explanation': gpt_score_result.get('fit_explanation', '') },
                                    
                                    {'response_idx': 'gpt',
                                    'aspect': 'style',
                                    'score': gpt_score_result.get('style_score', 0),
                                    'explanation': gpt_score_result.get('style_explanation', '')}]
            
            item['final_comparison'] = {'winner': compare_result.get('winner', ''),
                                        'justification': compare_result.get('justification', '')
                                        }

            
        base, ext = os.path.splitext(args.data_file)
        output_file = f"{base}_4o{ext}"

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)



if __name__ == "__main__":
    main()
