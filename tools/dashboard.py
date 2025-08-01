import streamlit as st
import os
import json
from PIL import Image
import streamlit.components.v1 as components
import base64
import pandas as pd

# --- CONFIG ---
BASE_DIR = "results"
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
EVALS_DIR = os.path.join(BASE_DIR, 'evals')


# 1. Define a specific captions dictionary (put this at the module level)
# Captions for Completion Experiments
completion_image_captions = {
    "completion_agent_win_rate.png": "Per-agent win rates showing how often each model's response is preferred over LLaMA‑70B in completion tasks.",
    "completion_overall_win_rate.png": "Aggregate win rate across all prompts, indicating how often each model is preferred over LLaMA‑70B.",
    "agent_violin.png": "Distribution of 1–5 judge scores per agent, illustrating variation in response quality across models.",
    "model_win_rate.png": "Breakdown of how frequently each model's responses are selected as best over LLaMA‑70B.",
    "violin_compare.png": "Comparison of 1–5 score distributions between fine-tuned models and LLaMA‑70B, showing differences in output quality.",
}

# Captions for Monologue Experiments
monologue_image_captions = {
    "monologue_agent_win_rate.png": "Per-agent win rates showing how often the fine-tuned model outperforms LLaMA‑70B on monologue prompts.",
    "monologue_overall_win_rate.png": "Overall model preference rate across all monologue prompts, based on comparisons with LLaMA‑70B.",
    "agent_violin.png": "Judge score distributions (1–5 scale) for each model, indicating variation in style fidelity and response quality.",
    "violin_compare.png": "Side-by-side distribution comparison of 1–5 judge scores for fine-tuned vs LLaMA‑70B models on monologue completions.",
}

# Captions for Alignment Experiments
alignment_image_captions = {
    "memory_overall_win_rate.png": "Overall win rate showing how well fine-tuned models recall agent-specific memories compared to LLaMA‑70B.",
    "memory_agent_win_rate.png": "Per-agent win rates for memory alignment, reflecting how well fine-tuned models retrieve specific facts over LLaMA‑70B.",
    "belief_overall_win_rate.png": "Overall model preference rate on belief-alignment tasks versus LLaMA‑70B, assessing alignment with speaker values.",
    "belief_agent_win_rate.png": "Per-agent win rates showing how well fine-tuned models reflect speaker beliefs compared to LLaMA‑70B.",
    "personality_agent_win_rate.png": "Per-agent win rates evaluating how well the model reflects speaker personality traits versus LLaMA‑70B.",
    "personality_overall_win_rate.png": "Aggregate win rate on personality-alignment prompts, showing fine-tuned model performance relative to LLaMA‑70B.",
    "agent_violin.png": "Judge score distributions (1–5 scale) across agents and alignment dimensions, showing variation in trait alignment.",
    "violin_compare.png": "Comparison of 1–5 alignment scores between fine-tuned models and LLaMA‑70B across all traits.",
}

# Captions for Vote Prediction Experiments
votes_image_captions = {
    "votes_accuracy_pie.png": "Overall distribution of correct vs incorrect vote predictions across all board members.",
    "votes_accuracy_by_agent.png": "Prediction accuracy by school board member, showing per-agent vote prediction performance.",
    "votes_confusion_matrix.png": "Confusion matrix showing patterns of misclassified Aye, Naye, and Unknown votes by the model.",
}

# Captions for Single-Speaker Fool Rate Experiments
single_speaker_image_captions = {
    "completion_fool_rates.png": "Fool rates for completion data, showing how often model outputs are mistaken for real speaker utterances by binary classifiers.",
    "belief_fool_rates.png": "Fool rates on belief-alignment data, reflecting how convincingly the model mimics speaker beliefs.",
    "personality_fool_rates.png": "Fool rates on personality-aligned completions, measuring how well models imitate speaker personality traits.",
    "fool_rates_by_agent.png": "Per-agent fool rates across datasets, indicating how successfully each model mimics individual speakers.",
    "monologue_fool_rates.png": "Fool rates on monologue completions, assessing realism of long-form generated speech.",
    "memory_fool_rates.png": "Fool rates on memory-alignment data, evaluating whether models capture and reflect speaker-specific facts.",
    "fool_rates_by_dataset.png": "Aggregated fool rates across all datasets and evaluation subsets, showing overall speaker mimicry strength.",
}

# Captions for Multi-Speaker Classification Experiments
multi_speaker_image_captions = {
    "beliefs.png": "Speaker classification accuracy by agent on belief-alignment completions, reflecting distinctiveness in expressed views.",
    "personality.png": "Classification accuracy by agent on personality-aligned completions, assessing stylistic fidelity.",
    "memory.png": "Classifier accuracy on memory-aligned content, evaluating reproduction of speaker-specific references.",
    "monologues.png": "Speaker classification accuracy by agent on monologue completions, indicating long-form voice consistency.",
    "completions.png": "Speaker classification accuracy for general completions, measuring baseline speaker imitation quality.",
    "all.png": "Overall speaker classification accuracy aggregated across all datasets, summarizing model voice fidelity.",
    "comparison.png": "Comparison of classification accuracies across datasets and agents, highlighting relative model fidelity by domain.",
}

# --- GENERIC TAB MARKDOWNS ---
    
EXPERIMENT_DESCRIPTIONS = {
    "completion": """
### Completion Dataset
- **Dataset**: The completion dataset consists of chat-completions from a held outset of meeting transcriptions.  Evaluates dialogue completion quality using real conversational prompts and ground truth continuations.
- **Setup**: Both fine-tuned and baseline (LLaMA‑70B with in-context persona) models generate completions for the same prompt. 
- **Evaluation**:
  - **Win Rate vs LLaMA‑70B**: A LLaMA‑70B judge decides produces a response more aligned with the ground of truth in terms of tone, sentiment, and vocabulary.
  - **Judge Scores (1–5)**: LLaMA‑70B scores each response for content alignment and quality.
- **Output Metrics**: Overall and dimension-wise win rates, score distributions, and qualitative themes from judge justifications.
""",

    "monologue": """
### Monologue Dataset
- **Dataset**:  Long speaker monologues (greater than 150 words) from transcripts are reverse-prompted into questions.  We ask LLaMA-70B to come up with a question that generated this monologue.  The completion dataset comprises of some short completions which can introduce some noise.  These longer form monolgues encourage longer answers and allow us to better study response tone and style.
- **Setup**: Both fine-tuned and baseline (LLaMA‑70B with in-context persona) models generate responses to the monologue question.
- **Evaluation**:
  - **Win Rate vs LLaMA‑70B**: A LLaMA‑70B judge decides produces a response more aligned with the ground of truth.
  - **Judge Scores (1–5)**: LLaMA‑70B scores each resopnse for quality, personality fit, and relevance.
- **Output Metrics**: Win rate, score distributions, and qualitative analysis of response preferences.
""",

    "alignment": """
### Alignment Dataset

- **Dataset**: Meeting transcripts are scraped for instances where speakers discuss their beliefs, memories, or personality traits.  LLaMA-70B is then used to come up with questions to ask the LLMs to express beliefs, recall memories and reveal personallity traits.
- **Setup**: LongBoth fine-tuned and baseline (LLaMA‑70B with in-context persona) models generate responses to the alignment questions.
- **Evaluation**:
  - **Win Rate vs LLaMA‑70B**: Judged on trait reflection accuracy.
  - **Judge Scores (1–5)**: Ratings based on alignment strength.
- **Output Metrics**: Win rates per trait, overall alignment accuracy, and score distributions.
""",

    "votes": """
### Vote Prediction Evaluation

- **Dataset**: Historical school board votes from meeting minutes were scraped and turned into yes/no questions.  Fine-tuned models predict each member's vote.
- **Evaluation**:
  - **Correctness**: Prediction compared to actual vote.
- **Output Metrics**: Overall accuracy, per-agent accuracy, and confusion matrices highlighting prediction errors.
""",

    "multi-speaker": """
### Multi-Speaker Classificaiton Evaluation

- **Datasets**: Completion, Monologue and Alignment (beliefs, memory and personality questions)
- **Setup**: A 7-way classifier trained on real utterances from transcripts is used to label completions generated by each fine-tuned model.  We assess if this classifier can associate the correct labels to the fine-tuned model the generated an utterance.
- **Evaluation**:
  - **Accuracy**: Whether the classifier correctly identifies the intended speaker.
- **Output Metrics**: Accuracy by dataset and by speaker, illustrating model fidelity across multiple linguistic dimensions.
""",

    "single-speaker": """
### Single-Speaker Experiments

- **Datasets**: Completion, Monologue and Alignment (beliefs, memory and personality questions)
- **Setup**: One-vs-all classifiers are trained for each speaker to identify if an utterance is from that speaker or some other speaker.  These classifiers then label the fine-tuned model completions.
- **Evaluation**:
  - **Fool Rate**: Percentage off fine-tuned model utterancess that successfully pass as real to the classifier.
- **Output Metrics**: Fool rates by agent, by dataset, and by alignment dimensions (beliefs, personality, memory).
""",

    "Prediction Model Evals": """
### Prediction Model Evaluations

This section summarizes test set validation performance of the Single-Speaker and Multi-Speaker Classificaiton models.

- **Evaluation**: Aggregates confusion matrices, score distributions, and classifier performance across all speaker modeling experiments.
""",
}

# --- GENERIC IMAGE CAPTIONS ---
# Fills with placeholder captions by default.
# 2. Use generic captions as fallback for images not covered above
def get_caption(img_name):
    lower = img_name.lower()
    # Try for a specific caption first
    if img_name in completion_image_captions:
        return completion_image_captions[img_name]
    if img_name in monologue_image_captions:
        return monologue_image_captions[img_name]
    if img_name in alignment_image_captions:
        return alignment_image_captions[img_name]
    if img_name in votes_image_captions:
        return votes_image_captions[img_name]
    if img_name in single_speaker_image_captions:
        return single_speaker_image_captions[img_name]
    if img_name in multi_speaker_image_captions:
        return multi_speaker_image_captions[img_name]

    # Generic fallback logic (from your previous function)
    if "win_rate" in lower:
        return "Illustrates model win rates across evaluation conditions."
    if "violin" in lower or "distribution" in lower or "compare" in lower:
        return "Shows the distribution of scores or comparative results for the models."
    if "confusion" in lower or "cm" in lower:
        return "Confusion matrix summarizing classification performance."
    if "scores" in lower:
        return "Model prediction score/certainty distribution."
    if "pie" in lower:
        return "Pie chart summarizing evaluation results within this category."
    if "fool" in lower:
        return "Displays 'fool rate' statistics, indicating misclassification rates."
    return "Figure relevant to this experiment category."




# --- HELPERS ---
@st.cache_data
def get_param_sets(figures_dir):
    if os.path.exists(figures_dir):
        return sorted([d for d in os.listdir(figures_dir) if os.path.isdir(os.path.join(figures_dir, d))])
    return []

def get_categories(param_set):
    desired_order = [ "completion", "monologue", "alignment", "votes", "multi-speaker", "single-speaker"
    ]
    path = os.path.join(FIGURES_DIR, param_set)
    existing = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return ['overview'] + [d for d in desired_order if d in existing]

def get_image_files(param_set, category):
    folder = os.path.join(FIGURES_DIR, param_set, category)
    return [
        (f, os.path.join(folder, f))
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

def load_layout(param_set, category):
    path = os.path.join(LAYOUT_DIR, f"{param_set}_{category}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {"order": data}
        elif isinstance(data, dict):
            return data
    return None

def display_carousel(image_paths, tab_key):
    if tab_key not in st.session_state:
        st.session_state[tab_key] = 0

    index = st.session_state[tab_key]
    max_index = len(image_paths) - 1

    img_path = image_paths[index]
    img_name = os.path.basename(img_path)
    st.markdown(f"**{img_name}** ({index + 1}/{len(image_paths)})")

    with open(img_path, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()
    image_html = f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{encoded}"
             style="max-width:750px; width:100%; height:auto; border:1px solid #ddd; border-radius:4px; padding:5px;">
    </div>
    """
    components.html(image_html, height=650)

    description = get_caption(img_name)
    if description:
        st.markdown(f'''
            <p style="
                font-size:18px;
                line-height:1.5;
                margin-top:10px;
                margin-bottom:24px;
                color:#333333;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                {description}
            </p>
        ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous", key=f"prev_{tab_key}"):
            st.session_state[tab_key] = max_index if index == 0 else index - 1
    with col3:
        if st.button("➡️ Next", key=f"next_{tab_key}"):
            st.session_state[tab_key] = 0 if index == max_index else index + 1

# --- UI START ---
st.set_page_config(page_title="Results Dashboard", layout="wide")
st.title("Fine-Tuned Agent Dashboard")

exp_names = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) if 'evals' not in d]
selected_experiment = st.selectbox("Select Experiment", exp_names, key="experiment_selector")

# Redefine FIGURES_DIR dynamically based on selected experiment
FIGURES_DIR = os.path.join(BASE_DIR, selected_experiment, 'figures')

param_sets = get_param_sets(FIGURES_DIR)
selected_param = st.selectbox("Select Parameter Set", param_sets)

# Explanation block
st.markdown("""
##### Parameter Descriptions
- **T (Temperature)**: Controls the randomness of the model's output. Lower values (e.g., 0.2) make responses more focused and deterministic; higher values (e.g., 0.8) increase creativity and diversity.
- **P (Top-p / Nucleus Sampling)**: Limits sampling to the smallest possible set of tokens whose cumulative probability is ≥ *p*. Helps balance randomness and relevance (e.g., p=0.9 means sampling from the top 90% of probability mass).
- **K (Top-k Sampling)**: Restricts the model to sampling from the top *k* most likely next tokens. Smaller *k* reduces randomness; larger *k* allows more variation.
- **R (Repetition Penalty)**: Penalizes tokens that have already been generated to reduce repetition. Values >1 discourage repetition (e.g., 1.2); values close to 1 have little effect.
""")

categories = get_categories(selected_param)
categories.append("Prediction Model Evals")
tabs = st.tabs(categories)

for tab, category in zip(tabs, categories):
    with tab:
        if category =='overview':
            st.markdown("### Experiment Overview")

            # Direct DataFrame rendering (preferred)
            st.markdown("### Model Summary")
            with open(os.path.join(BASE_DIR, selected_experiment, "model_summary.md"), "r") as f:
                st.markdown(f.read())

            st.markdown("### Perplexity Results")
            img_path = os.path.join(BASE_DIR, selected_experiment, 'perplexity.png')
            st.image(img_path, caption="Perplexity", use_container_width=True)
            continue

        # Insert generic experiment description per tab
        st.markdown(EXPERIMENT_DESCRIPTIONS.get(category, f"**Experiment Overview: {category.capitalize()}**"))

        # "completion", "monologue", etc.
        if category in ["completion",  "monologue", "alignment", "multi-speaker", "single-speaker", "votes"]:
            images = get_image_files(selected_param, category)
            if not images:
                st.info("No images available for this category.")
                continue
            image_paths = [img_path for _, img_path in images]
            display_carousel(image_paths, f"{selected_param}_{category}")



             ########### Completion TAB Manual Review ##############
            if category == "completion":
                # --- Manual Review Section for Alignment ---
                st.markdown("---")
                st.subheader("Manual Review: Completion Data")

                completion_path = os.path.join(BASE_DIR, selected_experiment, "completion_results", f"test_responses_{selected_param}.json")

                if os.path.exists(completion_path):
                    with open(completion_path, "r") as f:
                        completion_data = json.load(f)

                    speaker_names = list(completion_data.keys())
                    selected_speaker = st.selectbox("Select Agent", speaker_names, key="completion_agent")

                    speaker_data = completion_data[selected_speaker]

                    comp_idx_key = f'comp_index_{selected_param}_{selected_speaker}'
                    comp_prev_key = f'comp_prev_speaker_{selected_param}'

                    if comp_idx_key not in st.session_state:
                        st.session_state[comp_idx_key] = 0
                    if comp_prev_key not in st.session_state:
                        st.session_state[comp_prev_key] = None
                    if st.session_state[comp_prev_key] != selected_speaker:
                        st.session_state[comp_idx_key] = 0
                        st.session_state[comp_prev_key] = selected_speaker

                    def comp_go_prev():
                        st.session_state[comp_idx_key] = (st.session_state[comp_idx_key] - 1) % len(speaker_data)

                    def comp_go_next():
                        st.session_state[comp_idx_key] = (st.session_state[comp_idx_key] + 1) % len(speaker_data)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.button("⬅️ Previous", on_click=comp_go_prev, key=f"comp_prev_button_{selected_param}")
                    with col2:
                        st.markdown(f"### Example {st.session_state[comp_idx_key] + 1} of {len(speaker_data)}")
                    with col3:
                        st.button("➡️ Next", on_click=comp_go_next, key=f"comp_next_button_{selected_param}")

                    item = speaker_data[st.session_state[comp_idx_key]]

                    st.markdown("#### Prompt")
                    st.text_area("Prompt", item.get("prompt", ""), height=80)

                    st.markdown("#### True Completion")
                    st.text_area("True Completion", item.get("true_completion", ""), height=80)


                    st.markdown("#### A. Agent Response")
                    st.text_area("Agent Response", item.get("model_responses", "")[0], height=80)

                    st.markdown("#### B. LLaMA‑70B Response")
                    st.text_area("LLaMA‑70B Response", item.get("gpt_response", ""), height=80)

                    st.markdown("#### Scores")
                    
                    judgement = pd.DataFrame(item.get('gpt_judgment'))
                    filtered_df = judgement[judgement['response_idx'].isin([0, 'gpt'])]

                    def create_markdown_table(df):
                        md = "| Aspect | Score | Explanation |\n"
                        md += "|--------|-------|-------------|\n"
                        for _, row in df.iterrows():
                            md += f"| **{row['aspect']}** | {row['score']} | {row['explanation']} |\n"
                        return md
                    
                    judgement = pd.DataFrame(item.get('gpt_judgment'))
                    # Group by response_idx and display each separately
                    for idx in [0, 'gpt']:
                        if idx == 0:
                            name = 'Model'
                        else:
                            name = 'LLaMA‑70B'

                        st.subheader(f"{name} Response")
                        sub_df = filtered_df[filtered_df['response_idx'] == idx]
                        st.markdown(create_markdown_table(sub_df), unsafe_allow_html=True)

                    st.markdown("#### Comparison")
                    st.write('**Winner:**', item.get("final_comparison", {}).get('winner', 'Error parsing response'))
                    st.write('**Justification:**', item.get("final_comparison", {}).get('justification', 'Error parsing response'))
                else:
                    st.warning("Could not find the completion data file.")


             ########### Monologue TAB Manual Review ##############
            if category == "monologue":
                # --- Manual Review Section for Alignment ---
                st.markdown("---")
                st.subheader("Manual Review: Monologue Data")

                monologue_path = os.path.join(BASE_DIR, selected_experiment, "monologue_results", f"test_responses_{selected_param}.json")

                if os.path.exists(monologue_path):
                    with open(monologue_path, "r") as f:
                        monologue_data = json.load(f)

                    speaker_names = list(monologue_data.keys())
                    selected_speaker = st.selectbox("Select Agent", speaker_names, key="monologue_agent")

                    speaker_data = monologue_data[selected_speaker]

                    mono_idx_key = f'mono_index_{selected_param}_{selected_speaker}'
                    mono_prev_key = f'mono_prev_speaker_{selected_param}'

                    if mono_idx_key not in st.session_state:
                        st.session_state[mono_idx_key] = 0
                    if mono_prev_key not in st.session_state:
                        st.session_state[mono_prev_key] = None
                    if st.session_state[mono_prev_key] != selected_speaker:
                        st.session_state[mono_idx_key] = 0
                        st.session_state[mono_prev_key] = selected_speaker

                    def mono_go_prev():
                        st.session_state[mono_idx_key] = (st.session_state[mono_idx_key] - 1) % len(speaker_data)

                    def mono_go_next():
                        st.session_state[mono_idx_key] = (st.session_state[mono_idx_key] + 1) % len(speaker_data)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.button("⬅️ Previous", on_click=mono_go_prev, key=f"mono_prev_button_{selected_param}")
                    with col2:
                        st.markdown(f"### Example {st.session_state[mono_idx_key] + 1} of {len(speaker_data)}")
                    with col3:
                        st.button("➡️ Next", on_click=mono_go_next, key=f"mono_next_button_{selected_param}")

                    item = speaker_data[st.session_state[mono_idx_key]]

                    st.markdown("#### Question")
                    question = item.get("prompt", "").split('unknownspeaker:')[1].split('<|eot_id|>')[0]

                    st.text_area("Question", question, height=80)
                    st.markdown("#### A. Agent Response")
                    st.text_area("Agent Response", item.get("model_responses", "")[0], height=80)

                    st.markdown("#### B. LLaMA‑70B Response")
                    st.text_area("LLaMA‑70B Response", item.get("gpt_response", ""), height=80)

                    st.markdown("#### Scores")
                    
                    judgement = pd.DataFrame(item.get('gpt_judgment'))
                    filtered_df = judgement[judgement['response_idx'].isin([0, 'gpt'])]

                    def create_markdown_table(df):
                        md = "| Aspect | Score | Explanation |\n"
                        md += "|--------|-------|-------------|\n"
                        for _, row in df.iterrows():
                            md += f"| **{row['aspect']}** | {row['score']} | {row['explanation']} |\n"
                        return md
                    
                    judgement = pd.DataFrame(item.get('gpt_judgment'))
                    # Group by response_idx and display each separately
                    for idx in [0, 'gpt']:
                        if idx == 0:
                            name = "Model"
                        else:
                            name = 'LLaMA‑70B'

                        st.subheader(f"{name} Response")
                        sub_df = filtered_df[filtered_df['response_idx'] == idx]
                        st.markdown(create_markdown_table(sub_df), unsafe_allow_html=True)

                    st.markdown("#### Comparison")
                    st.write('**Winner:**', item.get("final_comparison", {}).get('winner', 'Error parsing response'))
                    st.write('**Justification:**', item.get("final_comparison", {}).get('justification', 'Error parsing response'))
                else:
                    st.warning("⚠️ Could not find the monologue data file.")



             ########### ALIGNMENT TAB Manual Review ##############
            if category == "alignment":
                # --- Manual Review Section for Alignment ---
                st.markdown("---")
                st.subheader("Manual Review: Alignment Data")

                alignment_json_map = {
                    "Memory Alignment": f"memory_results_{selected_param}.json",
                    "Belief Alignment": f"belief_results_{selected_param}.json",
                    "Personality Alignment": f"personality_results_{selected_param}.json"
                }

                alignment_choice = st.selectbox("Select Alignment Dataset", list(alignment_json_map.keys()), key="alignment_dataset")
                alignment_path = os.path.join(BASE_DIR, selected_experiment, "alignment_results", alignment_json_map[alignment_choice])
                print(alignment_path)
                if os.path.exists(alignment_path):
                    with open(alignment_path, "r") as f:
                        alignment_data = json.load(f)

                    speaker_names = list(alignment_data.keys())
                    selected_speaker = st.selectbox("Select Agent", speaker_names, key="align_agent")

                    speaker_data = alignment_data[selected_speaker]
                    idx_key = f'align_index_{alignment_choice}'
                    prev_key = f'align_prev_speaker_{alignment_choice}'

                    if idx_key not in st.session_state:
                        st.session_state[idx_key] = 0
                    if prev_key not in st.session_state:
                        st.session_state[prev_key] = None
                    if st.session_state[prev_key] != selected_speaker:
                        st.session_state[idx_key] = 0
                        st.session_state[prev_key] = selected_speaker

                    def align_go_prev():
                        st.session_state[idx_key] = (st.session_state[idx_key] - 1) % len(speaker_data)

                    def align_go_next():
                        st.session_state[idx_key] = (st.session_state[idx_key] + 1) % len(speaker_data)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.button("⬅️ Previous", on_click=align_go_prev, key=f"align_prev_button_{alignment_choice}")
                    with col2:
                        st.markdown(f"### Example {st.session_state[idx_key] + 1} of {len(speaker_data)}")
                    with col3:
                        st.button("➡️ Next", on_click=align_go_next, key=f"align_next_button_{alignment_choice}")

                    item = speaker_data[st.session_state[idx_key]]
                    st.markdown("#### Question")
                    st.text_area("Question", item.get("question", ""), height=80)
                    st.markdown("#### A. Agent Response")
                    st.text_area("Response", item.get("response", ""), height=80)
                    st.markdown("#### B. LLaMA‑70B Response")
                    st.text_area("Response", item.get("gpt_response", ""), height=80)

                    st.markdown("#### Scores")
                    st.write('**Agent response Score:** ' + str(item.get("evaluation_response", {}).get('score', '')))
                    st.write('\n**Agent response Explanation:** ' + item.get("evaluation_response", {}).get('explanation', ''))

                    st.write('\n\n**LLaMA‑70B response Score:** ' + str(item.get("evaluation_gpt_response", {}).get('score', '')))
                    st.write('\n**LLaMA‑70B response Explanation:** ' + item.get("evaluation_gpt_response", {}).get('explanation', ''))

                    st.markdown("#### Comparison")
                    st.write('**Winner:** ' + item.get("final_comparison", {}).get('winner', 'Error parsing response'))
                    st.write('\n**Justification:** ' + item.get("final_comparison", {}).get('justification', 'Error parsing response'))
                else:
                    st.warning("Could not find the selected alignment data file.")


        elif category == "Prediction Model Evals":
            # Example: show multi- and single-speaker model evals, generic/coverage only
            st.subheader("Multi-Speaker Models")
            multi_speaker_imgs = [
                os.path.join(EVALS_DIR, "multi-speaker_scores.png"),
                os.path.join(EVALS_DIR, "multi-speaker_cm.png"),
            ]
            display_carousel(multi_speaker_imgs, "multi_speaker_eval")
            st.markdown("---")
            st.subheader("Single-Speaker Models")
            single_speaker_imgs = [
                os.path.join(EVALS_DIR, "single-speaker_scores.png"),
                os.path.join(EVALS_DIR, "single-speaker_cm.png"),
            ]
            display_carousel(single_speaker_imgs, "single_speaker_eval")