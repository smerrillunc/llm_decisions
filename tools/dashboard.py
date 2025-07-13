import streamlit as st
import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="LLM Decisions Dashboard", layout="wide")

# --- Utility Functions ---
def extract_params_from_filenames(directory):
    temperature_set = set()
    top_p_set = set()
    top_k_set = set()
    r_set = set()

    pattern = re.compile(r'_T([0-9.]+)_P([0-9.]+)_K([0-9]+)(?:_R([0-9.]+))?')

    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        match = pattern.search(fname)
        if match:
            temperature_set.add(float(match.group(1)))
            top_p_set.add(float(match.group(2)))
            top_k_set.add(int(match.group(3)))
            r_val = match.group(4)
            if r_val:
                r_val = r_val.rstrip('.')
                try:
                    r_set.add(float(r_val))
                except ValueError:
                    pass

    return {
        "temperature": sorted(temperature_set),
        "top_p": sorted(top_p_set),
        "top_k": sorted(top_k_set),
        "repetition_penalty": sorted(r_set) if r_set else None
    }

def get_dfs(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    comparison_df = pd.DataFrame()
    judgment_df = pd.DataFrame()
    for agent in data.keys():
        tmp = pd.DataFrame()
        tmp2 = pd.DataFrame()
        for i, entry in enumerate(data[agent]):
            try:
                tmp = pd.concat([tmp,  pd.DataFrame([data[agent][i]['final_comparison']])])
                tmp3 = pd.DataFrame(data[agent][i]['gpt_judgment'])
                tmp3['example_idx'] = i
                tmp2 = pd.concat([tmp2, tmp3])
            except:
                continue
        tmp['agent'] = agent
        tmp2['agent'] = agent
        comparison_df = pd.concat([comparison_df, tmp], ignore_index=True)    
        judgment_df = pd.concat([judgment_df, tmp2], ignore_index=True)
    comparison_df = comparison_df[comparison_df.winner.isin(['A', 'B'])]
    judgment_df = judgment_df[judgment_df.score.isin(range(1, 6))]
    return comparison_df, judgment_df

def plot_win_rate_pie(finetuned_win_rate, gpt_win_rate):
    total = finetuned_win_rate + gpt_win_rate
    sizes = [finetuned_win_rate / total, gpt_win_rate / total]
    labels = ['Fine-tuned Model', 'GPT']
    colors = ['#4CAF50', '#007ACC']
    explode = (0.07, 0.07)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=120)
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=120,
        pctdistance=0.8,
        shadow=False,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_color("white")
        autotext.set_weight("bold")
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title("Win Rate vs. GPT", fontsize=18, weight='bold', pad=30)
    ax.axis('equal')
    plt.subplots_adjust(top=0.88, bottom=0.1)
    st.pyplot(fig)

def compute_agent_win_rates(df):
    df = df.copy()
    df['win'] = df['winner'] == 'A'
    summary = df.groupby('agent').agg(
        total_comparisons=('agent', 'count'),
        wins=('win', 'sum')
    )
    summary['win_rate'] = summary['wins'] / summary['total_comparisons']
    return summary.sort_values('win_rate', ascending=False)

def plot_agent_win_rates(df):
    summary = compute_agent_win_rates(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=summary.index, y=summary['win_rate'], palette='viridis', ax=ax)
    ax.set_title('Win Rate by Agent')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def analyze_judgement_df(judgment_df):
    df = judgment_df.copy()
    df = df[df.response_idx != 'gpt']
    deviation_df = (
        df.groupby(['agent', 'example_idx', 'aspect'])['score']
        .std()
        .reset_index(name='std_dev')
    )
    avg_deviation = (
        deviation_df.groupby(['agent', 'aspect'])['std_dev']
        .mean()
        .reset_index(name='avg_std_dev')
    )
    avg_score_df = (
        df.groupby(['agent', 'aspect'])['score']
        .mean()
        .reset_index(name='avg_score')
    )
    return avg_score_df, avg_deviation

def plot_score_distributions(judgement_df):
    df = judgement_df.copy()
    df = df[df.response_idx != 'gpt']
    agents = df['agent'].unique()
    aspects = df['aspect'].unique()
    num_agents = len(agents)
    ncols = 3
    nrows = (num_agents + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 8 * nrows), sharey=True)
    axes = axes.flatten()
    for i, agent in enumerate(agents):
        ax = axes[i]
        agent_data = df[df['agent'] == agent]
        sns.violinplot(
            data=agent_data,
            x='aspect',
            y='score',
            ax=ax,
            palette='pastel'
        )
        ax.set_title(f'Score Distribution: {agent}', fontsize=16)
        ax.set_ylim(0, 6)
        ax.set_xlabel('Aspect', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
    # GPT plot in last slot
    ax = axes[num_agents]
    gpt_data = judgement_df[judgement_df.response_idx == 'gpt']
    sns.violinplot(
        data=gpt_data,
        x='aspect',
        y='score',
        ax=ax,
        palette='pastel'
    )
    ax.set_title(f'Score Distribution: GPT', fontsize=16)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Aspect', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    # Hide unused axes
    for j in range(num_agents + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

# --- Tabs ---
tabs = st.tabs(["1. Perplexity", "2. Completion Data", "3. Monologue Data"])

# --- Tab 1: Perplexity ---
with tabs[0]:
    st.header("Perplexity")
    # Load models config
    with open("/playpen-ssd/smerrill/llm_decisions/configs/models.json", "r", encoding="utf-8") as f:
        models_config = json.load(f)
    model_dirs = models_config.values()
    all_dfs = []
    for path in model_dirs:
        csv_path = os.path.join(path.replace('/merged',''), "perplexity_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["model_path"] = path
            all_dfs.append(df)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.model = combined_df.model.apply(lambda x: x.split("/")[-1])
        combined_df.sort_values('perplexity', inplace=True)
        combined_df = combined_df[combined_df.dataset != "davidoberg"]
        ppl_matrix = combined_df.pivot_table(
            index="dataset",
            columns="model_name",
            values="perplexity"
        )
        normalized_ppl = ppl_matrix.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(normalized_ppl, annot=ppl_matrix.round(2), fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Normalized Perplexity'}, ax=ax)
        ax.set_title("Perplexity of Each Model on Each Dataset")
        ax.set_ylabel("Dataset")
        ax.set_xlabel("Model")
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(ppl_matrix)
    else:
        st.warning("No perplexity results found.")

# --- Tab 2: Completion Data ---
with tabs[1]:
    st.header("Completion Data")
    completion_dir = '/playpen-ssd/smerrill/llm_decisions/completion_results'
    param_dict = extract_params_from_filenames(completion_dir)
    temperature = st.selectbox("Temperature", param_dict['temperature'])
    top_p = st.selectbox("Top-p", param_dict['top_p'])
    top_k = st.selectbox("Top-k", param_dict['top_k'])
    r = st.selectbox("Repetition Penalty", param_dict['repetition_penalty'])
    file_name = f'{completion_dir}/test_responses_T{temperature}_P{top_p}_K{top_k}_R{r}.json'
    if os.path.exists(file_name):
        comparison_df, judgment_df = get_dfs(file_name)
        st.subheader("Win Rate Pie Chart")
        model_win, gpt_win = (comparison_df.groupby('winner').count()/len(comparison_df)).values[:, 0]
        plot_win_rate_pie(model_win, gpt_win)
        st.subheader("Agent Win Rates")
        plot_agent_win_rates(comparison_df)
        st.subheader("Score Distributions")
        plot_score_distributions(judgment_df)
        # --- Detailed Analysis ---
        st.markdown("---")
        st.subheader("Manual Review: Completion Data")
        with open(file_name, 'r') as f:
            data = json.load(f)
        speaker_names = list(data.keys())
        selected_speaker = st.selectbox("Select a speaker for review", speaker_names, key="comp_speaker")
        speaker_data = data[selected_speaker]
        if 'comp_index' not in st.session_state:
            st.session_state.comp_index = 0
        if 'comp_prev_speaker' not in st.session_state:
            st.session_state.comp_prev_speaker = None
        if st.session_state.comp_prev_speaker != selected_speaker:
            st.session_state.comp_index = 0
            st.session_state.comp_prev_speaker = selected_speaker
        def comp_go_prev():
            st.session_state.comp_index = (st.session_state.comp_index - 1) % len(speaker_data)
        def comp_go_next():
            st.session_state.comp_index = (st.session_state.comp_index + 1) % len(speaker_data)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("Previous", on_click=comp_go_prev, key="comp_prev_button")
        with col2:
            st.markdown(f"### Prompt {st.session_state.comp_index + 1} of {len(speaker_data)}")
        with col3:
            st.button("Next", on_click=comp_go_next, key="comp_next_button")
        item = speaker_data[st.session_state.comp_index]
        st.markdown("#### Prompt")
        st.text_area("Prompt", item.get("prompt", ""), height=150)
        st.markdown("#### Reference Completion")
        st.text_area("True Completion", item.get("true_completion", ""), height=100)
        st.markdown("#### Model Responses")
        responses = item.get("model_responses", [])
        for i, r in enumerate(responses):
            st.text_area(f"Response {i+1}", r, height=100, key=f"comp_response_{selected_speaker}_{i}_{st.session_state.comp_index}")
        gpt_response = item.get("gpt_response")
        if gpt_response:
            st.markdown("#### GPT Response")
            st.text_area("GPT Response", gpt_response, height=80, key=f"comp_gpt_response_{st.session_state.comp_index}")
        final_comparison = item.get("final_comparison", {})
        if final_comparison:
            st.markdown("#### Final Comparison")
            st.write(f"**Winner:** Response {final_comparison.get('winner', '')}")
            st.markdown("**Justification:**")
            st.write(final_comparison.get("justification", ""))
        st.markdown("---")
    else:
        st.warning(f"No completion data found for selected parameters: {file_name}")

# --- Tab 3: Monologue Data ---
with tabs[2]:
    st.header("Monologue Data")
    monologue_dir = '/playpen-ssd/smerrill/llm_decisions/monologue_results copy'
    param_dict = extract_params_from_filenames(monologue_dir)
    temperature = st.selectbox("Temperature", param_dict['temperature'], key="mono_temp")
    top_p = st.selectbox("Top-p", param_dict['top_p'], key="mono_top_p")
    top_k = st.selectbox("Top-k", param_dict['top_k'], key="mono_top_k")
    r = st.selectbox("Repetition Penalty", param_dict['repetition_penalty'], key="mono_r")
    file_name = f'{monologue_dir}/test_responses_T{temperature}_P{top_p}_K{top_k}_R{r}.json'
    if os.path.exists(file_name):
        comparison_df, judgment_df = get_dfs(file_name)
        st.subheader("Win Rate Pie Chart")
        model_win, gpt_win = (comparison_df.groupby('winner').count()/len(comparison_df)).values[:, 0]
        plot_win_rate_pie(model_win, gpt_win)
        st.subheader("Agent Win Rates")
        plot_agent_win_rates(comparison_df)
        st.subheader("Score Distributions")
        plot_score_distributions(judgment_df)
        # --- Detailed Analysis ---
        st.markdown("---")
        st.subheader("Manual Review: Monologue Data")
        with open(file_name, 'r') as f:
            data = json.load(f)
        speaker_names = list(data.keys())
        selected_speaker = st.selectbox("Select a speaker for review", speaker_names, key="mono_speaker")
        speaker_data = data[selected_speaker]
        if 'mono_index' not in st.session_state:
            st.session_state.mono_index = 0
        if 'mono_prev_speaker' not in st.session_state:
            st.session_state.mono_prev_speaker = None
        if st.session_state.mono_prev_speaker != selected_speaker:
            st.session_state.mono_index = 0
            st.session_state.mono_prev_speaker = selected_speaker
        def mono_go_prev():
            st.session_state.mono_index = (st.session_state.mono_index - 1) % len(speaker_data)
        def mono_go_next():
            st.session_state.mono_index = (st.session_state.mono_index + 1) % len(speaker_data)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("Previous", on_click=mono_go_prev, key="mono_prev_button")
        with col2:
            st.markdown(f"### Prompt {st.session_state.mono_index + 1} of {len(speaker_data)}")
        with col3:
            st.button("Next", on_click=mono_go_next, key="mono_next_button")
        item = speaker_data[st.session_state.mono_index]
        st.markdown("#### Prompt")
        st.text_area("Prompt", item.get("prompt", ""), height=150)
        st.markdown("#### True Monologue")
        st.text_area("True Monologue", item.get("monologue", ""), height=100)
        st.markdown("#### Model Responses")
        responses = item.get("model_responses", [])
        for i, r in enumerate(responses):
            st.text_area(f"Response {i+1}", r, height=100, key=f"mono_response_{selected_speaker}_{i}_{st.session_state.mono_index}")
        gpt_response = item.get("gpt_response")
        if gpt_response:
            st.markdown("#### GPT Response")
            st.text_area("GPT Response", gpt_response, height=80, key=f"mono_gpt_response_{st.session_state.mono_index}")
        final_comparison = item.get("final_comparison", {})
        if final_comparison:
            st.markdown("#### Final Comparison")
            st.write(f"**Winner:** Response {final_comparison.get('winner', '')}")
            st.markdown("**Justification:**")
            st.write(final_comparison.get("justification", ""))
        st.markdown("---")
    else:
        st.warning(f"No monologue data found for selected parameters: {file_name}")