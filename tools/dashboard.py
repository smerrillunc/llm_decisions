import streamlit as st
import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import re
from functools import lru_cache

st.set_page_config(page_title="LLM Decisions Dashboard", layout="wide")

# --- Configurable paths (defaults) ---
if 'config_paths' not in st.session_state:
    st.session_state['config_paths'] = {
        'alignment_results': '/playpen-ssd/smerrill/llm_decisions/alignment_results',
        'completion_results': '/playpen-ssd/smerrill/llm_decisions/completion_results',
        'monologue_results': '/playpen-ssd/smerrill/llm_decisions/monologue_results',
        'models_json': '/playpen-ssd/smerrill/llm_decisions/configs/models.json'
    }
config_paths = st.session_state['config_paths']

# --- Utility Functions ---
def compute_agent_win_rates(df):
    # Create a win column: 1 if winner is 'A', 0 if 'B'
    df = df.copy()
    df['win'] = df['winner'] == 'A'
    
    # Group by agent and calculate total games and wins
    summary = df.groupby('agent').agg(
        total_comparisons=('agent', 'count'),
        wins=('win', 'sum')
    )
    
    # Calculate win rate
    summary['win_rate'] = summary['wins'] / summary['total_comparisons']
    
    return summary.sort_values('win_rate', ascending=False)


def extract_params_from_filenames(directory):
    temperature_set = set()
    top_p_set = set()
    top_k_set = set()
    r_set = set()

    pattern = re.compile(r'_T([0-9.]+)_P([0-9.]+)_K([0-9]+)(?:_R([0-9.]+))?')

    try:
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
    except (FileNotFoundError, OSError):
        return {
            "temperature": [],
            "top_p": [],
            "top_k": [],
            "repetition_penalty": []
        }

    return {
        "temperature": sorted(temperature_set),
        "top_p": sorted(top_p_set),
        "top_k": sorted(top_k_set),
        "repetition_penalty": sorted(r_set) if r_set else None
    }

def get_dfs(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return pd.DataFrame(), pd.DataFrame()
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

@st.cache_data(show_spinner=False)
def cached_completion_plots(file_name):
    comparison_df, judgment_df = get_dfs(file_name)
    if comparison_df.empty or judgment_df.empty:
        return None, None, None, comparison_df, judgment_df
    # Win Rate Pie
    fig_pie = plot_win_rate_pie_fig(comparison_df)
    # Agent Win Rates
    fig_agent = plot_agent_win_rates_fig(comparison_df)
    # Score Distributions
    fig_score = plot_score_distributions_fig(judgment_df)
    return fig_pie, fig_agent, fig_score, comparison_df, judgment_df

@st.cache_data(show_spinner=False)
def cached_monologue_plots(file_name):
    comparison_df, judgment_df = get_dfs(file_name)
    if comparison_df.empty or judgment_df.empty:
        return None, None, None, comparison_df, judgment_df
    fig_pie = plot_win_rate_pie_fig(comparison_df)
    fig_agent = plot_agent_win_rates_fig(comparison_df)
    fig_score = plot_score_distributions_fig(judgment_df)
    return fig_pie, fig_agent, fig_score, comparison_df, judgment_df

@st.cache_data(show_spinner=False)
def cached_alignment_plots(df):
    if df.empty:
        return None
    agents = df['agent'].unique()
    ncols = 3
    nrows = (len(agents) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 8 * nrows), sharey=True)
    axes = axes.flatten()
    for i, agent in enumerate(agents):
        ax = axes[i]
        agent_data = df[df['agent'] == agent]
        sns.violinplot(
            data=agent_data,
            x='alignment',
            y='eval_score',
            ax=ax,
            palette='pastel',
            inner='box'
        )
        ax.set_title(f'Score Distribution: {agent}', fontsize=16)
        ax.set_ylim(0, 5)
        ax.set_xlabel('Alignment', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
    for j in range(len(agents), len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Helper functions to return figures only

def plot_win_rate_pie_fig(comparison_df):
    model_win, gpt_win = (comparison_df.groupby('winner').count()/len(comparison_df)).values[:, 0]
    total = model_win + gpt_win
    sizes = [model_win / total, gpt_win / total]
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
    return fig

def plot_agent_win_rates_fig(comparison_df):
    summary = compute_agent_win_rates(comparison_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=summary.index, y=summary['win_rate'], palette='viridis', ax=ax)
    ax.set_title('Win Rate by Agent')
    ax.set_xlabel('Agent')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_score_distributions_fig(judgment_df):
    df = judgment_df.copy()
    df = df[df.response_idx != 'gpt']
    agents = df['agent'].unique()
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
    ax = axes[num_agents]
    gpt_data = judgment_df[judgment_df.response_idx == 'gpt']
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
    for j in range(num_agents + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# --- Tabs ---
tabs = st.tabs(["1. Perplexity", "2. Completion Data", "3. Monologue Data", "4. Alignment Analysis", "5. Votes Analysis", "6. Config"])

# --- Tab 1: Perplexity ---
with tabs[0]:
    st.header("Perplexity")
    models_json_path = config_paths['models_json']
    try:
        if 'perplexity_loaded' not in st.session_state or not st.session_state['perplexity_loaded'] or st.session_state.get('perplexity_models_json_path') != models_json_path:
            if not os.path.exists(models_json_path):
                st.session_state['perplexity_figs'] = None
                st.session_state['perplexity_loaded'] = True
                st.session_state['perplexity_warning'] = f"Models JSON not found: {models_json_path}"
            else:
                with open(models_json_path, "r", encoding="utf-8") as f:
                    models_config = json.load(f)
                st.session_state['perplexity_models_json_path'] = models_json_path
                model_dirs = list(models_config.values())
                all_dfs = []
                for path in model_dirs + ['/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/baseline']:
                    csv_path = os.path.join(path.replace('/merged',''), "perplexity_results.csv")
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            df["model_path"] = path
                            all_dfs.append(df)
                        except Exception:
                            continue
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    combined_df.model = combined_df.model.apply(lambda x: x.split("/")[-1])
                    combined_df.sort_values('perplexity', inplace=True)
                    ppl_matrix = combined_df.pivot_table(
                        index="dataset",
                        columns="model",
                        values="perplexity"
                    )
                    baseline_col = 'baseline'
                    if baseline_col in ppl_matrix.columns:
                        baseline_ppl = ppl_matrix[baseline_col]
                        other_ppl = ppl_matrix.drop(columns=[baseline_col])
                        other_ppl = other_ppl.sort_index()
                        baseline_ppl = baseline_ppl.sort_index()
                        other_ppl.columns = other_ppl.columns.astype(str)
                        other_ppl = other_ppl[sorted(other_ppl.columns, key=lambda x: x.lower())]
                        normalized_ppl = other_ppl.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        sns.heatmap(normalized_ppl, annot=other_ppl.round(2), fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Normalized Perplexity'}, ax=ax1)
                        ax1.set_title("Perplexity of Each Model (Excl. Baseline)")
                        ax1.set_ylabel("Dataset")
                        ax1.set_xlabel("Model")
                        plt.tight_layout()
                        fig2, ax2 = plt.subplots(figsize=(10, 1.5))
                        sns.heatmap(baseline_ppl.to_frame().T, annot=baseline_ppl.to_frame().T.round(2), fmt=".2f",
                                    cmap="Blues", cbar_kws={'label': 'Baseline Perplexity'}, ax=ax2)
                        ax2.set_title("Baseline Perplexity")
                        ax2.set_ylabel("")
                        ax2.set_xlabel("Dataset")
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        st.session_state['perplexity_figs'] = (fig1, fig2)
                        st.session_state['perplexity_loaded'] = True
                        st.session_state['perplexity_warning'] = None
                    else:
                        st.session_state['perplexity_figs'] = None
                        st.session_state['perplexity_loaded'] = True
                        st.session_state['perplexity_warning'] = "No baseline column found in perplexity results."
                else:
                    st.session_state['perplexity_figs'] = None
                    st.session_state['perplexity_loaded'] = True
                    st.session_state['perplexity_warning'] = "No perplexity results found."
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        st.session_state['perplexity_figs'] = None
        st.session_state['perplexity_loaded'] = True
        st.session_state['perplexity_warning'] = f"Could not load models config or perplexity results from {models_json_path}."
    # Display cached figures or warning
    if st.session_state.get('perplexity_figs'):
        fig1, fig2 = st.session_state['perplexity_figs']
        st.pyplot(fig1)
        st.pyplot(fig2)
    elif st.session_state.get('perplexity_warning'):
        st.warning(st.session_state['perplexity_warning'])

# --- Tab 2: Completion Data ---
with tabs[1]:
    st.header("Completion Data")
    completion_dir = config_paths['completion_results']
    param_dict = extract_params_from_filenames(completion_dir)
    if not param_dict['temperature'] or not param_dict['top_p'] or not param_dict['top_k'] or not param_dict['repetition_penalty']:
        st.warning(f"No valid completion data found in directory: {completion_dir}")
    else:
        temperature = st.selectbox("Temperature", param_dict['temperature'])
        top_p = st.selectbox("Top-p", param_dict['top_p'])
        top_k = st.selectbox("Top-k", param_dict['top_k'])
        r = st.selectbox("Repetition Penalty", param_dict['repetition_penalty'])
        file_name = f'{completion_dir}/test_responses_T{temperature}_P{top_p}_K{top_k}_R{r}.json'
        if 'show_comp_plots' not in st.session_state:
            st.session_state['show_comp_plots'] = True
        show_plots = st.checkbox("Show Plots", value=st.session_state['show_comp_plots'], key="comp_show_plots")
        st.session_state['show_comp_plots'] = show_plots
        plot_key = f"comp_plot_{temperature}_{top_p}_{top_k}_{r}"
        # Only load/cache when tab is active
        if plot_key not in st.session_state:
            if os.path.exists(file_name):
                fig_pie, fig_agent, fig_score, comparison_df, judgment_df = cached_completion_plots(file_name)
                st.session_state[plot_key] = {
                    "fig_pie": fig_pie,
                    "fig_agent": fig_agent,
                    "fig_score": fig_score,
                    "comparison_df": comparison_df,
                    "judgment_df": judgment_df,
                    "file_name": file_name
                }
            else:
                st.session_state[plot_key] = None
        plot_data = st.session_state.get(plot_key)
        plot_container = st.container()
        with plot_container:
            if show_plots:
                if plot_data and plot_data["fig_pie"]:
                    st.subheader("Win Rate Pie Chart")
                    st.pyplot(plot_data["fig_pie"])
                    st.subheader("Agent Win Rates")
                    st.pyplot(plot_data["fig_agent"])
                    st.subheader("Score Distributions")
                    st.pyplot(plot_data["fig_score"])
                else:
                    st.warning(f"No completion data found for selected parameters: {file_name}")
        review_container = st.container()
        with review_container:
            st.markdown("---")
            st.subheader("Manual Review: Completion Data")
            if plot_data and plot_data["file_name"] and os.path.exists(plot_data["file_name"]):
                try:
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    st.warning(f"Could not load completion data for manual review: {file_name}")
                    data = None
                if data:
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
                        st.button("Previous", on_click=comp_go_prev, key="comp_prev_button", use_container_width=True)
                    with col2:
                        st.markdown(f"### Prompt {st.session_state.comp_index + 1} of {len(speaker_data)}")
                    with col3:
                        st.button("Next", on_click=comp_go_next, key="comp_next_button", use_container_width=True)
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
    monologue_dir = config_paths['monologue_results']
    param_dict = extract_params_from_filenames(monologue_dir)
    if not param_dict['temperature'] or not param_dict['top_p'] or not param_dict['top_k'] or not param_dict['repetition_penalty']:
        st.warning(f"No valid monologue data found in directory: {monologue_dir}")
    else:
        temperature = st.selectbox("Temperature", param_dict['temperature'], key="mono_temp")
        top_p = st.selectbox("Top-p", param_dict['top_p'], key="mono_top_p")
        top_k = st.selectbox("Top-k", param_dict['top_k'], key="mono_top_k")
        r = st.selectbox("Repetition Penalty", param_dict['repetition_penalty'], key="mono_r")
        file_name = f'{monologue_dir}/test_responses_T{temperature}_P{top_p}_K{top_k}_R{r}.json'
        if 'show_mono_plots' not in st.session_state:
            st.session_state['show_mono_plots'] = True
        show_plots = st.checkbox("Show Plots", value=st.session_state['show_mono_plots'], key="mono_show_plots")
        st.session_state['show_mono_plots'] = show_plots
        cache_key = f"mono_plots_{temperature}_{top_p}_{top_k}_{r}"
        # Only load/cache when tab is active
        if cache_key not in st.session_state:
            if os.path.exists(file_name):
                figs = cached_monologue_plots(file_name)
                st.session_state[cache_key] = figs
            else:
                st.session_state[cache_key] = None
        figs = st.session_state.get(cache_key)
        plot_container = st.container()
        review_container = st.container()
        if os.path.exists(file_name) and figs and figs[0]:
            fig_pie, fig_agent, fig_score, comparison_df, judgment_df = figs
            with plot_container:
                if show_plots:
                    st.subheader("Win Rate Pie Chart")
                    st.pyplot(fig_pie)
                    st.subheader("Agent Win Rates")
                    st.pyplot(fig_agent)
                    st.subheader("Score Distributions")
                    st.pyplot(fig_score)
            # --- Manual Review ---
            with review_container:
                st.markdown("---")
                st.subheader("Manual Review: Monologue Data")
                try:
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    st.warning(f"Could not load monologue data for manual review: {file_name}")
                    data = None
                if data:
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
                        st.button("Previous", on_click=mono_go_prev, key="mono_prev_button", use_container_width=True)
                    with col2:
                        st.markdown(f"### Prompt {st.session_state.mono_index + 1} of {len(speaker_data)}")
                    with col3:
                        st.button("Next", on_click=mono_go_next, key="mono_next_button", use_container_width=True)
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

# --- Tab 4: Alignment Analysis ---
with tabs[3]:
    st.header("Alignment Analysis")
    alignment_dir = config_paths['alignment_results']
    param_dict = extract_params_from_filenames(alignment_dir)
    if not param_dict['temperature'] or not param_dict['top_p'] or not param_dict['top_k'] or not param_dict['repetition_penalty']:
        st.warning(f"No valid alignment data found in directory: {alignment_dir}")
        dfs = []
        alignment_data = {}
    else:
        temperature = st.selectbox("Temperature", param_dict['temperature'], key="align_temp")
        top_p = st.selectbox("Top-p", param_dict['top_p'], key="align_top_p")
        top_k = st.selectbox("Top-k", param_dict['top_k'], key="align_top_k")
        r = st.selectbox("Repetition Penalty", param_dict['repetition_penalty'], key="align_r")
        alignments = ['belief', 'memory', 'personality']
        dfs = []
        alignment_data = {}
        for alignment in alignments:
            file_name = f'{alignment_dir}/{alignment}_results_T{temperature}_P{top_p}_K{top_k}_R{r}.json'
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                    alignment_data[alignment] = data
                    for agent in data.keys():
                        tmp = pd.DataFrame(data[agent])
                        if 'evaluation' in tmp.columns:
                            eval_expanded = tmp['evaluation'].apply(pd.Series)
                            eval_expanded = eval_expanded.rename(columns={
                                'score': 'eval_score',
                                'explanation': 'eval_explanation',
                                'justification': 'eval_justification'
                            })
                            tmp = pd.concat([tmp.drop(columns=['evaluation']), eval_expanded], axis=1)
                        tmp['agent'] = agent
                        tmp['alignment'] = alignment
                        dfs.append(tmp)
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    continue
    if 'show_align_plots' not in st.session_state:
        st.session_state['show_align_plots'] = True
    show_plots = st.checkbox("Show Plots", value=st.session_state['show_align_plots'], key="align_show_plots")
    st.session_state['show_align_plots'] = show_plots
    cache_key = f"align_plots_{param_dict['temperature']}_{param_dict['top_p']}_{param_dict['top_k']}_{param_dict['repetition_penalty']}"
    # Only load/cache when tab is active
    if dfs and cache_key not in st.session_state:
        df = pd.concat(dfs, ignore_index=True)
        df = df[df.eval_score.isin(range(1, 6))]
        fig = cached_alignment_plots(df)
        st.session_state[cache_key] = fig
    fig = st.session_state.get(cache_key)
    plot_container = st.container()
    review_container = st.container()
    if dfs and fig:
        with plot_container:
            if show_plots:
                st.subheader("Score Distributions by Agent and Alignment")
                st.pyplot(fig)
        # --- Manual Review ---
        with review_container:
            st.markdown("---")
            st.subheader("Manual Review: Alignment Data")
            alignment_select = st.selectbox("Select Alignment Type", alignments, key="align_type")
            if alignment_select in alignment_data:
                data = alignment_data[alignment_select]
                speaker_names = list(data.keys())
                selected_speaker = st.selectbox("Select Agent", speaker_names, key="align_agent")
                speaker_data = data[selected_speaker]
                idx_key = f'align_index_{alignment_select}'
                prev_key = f'align_prev_speaker_{alignment_select}'
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
                    st.button("Previous", on_click=align_go_prev, key=f"align_prev_button_{alignment_select}")
                with col2:
                    st.markdown(f"### Example {st.session_state[idx_key] + 1} of {len(speaker_data)}")
                with col3:
                    st.button("Next", on_click=align_go_next, key=f"align_next_button_{alignment_select}")
                item = speaker_data[st.session_state[idx_key]]
                st.markdown("#### Input Chunk")
                st.text_area("Chunk", item.get("chunk", ""), height=100)
                st.markdown("#### Question")
                st.text_area("Question", item.get("question", ""), height=80)
                st.markdown("#### Agent Response")
                st.text_area("Response", item.get("response", ""), height=80)
                evaluation = item.get("evaluation", {})
                if evaluation:
                    st.markdown("#### GPT Score")
                    st.write(evaluation.get("score", ""))
                    st.markdown("#### GPT Explanation")
                    st.write(evaluation.get("explanation", ""))
                else:
                    st.markdown("#### GPT Score")
                    st.write("")
                    st.markdown("#### GPT Explanation")
                    st.write("")
                st.markdown("---")
    else:
        st.warning("No alignment data found for selected parameters.")

# --- Tab 5: Votes Analysis ---
with tabs[4]:
    st.header("Votes Analysis")
    votes_file = os.path.join(config_paths['alignment_results'], 'votes.json')
    if 'show_votes_plots' not in st.session_state:
        st.session_state['show_votes_plots'] = True
    show_plots = st.checkbox("Show Plots", value=st.session_state['show_votes_plots'], key="votes_show_plots")
    st.session_state['show_votes_plots'] = show_plots
    votes_cache_key = 'votes_plots'
    # Only load/cache when tab is active
    if votes_cache_key not in st.session_state:
        import pandas as pd
        import matplotlib.ticker as mtick
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json
        from sklearn.metrics import confusion_matrix
        if os.path.exists(votes_file):
            try:
                with open(votes_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame()
                for agent in data.keys():
                    tmp = pd.DataFrame(data[agent])
                    tmp['agent'] = agent
                    df = pd.concat([df, tmp], ignore_index=True)
                # Overall accuracy pie
                df_acc = df.groupby('agent').aggregate({'correct':'mean','true_vote':'count'}).reset_index()
                total_votes = df_acc['true_vote'].sum()
                total_correct_votes = (df_acc['correct'] * df_acc['true_vote']).sum()
                total_incorrect_votes = total_votes - total_correct_votes
                sizes = [total_correct_votes, total_incorrect_votes]
                labels = ['Correct', 'Incorrect']
                colors = ['#4CAF50', '#007ACC']
                explode = (0.07, 0.07)
                fig_pie, ax_pie = plt.subplots(figsize=(7.5, 6), dpi=120)
                wedges, texts, autotexts = ax_pie.pie(
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
                fig_pie.gca().add_artist(centre_circle)
                ax_pie.set_title("Overall Correct vs Incorrect Votes (All Agents)", fontsize=18, weight='bold', pad=30)
                ax_pie.axis('equal')
                plt.subplots_adjust(top=0.88, bottom=0.1)
                # Agent accuracy bar plot
                fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
                sns.barplot(
                    data=df_acc,
                    x='agent',
                    y='correct',
                    color='skyblue',
                    edgecolor='black',
                    ax=ax_bar
                )
                for i, row in df_acc.iterrows():
                    ax_bar.text(
                        i, row['correct'] / 2,
                        f"{int(row['true_vote'])} votes",
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black'
                    )
                ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                plt.ylim(0, 1.05)
                plt.xticks(rotation=45, ha='right', fontsize=11)
                plt.yticks(fontsize=11)
                plt.ylabel("Correct (%)", fontsize=12)
                plt.xlabel("Agent", fontsize=12)
                plt.title("Accuracy of Agent Votes (Total Votes Made)", fontsize=14, fontweight='bold')
                sns.despine()
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
                # Confusion matrix
                true_labels = df['true_vote']
                pred_labels = df['pred_vote']
                classes = sorted(pd.concat([true_labels, pred_labels]).unique())
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(true_labels, pred_labels, labels=classes)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Count'}, ax=ax_cm
                )
                ax_cm.set_xlabel('Predicted Vote', fontsize=12, fontweight='bold')
                ax_cm.set_ylabel('True Vote', fontsize=12, fontweight='bold')
                ax_cm.set_title('Confusion Matrix: True Vote vs Predicted Vote', fontsize=16, fontweight='bold')
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11, rotation=0)
                plt.tight_layout()
                st.session_state[votes_cache_key] = (fig_pie, fig_bar, fig_cm)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                st.session_state[votes_cache_key] = None
        else:
            st.session_state[votes_cache_key] = None
    figs = st.session_state.get(votes_cache_key)
    plot_container = st.container()
    with plot_container:
        if show_plots and figs and figs[0]:
            fig_pie, fig_bar, fig_cm = figs
            st.subheader("Overall Correct vs Incorrect Votes (All Agents)")
            st.pyplot(fig_pie)
            st.subheader("Accuracy of Agent Votes (Total Votes Made)")
            st.pyplot(fig_bar)
            st.subheader("Confusion Matrix: True Vote vs Predicted Vote")
            st.pyplot(fig_cm)
        elif not figs:
            st.warning(f"No votes data found at {votes_file}")

# --- Tab 6: Config ---
with tabs[5]:
    st.header("Dashboard Configuration")
    st.markdown("Select a results date to automatically configure all data source paths, or manually override below.")
    results_root = '/playpen-ssd/smerrill/llm_decisions/results'
    try:
        date_folders = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
        date_folders = sorted(date_folders, reverse=True)
    except (FileNotFoundError, OSError):
        date_folders = []
    # Set default config_paths to first available date if not already set to a results date
    if date_folders:
        first_date = date_folders[0]
        auto_alignment_results = os.path.join(results_root, first_date, 'alignment_results')
        auto_completion_results = os.path.join(results_root, first_date, 'completion_results')
        auto_monologue_results = os.path.join(results_root, first_date, 'monologue_results')
        auto_models_json = os.path.join(results_root, first_date, 'models.json')
        # Only update if config_paths is still at the hardcoded default
        default_paths = {
            'alignment_results': '/playpen-ssd/smerrill/llm_decisions/alignment_results',
            'completion_results': '/playpen-ssd/smerrill/llm_decisions/completion_results',
            'monologue_results': '/playpen-ssd/smerrill/llm_decisions/monologue_results',
            'models_json': '/playpen-ssd/smerrill/llm_decisions/configs/models.json'
        }
        if st.session_state['config_paths'] == default_paths:
            st.session_state['config_paths'] = {
                'alignment_results': auto_alignment_results,
                'completion_results': auto_completion_results,
                'monologue_results': auto_monologue_results,
                'models_json': auto_models_json
            }
        selected_date = st.selectbox("Select Results Date", date_folders, index=0)
        auto_alignment_results = os.path.join(results_root, selected_date, 'alignment_results')
        auto_completion_results = os.path.join(results_root, selected_date, 'completion_results')
        auto_monologue_results = os.path.join(results_root, selected_date, 'monologue_results')
        auto_models_json = os.path.join(results_root, selected_date, 'models.json')
        st.markdown(f"**Auto-configured paths for {selected_date}:**")
        st.text(f"Alignment Results: {auto_alignment_results}")
        st.text(f"Completion Results: {auto_completion_results}")
        st.text(f"Monologue Results: {auto_monologue_results}")
        st.text(f"Models JSON: {auto_models_json}")
        if st.button("Use Selected Date Paths", key="use_date_paths"):
            st.session_state['config_paths'] = {
                'alignment_results': auto_alignment_results,
                'completion_results': auto_completion_results,
                'monologue_results': auto_monologue_results,
                'models_json': auto_models_json
            }
            st.success(f"Paths updated to {selected_date}! Switch tabs to reload data.")
    else:
        st.warning(f"No results folders found in {results_root}")
    st.markdown("---")
    st.markdown("**Models in Current Config:**")
    models_json_path = st.session_state['config_paths']['models_json']
    import pandas as pd
    import re
    try:
        with open(models_json_path, 'r') as f:
            models_config = json.load(f)
        rows = []
        for agent, path in models_config.items():
            # Extract the config string from the path
            match = re.search(r'/([^/]+?)(_\d+_\d*\.?\d*_\d*e-\d+_\d+)?/merged$', path)
            if match:
                config_str = match.group(1) + (match.group(2) if match.group(2) else '')
            else:
                config_str = agent
            parts = config_str.split('_')
            # Defaults
            lora_factors = parts[1] if len(parts) > 1 else ''
            lora_dropout = parts[2] if len(parts) > 2 else ''
            learning_rate = parts[3] if len(parts) > 3 else '1e-5'
            train_epochs = parts[4] if len(parts) > 4 else '2'
            rows.append({
                'Agent': agent,
                'LORA_FACTORS': lora_factors,
                'LORA_DROPOUT': lora_dropout,
                'LEARNING_RATE': learning_rate,
                'TRAIN_EPOCHS': train_epochs,
                'Model Path': path
            })
        df_models = pd.DataFrame(rows)
        st.dataframe(df_models, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load models config from {models_json_path}: {e}")
    st.markdown("---")
    st.markdown("**Manual override:**")
    alignment_results = st.text_input("Alignment Results Directory", value=config_paths['alignment_results'])
    completion_results = st.text_input("Completion Results Directory", value=config_paths['completion_results'])
    monologue_results = st.text_input("Monologue Results Directory", value=config_paths['monologue_results'])
    models_json = st.text_input("Models JSON Path", value=config_paths['models_json'])
    if st.button("Update Paths", key="update_config_paths"):
        st.session_state['config_paths'] = {
            'alignment_results': alignment_results,
            'completion_results': completion_results,
            'monologue_results': monologue_results,
            'models_json': models_json
        }
        st.success("Paths updated! Switch tabs to reload data.")