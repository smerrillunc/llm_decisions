import argparse
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Generate all figures from model_performance notebook.")
    parser.add_argument('--results_path', type=str, required=True, help='Path to results directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output figures')
    return parser.parse_args()

def compute_agent_win_rates(df):
    df = df.copy()
    df = df[df.winner.isin(['A', 'B'])]
    df['win'] = df['winner'] == 'A'
    summary = df.groupby('agent_name').agg(total_comparisons=('agent_name', 'count'), wins=('win', 'sum'))
    summary['win_rate'] = summary['wins'] / summary['total_comparisons']
    return summary.sort_values('win_rate', ascending=False)

def plot_win_rate_pie(finetuned_win_rate, gpt_win_rate, save_path=None):
    total = finetuned_win_rate + gpt_win_rate
    if total == 0:
        return
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
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_agent_win_rates(df, save_path=None):
    import matplotlib.ticker as mtick

    summary = compute_agent_win_rates(df).reset_index()

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=summary, x='agent_name', y='win_rate', palette='viridis', edgecolor='black')

    # Add total comparison count as labels inside bars
    for i, row in summary.iterrows():
        ax.text(i, row['win_rate'] / 2, f"{row['total_comparisons']} comps", 
                ha='center', va='center', fontsize=11, fontweight='bold', color='black')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.xlabel("Agent", fontsize=12)
    plt.title("Agent Win Rates (Total Comparisons Shown)", fontsize=14, fontweight='bold')
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_score_distributions(judgment_df, output_dir):
    df = judgment_df.copy()
    agents = df['agent'].unique()
    for agent in agents:
        agent_data = df[(df['agent'] == agent) & (df['response_idx'] != 'gpt')]
        if agent_data.empty:
            continue
        plt.figure(figsize=(10, 8))
        sns.violinplot(data=agent_data, x='aspect', y='score', palette='pastel')
        plt.title(f'Score Distribution: {agent}', fontsize=24)
        plt.ylim(0, 6)
        plt.xlabel('Aspect', fontsize=18)
        plt.ylabel('Score', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'score_distribution_{agent}.png'))
        plt.close()
    gpt_data = df[df['response_idx'] == 'gpt']
    if not gpt_data.empty:
        plt.figure(figsize=(10, 8))
        sns.violinplot(data=gpt_data, x='aspect', y='score', palette='pastel')
        plt.title('Score Distribution: GPT', fontsize=24)
        plt.ylim(0, 6)
        plt.xlabel('Aspect', fontsize=18)
        plt.ylabel('Score', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'score_distribution_GPT.png'))
        plt.close()




def load_data(completion_file, monologue_file, belief_file, memory_file, personality_file):
    # Completion DF
    with open(completion_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    completion_df = pd.DataFrame()
    for agent in data.keys():
        for i in range(len(data[agent])):
            model_responses = data[agent][i].get('model_responses', np.nan)
            best_selection = data[agent][i].get('best_selection', None)
            if best_selection:
                model_response = model_responses[best_selection['best_index']-1]
            else:
                model_response = model_responses[random.randint(0, len(model_responses)-1)]
            gpt_response = data[agent][i].get('gpt_response', np.nan)
            gpt_judgment = data[agent][i].get('gpt_judgment', [])
            final_comparison = data[agent][i].get('final_comparison', {})
            winner = final_comparison.get('winner', np.nan)
            justification = final_comparison.get('justification', np.nan)
            tmp = {'resopnse_idx':i,
                   'agent_name':agent,
                   'model_response':model_response,
                   'model_responses':model_responses,
                   'gpt_response':gpt_response,
                   'gpt_judgment':gpt_judgment,
                   'winner':winner,
                   'justification':justification,
                   'true_completion':data[agent][i].get('true_completion', np.nan),
                   }
            completion_df = pd.concat([completion_df, pd.DataFrame([tmp])])
    completion_df = completion_df.reset_index(drop=True)

    # Monologue DF
    with open(monologue_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    monologue_df = pd.DataFrame()
    for agent in data.keys():
        for i in range(len(data[agent])):
            model_responses = data[agent][i].get('model_responses', np.nan)
            best_selection = data[agent][i].get('best_selection', None)
            if best_selection:
                model_response = model_responses[best_selection['best_index']-1]
            else:
                model_response = model_responses[random.randint(0, len(model_responses)-1)]
            gpt_response = data[agent][i].get('gpt_response', np.nan)
            gpt_judgment = data[agent][i].get('gpt_judgment', [])
            final_comparison = data[agent][i].get('final_comparison', {})
            winner = final_comparison.get('winner', np.nan)
            justification = final_comparison.get('justification', np.nan)
            tmp = {'resopnse_idx':i,
                   'agent_name':agent,
                   'model_response':model_response,
                   'model_responses':model_responses,
                   'gpt_response':gpt_response,
                   'gpt_judgment':gpt_judgment,
                   'winner':winner,
                   'justification':justification,
                   'true_completion':data[agent][i].get('monologue', np.nan),
                   }
            monologue_df = pd.concat([monologue_df, pd.DataFrame([tmp])])
    monologue_df = monologue_df.reset_index(drop=True)

    # Belief DF
    with open(belief_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    belief_df = pd.DataFrame()
    for agent in data.keys():
        for i in range(len(data[agent])):
            model_response = data[agent][i].get('response', None)
            gpt_response = data[agent][i].get('gpt_response', None)
            final_comparison = data[agent][i].get('final_comparison', {})
            winner = final_comparison.get('winner', np.nan)
            justification = final_comparison.get('justification', np.nan)
            evaluation_response = data[agent][i].get('evaluation_response', {})
            evaluation_gpt_response = data[agent][i].get('evaluation_gpt_response', {})

            if not model_response:
                continue
            tmp = {'resopnse_idx':i,
                   'agent_name':agent,
                   'model_response':model_response,
                   'gpt_response':gpt_response,
                   'gpt_judgment': [{'response_idx': 0, 'score':evaluation_response.get('score', np.nan), 'explanation':evaluation_response.get('explanation', np.nan)},
                       {'response_idx': 'gpt', 'score':evaluation_gpt_response.get('score', np.nan), 'explanation':evaluation_gpt_response.get('explanation', np.nan)} ],                
                   'winner':winner,
                   'justification':justification,
                   'true_completion':data[agent][i].get('chunk', '')

                   }
            belief_df = pd.concat([belief_df, pd.DataFrame([tmp])])
    belief_df = belief_df.reset_index(drop=True)

    # Memory DF
    with open(memory_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    memory_df = pd.DataFrame()
    for agent in data.keys():
        for i in range(len(data[agent])):
            model_response = data[agent][i].get('response', None)
            gpt_response = data[agent][i].get('gpt_response', None)
            final_comparison = data[agent][i].get('final_comparison', {})
            winner = final_comparison.get('winner', np.nan)
            justification = final_comparison.get('justification', np.nan)
            evaluation_response = data[agent][i].get('evaluation_response', {})
            evaluation_gpt_response = data[agent][i].get('evaluation_gpt_response', {})

            if not model_response:
                continue
            tmp = {'resopnse_idx':i,
                   'agent_name':agent,
                   'model_response':model_response,
                   'gpt_response':gpt_response,
                   'gpt_judgment': [{'response_idx': 0, 'score':evaluation_response.get('score', np.nan), 'explanation':evaluation_response.get('explanation', np.nan)},
                       {'response_idx': 'gpt', 'score':evaluation_gpt_response.get('score', np.nan), 'explanation':evaluation_gpt_response.get('explanation', np.nan)} ],
                   'winner':winner,
                   'justification':justification,
                   'true_completion':data[agent][i].get('chunk', '')

                   }
            memory_df = pd.concat([memory_df, pd.DataFrame([tmp])])
    memory_df = memory_df.reset_index(drop=True)

    # Personality DF
    with open(personality_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    personality_df = pd.DataFrame()
    for agent in data.keys():
        for i in range(len(data[agent])):
            model_response = data[agent][i].get('response', None)
            gpt_response = data[agent][i].get('gpt_response', None)
            final_comparison = data[agent][i].get('final_comparison', {})
            winner = final_comparison.get('winner', np.nan)
            justification = final_comparison.get('justification', np.nan)
            evaluation_response = data[agent][i].get('evaluation_response', {})
            evaluation_gpt_response = data[agent][i].get('evaluation_gpt_response', {})

            if not model_response:
                continue
            tmp = {'resopnse_idx':i,
                   'agent_name':agent,
                   'model_response':model_response,
                   'gpt_response':gpt_response,
                   'gpt_judgment': [{'response_idx': 0, 'score':evaluation_response.get('score', np.nan), 'explanation':evaluation_response.get('explanation', np.nan)},
                       {'response_idx': 'gpt', 'score':evaluation_gpt_response.get('score', np.nan), 'explanation':evaluation_gpt_response.get('explanation', np.nan)} ],
                   'winner':winner,
                   'justification':justification,
                   'true_completion':data[agent][i].get('chunk', '')

                   }
            personality_df = pd.concat([personality_df, pd.DataFrame([tmp])])
    personality_df = personality_df.reset_index(drop=True)

    return completion_df, monologue_df, belief_df, memory_df, personality_df


def make_judgment_df(df):
    # Convert to plain list of dicts
    flat_records = []
    for i, lst in df.iterrows():
        for item in lst['gpt_judgment']:
            
            flat_records.append({
                'example_idx': i,
                'response_idx': item.get('response_idx'),
                'agent':lst['agent_name'],
                'aspect': item.get('aspect'),
                'score': item.get('score'),
                'explanation': item.get('explanation'),
            })
    return pd.DataFrame(flat_records)



def predict_top_speaker(texts, model, tokenizer, SPEAKER2ID, ID2SPEAKER, TARGETS):
    model.eval()
    expected_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    preds = []
    for i, text in enumerate(texts):
        try:
            inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=1044)
            model_inputs = {k: v.to(model.device) for k, v in inputs.items() if k in expected_keys}
            with torch.no_grad():
                outputs = model(**model_inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                top_idx = torch.argmax(probs, dim=1).item()
                top_speaker = ID2SPEAKER.get(top_idx, "unknown")
                preds.append({
                    "speaker_id": top_idx,
                    "speaker_name": top_speaker,
                    "probs": probs.squeeze().tolist()
                })
        except Exception as e:
            preds.append({
                "speaker_id": -1,
                "speaker_name": "No Speaker",
                "probs": [0 for i in range(len(TARGETS))]
            })
    return preds


def compute_monologue_metrics(df, label):
    return {
        "Data": label,
        "Model Accuracy": (df['model_prediction'] == df['agent_name']).mean(),
        "GPT Accuracy": (df['gpt_prediction'] == df['agent_name']).mean(),
        "Model > GPT": (df['model_agent_prob'] >= df['gpt_agent_prob']).mean(),
    }

def plot_monologue_comparison(metrics_df, output_path):
    long_df = metrics_df.melt(id_vars="Data", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_df, x="Data", y="Score", hue="Metric", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Multi-Speaker Model Prediction Across Datasets")
    plt.ylabel("Score")
    plt.xlabel("Dataset")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_model_and_tokenizer(speaker):
    repo_id = f"smerrillunc/{speaker}_prediction"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

def classify_batch(texts, speaker, threshold=0.1, batch_size=32):
    model, tokenizer = load_model_and_tokenizer(speaker)
    device = next(model.parameters()).device
    texts = list(map(str, texts))
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=1024, return_tensors='pt')
    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_probs = []
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
    all_preds = (np.array(all_probs) >= threshold).astype(int)
    return [
        {"text": txt, "probability": float(prob), "predicted_label": int(pred)}
        for txt, prob, pred in zip(texts, all_probs, all_preds)
    ]

def evaluate_fool_rate(df, agent_name, threshold=0.5):
    available_cols = {
        "Model": "model_response",
        "GPT": "gpt_response",
        "TrueCompletion": "true_completion"
    }
    present_cols = {k: v for k, v in available_cols.items() if v in df.columns}
    if not present_cols:
        raise ValueError("No known response columns found in DataFrame.")
    tmp = df.copy()
    tmp = tmp[tmp['agent_name'] == agent_name].dropna(subset=present_cols.values())
    if len(tmp) == 0:
        raise ValueError(f"No valid rows found for agent '{agent_name}'.")
    results = {}
    for label, col in present_cols.items():
        preds = classify_batch(tmp[col].values, agent_name, threshold)
        predicted_labels = np.array([x["predicted_label"] for x in preds])
        fool_rate = predicted_labels.mean()
        results[label] = fool_rate
    return pd.DataFrame.from_dict(results, orient='index', columns=["fool_rate"])

def evaluate_all_fool_rates(df, target_names, threshold=0.5, min_samples=0):
    all_results = []
    for agent in target_names:
        try:
            tmp = df[df["agent_name"] == agent]
            available_cols = [col for col in ["model_response", "gpt_response", "true_completion"] if col in tmp.columns]
            tmp = tmp.dropna(subset=available_cols)
            n = len(tmp)
            if n < min_samples:
                print(f"Skipping {agent} (only {n} examples)")
                continue
            metrics_df = evaluate_fool_rate(df, agent, threshold)
            metrics_df['Agent'] = agent
            metrics_df['num_samples'] = n
            all_results.append(metrics_df.reset_index())
        except ValueError as e:
            print(f"Skipping {agent}: {e}")
    if not all_results:
        raise RuntimeError("No valid results computed for any agents.")
    return pd.concat(all_results, ignore_index=True)

def plot_all_fool_rates(result_df, output_path, min_samples=0):
    # Count number of samples per agent per response type
    if 'count' in result_df.columns:
        counts = result_df.groupby("Agent")["count"].min().reset_index()
        valid_agents = counts[counts["count"] >= min_samples]["Agent"]
        filtered_df = result_df[result_df["Agent"].isin(valid_agents)]
    else:
        filtered_df = result_df
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=filtered_df,
        x='Agent',
        y='fool_rate',
        hue='index',
        palette='Set2'
    )
    plt.title("Fool Rate Across Agents and Response Types")
    plt.ylabel("Fool Rate")
    plt.xlabel("Agent")
    plt.ylim(0, 1)
    plt.legend(title="Response Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_overall_fool_rate_by_response_type(df, target_names, output_path, threshold=0.5):
    response_types = [col for col in ["model_response", "gpt_response", "true_completion"] if col in df.columns]
    results = []
    for col in response_types:
        all_preds = []
        for agent in target_names:
            agent_df = df[df["agent_name"] == agent].dropna(subset=[col])
            if agent_df.empty:
                continue
            preds = classify_batch(agent_df[col].values, agent, threshold)
            pred_labels = [p["predicted_label"] for p in preds]
            all_preds.extend(pred_labels)
        if not all_preds:
            continue
        fool_rate = np.mean(all_preds)
        results.append({"Response Type": col, "Fool Rate": fool_rate, "Total Samples": len(all_preds)})
    summary_df = pd.DataFrame(results)
    plt.figure(figsize=(6, 5))
    sns.barplot(data=summary_df, x="Response Type", y="Fool Rate", palette="Set2")
    plt.title("Overall Fool Rate by Response Type")
    plt.ylabel("Fool Rate")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_win_rate_pie(finetuned_win_rate, gpt_win_rate, save_path=None):
    total = finetuned_win_rate + gpt_win_rate
    if total == 0:
        return
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
    if save_path:
        plt.savefig(save_path)
    plt.close()

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

def plot_votes_accuracy_by_agent(df_, output_dir):
    import matplotlib.ticker as mtick
    df = df_.copy()
    df = df.groupby('agent').aggregate({'correct':'mean','true_vote':'count'})
    df = df.reset_index().rename(columns={'index': 'agent'})
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df, x='agent', y='correct', color='skyblue', edgecolor='black')
    for i, row in df.iterrows():
        ax.text(i, row['correct'] / 2, f"{row['true_vote']} votes", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel("Correct (%)", fontsize=12)
    plt.xlabel("Agent", fontsize=12)
    plt.title("Accuracy of Agent Votes (Total Votes Made)", fontsize=14, fontweight='bold')
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'votes_accuracy_by_agent.png'))
    plt.close()

def plot_votes_accuracy(df_, output_dir):
    df = df_.copy()
    df = df.groupby('agent').aggregate({'correct':'mean','true_vote':'count'})
    df = df.reset_index().rename(columns={'index': 'agent'})
    total_votes = df['true_vote'].sum()
    total_correct_votes = (df['correct'] * df['true_vote']).sum()
    total_incorrect_votes = total_votes - total_correct_votes
    sizes = [total_correct_votes, total_incorrect_votes]
    labels = ['Correct', 'Incorrect']
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
    ax.set_title("Overall Correct vs Incorrect Votes (All Agents)", fontsize=18, weight='bold', pad=30)
    ax.axis('equal')
    plt.subplots_adjust(top=0.88, bottom=0.1)
    plt.savefig(os.path.join(output_dir, 'votes_accuracy_pie.png'))
    plt.close()

def plot_votes_CM(df_, output_dir):
    df = df_.copy()
    true_labels = df['true_vote']
    pred_labels = df['pred_vote']
    classes = sorted(pd.concat([true_labels, pred_labels]).unique())
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Vote', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Vote', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: True Vote vs Predicted Vote', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'votes_confusion_matrix.png'))
    plt.close()

def get_votes_df(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame()
    for agent in data.keys():
        tmp = pd.DataFrame(data[agent])
        tmp['agent'] = agent
        df = pd.concat([df, tmp], ignore_index=True)
    return df





def format_param(x):
    return str(float(x)).rstrip('0').rstrip('.') if '.' in str(x) else str(x)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ##############################################################################################################
    import re
    # --- 1. Perplexity Plots ---
    # Load models config
    models_config_path = os.path.join(args.results_path, "models.json")
    with open(models_config_path, "r", encoding="utf-8") as f:
        models_config = json.load(f)
    model_dirs = list(models_config.values())
    print(model_dirs)
    all_dfs = []
    for path in model_dirs + ['/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/baseline']:
        csv_path = os.path.join(path.replace('/merged',''), "perplexity_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["model_path"] = path
            all_dfs.append(df)
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['model'] = combined_df['model'].apply(lambda x: x.split("/")[-1])
        combined_df.sort_values('perplexity', inplace=True)

        ppl_matrix = combined_df.pivot_table(index="dataset", columns="model", values="perplexity")
        baseline_col = 'baseline'

        if baseline_col in ppl_matrix:
            baseline_ppl = ppl_matrix[baseline_col]
            other_ppl = ppl_matrix.drop(columns=[baseline_col])

            # Rename model columns using the first part before '_'
            new_columns = {col: col.split('_')[0] for col in other_ppl.columns}
            other_ppl.rename(columns=new_columns, inplace=True)
            other_ppl = other_ppl[sorted(other_ppl.columns, key=lambda x: x.lower())]

            # Normalize
            normalized_ppl = other_ppl.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)

            # --- Heatmap for non-baseline models ---
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                normalized_ppl,
                annot=other_ppl.round(2),
                fmt=".2f",
                cmap="coolwarm",
                cbar_kws={'label': 'Normalized Perplexity'}
            )
            plt.title("Perplexity of Each Model (Excl. Baseline)")
            plt.ylabel("Dataset")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            os.makedirs(os.path.join(args.output_dir, 'perplexity'), exist_ok=True)

            plt.savefig(os.path.join(args.output_dir, 'perplexity', "perplexity_heatmap.png"), bbox_inches='tight')
            plt.close()

            # --- Heatmap for baseline model only ---
            plt.figure(figsize=(12, 2))
            sns.heatmap(
                baseline_ppl.to_frame().T,
                annot=baseline_ppl.to_frame().T.round(2),
                fmt=".2f",
                cmap="Blues",
                cbar_kws={'label': 'Baseline Perplexity'}
            )
            plt.title("Baseline Perplexity")
            plt.xlabel("Dataset")
            plt.yticks(rotation=0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'perplexity',  "baseline_perplexity.png"), bbox_inches='tight')
            plt.close()
        else:
            print("No baseline?")


    import glob
    # --- Group all param sets across completion, monologue, and alignment ---
    model = AutoModelForSequenceClassification.from_pretrained("smerrillunc/speaker_prediction")
    tokenizer = AutoTokenizer.from_pretrained("smerrillunc/speaker_prediction", use_fast=True)
        
    completion_path = os.path.join(args.results_path, 'completion_results')
    monologue_path = os.path.join(args.results_path, 'monologue_results')
    alignment_path = os.path.join(args.results_path, 'alignment_results')
    
    completion_files = glob.glob(os.path.join(completion_path, 'test_responses_T*_P*_K*_R*.json'))
    monologue_files = glob.glob(os.path.join(monologue_path, 'test_responses_T*_P*_K*_R*.json'))
    alignment_files = glob.glob(os.path.join(alignment_path, '*_results_T*_P*_K*_R*.json'))
    param_re = re.compile(r'T([\d.]+)_P([\d.]+)_K(\d+)_R([\d.]+)')
    param_sets = set()
    
    # Collect param sets from all three types
    for file in completion_files + monologue_files + alignment_files:
        print(file)
        m = param_re.search(file)
        if m:
            param_sets.add(m.groups())
            
    print(param_sets)
    # For each param set, process all available data
    for (temperature, top_p, top_k, r) in sorted(param_sets):
        r = r[:-1]
        param_dir = os.path.join(args.output_dir, f'T{temperature}_P{top_p}_K{top_k}_R{r}')
        os.makedirs(param_dir, exist_ok=True)
        # --- Completion ---
        comp_file = os.path.join(completion_path, f'test_responses_T{temperature}_P{top_p}_K{top_k}_R{r}.json')
        mono_file = os.path.join(monologue_path, f'test_responses_T{temperature}_P{top_p}_K{top_k}_R{r}.json')
        belief_file = os.path.join(alignment_path, f'belief_results_T{temperature}_P{top_p}_K{top_k}_R{r}.json')
        memory_file = os.path.join(alignment_path, f'memory_results_T{temperature}_P{top_p}_K{top_k}_R{r}.json')
        personality_file = os.path.join(alignment_path, f'personality_results_T{temperature}_P{top_p}_K{top_k}_R{r}.json')

        os.makedirs(os.path.join(param_dir, 'votes'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'completion'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'monologue'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'belief'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'memory'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'personality'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'multi-speaker'), exist_ok=True)
        os.makedirs(os.path.join(param_dir, 'single-speaker'), exist_ok=True)


        completion_df, monologue_df, belief_df, memory_df, personality_df = load_data(comp_file, mono_file, belief_file, memory_file, personality_file)
        
        print("Running Votes Analysis")
        votes_file = os.path.join(alignment_path, f'votes.json')
        if os.path.exists(votes_file):
            votes_df = get_votes_df(votes_file)
            try:
                plot_votes_CM(votes_df, os.path.join(param_dir, 'votes'))
            except Exception as e:
                print(f"[ERROR] Votes confusion matrix failed for param set {temperature},{top_p},{top_k},{r}: {e}")
            try:
                plot_votes_accuracy(votes_df, os.path.join(param_dir, 'votes'))
            except Exception as e:
                print(f"[ERROR] Votes accuracy pie failed for param set {temperature},{top_p},{top_k},{r}: {e}")
            try:
                plot_votes_accuracy_by_agent(votes_df, os.path.join(param_dir, 'votes'))
            except Exception as e:
                print(f"[ERROR] Votes accuracy by agent failed for param set {temperature},{top_p},{top_k},{r}: {e}")




        # Speaker prediction models
        TARGETS = [
            'ellenosborne','davidoberg','grahampaige','jonnoalcaro',
            'katrinacallsen','kateacuff','judyle'
        ]
        SPEAKER2ID = {name: i for i, name in enumerate(TARGETS)}
        ID2SPEAKER = {v: k for k, v in SPEAKER2ID.items()}

        # Predict and add columns for each dataframe
        names = ['completion', 'monologue', 'belief', 'memory', 'personality']
        for i, df in enumerate([completion_df, monologue_df, belief_df, memory_df, personality_df]):
            name = names[i]
            try:
                tmp = completion_df[completion_df.winner.isin(['A', 'B'])].groupby('winner').count()/len(completion_df)
                model_win = tmp[tmp.index == 'A'].values[0,0]
                gpt_win = tmp[tmp.index == 'B'].values[0,0]
                plot_win_rate_pie(model_win, gpt_win, save_path=os.path.join(param_dir, name, f'overall_win_rate.png'))
            except Exception as e:
                print("Failed to plot win rate vs gpt: ", e)
            try:
                plot_agent_win_rates(df, save_path=os.path.join(param_dir, name, f'agent_win_rate.png'))
            except Exception as e:
                print("Failed to plot win rates by agent: ", e)
            
            try:
                if i > 1:
                    tmp1 = make_judgment_df(belief_df)
                    tmp1['aspect'] = 'belief'
                    tmp2 = make_judgment_df(memory_df)
                    tmp2['aspect'] = 'memory'
                    tmp3 = make_judgment_df(personality_df)
                    tmp3['aspect'] = 'personality'
                    alignment = pd.concat([tmp1, tmp2, tmp3])
                    alignment = alignment.dropna(subset=['score'])            
                    plot_score_distributions(alignment,  output_dir=os.path.join(param_dir, name))

                else:
                    judgement_df = make_judgment_df(df)
                    plot_score_distributions(judgement_df, output_dir=os.path.join(param_dir, name))
            except Exception as e:
                print("Failed to make judgement df: ", e)    
            
            
            
            
            
            
            responses = df['model_response'].tolist()
            predictions = predict_top_speaker(responses, model, tokenizer, SPEAKER2ID, ID2SPEAKER, TARGETS)
            df['model_prediction'] = [x['speaker_name'] for x in predictions]
            df['model_top_prob'] = [np.max(x['probs']) for x in predictions]
            df['model_speaker_probs'] = [x['probs'] for x in predictions]
            df["model_agent_prob"] = df.apply(
                lambda row: row["model_speaker_probs"][SPEAKER2ID[row["agent_name"]]], axis=1
            )
            responses = df['gpt_response'].tolist()
            predictions = predict_top_speaker(responses, model, tokenizer, SPEAKER2ID, ID2SPEAKER, TARGETS)
            df['gpt_prediction'] = [x['speaker_name'] for x in predictions]
            df['gpt_top_prob'] = [np.max(x['probs']) for x in predictions]
            df['gpt_speaker_probs'] = [x['probs'] for x in predictions]
            df["gpt_agent_prob"] = df.apply(
                lambda row: row["gpt_speaker_probs"][SPEAKER2ID[row["agent_name"]]], axis=1
            )

        # Compute metrics and plot
        try:
            results = [
                compute_monologue_metrics(completion_df, "Completions"),
                compute_monologue_metrics(monologue_df, "Monologues"),
                compute_monologue_metrics(belief_df, "Beliefs"),
                compute_monologue_metrics(memory_df, "Memory"),
                compute_monologue_metrics(personality_df, "Personality"),
            ]
            metrics_df = pd.DataFrame(results)
            plot_monologue_comparison(metrics_df, os.path.join(param_dir, 'multi-speaker', "comparison.png"))
        except Exception as e:
            print(f"[ERROR] Failed to plot monologue comparison: {e}")


        # --- Speaker Models Section Plots ---
        # ...existing code for speaker models plots...

        # --- 2.1 Completion Results ---
        try:
            results_df = evaluate_all_fool_rates(completion_df, TARGETS)
            plot_all_fool_rates(results_df, os.path.join(param_dir,  'single-speaker', "completion_fool_rates.png"))
            plot_overall_fool_rate_by_response_type(completion_df, TARGETS, os.path.join(param_dir, 'single-speaker', "completion_overall_fool_rate.png"))
        except Exception as e:
            print(f"[ERROR] Completion fool rate plots failed: {e}")

        # --- 2.2 Monologue Results ---
        try:
            results_df = evaluate_all_fool_rates(monologue_df, TARGETS)
            plot_all_fool_rates(results_df, os.path.join(param_dir, 'single-speaker', "monologue_fool_rates.png"))
            plot_overall_fool_rate_by_response_type(monologue_df, TARGETS, os.path.join(param_dir, 'single-speaker', "monologue_overall_fool_rate.png"))
        except Exception as e:
            print(f"[ERROR] Monologue fool rate plots failed: {e}")

        # --- 2.3 Belief Results ---
        try:
            results_df = evaluate_all_fool_rates(belief_df, TARGETS)
            plot_all_fool_rates(results_df, os.path.join(param_dir,  'single-speaker', "belief_fool_rates.png"))
            plot_overall_fool_rate_by_response_type(belief_df, TARGETS, os.path.join(param_dir,  'single-speaker', "belief_overall_fool_rate.png"))
        except Exception as e:
            print(f"[ERROR] Belief fool rate plots failed: {e}")

        # --- 2.4 Memory Results ---
        try:
            results_df = evaluate_all_fool_rates(memory_df, TARGETS)
            plot_all_fool_rates(results_df, os.path.join(param_dir,  'single-speaker', "memory_fool_rates.png"))
            plot_overall_fool_rate_by_response_type(memory_df, TARGETS, os.path.join(param_dir,  'single-speaker', "memory_overall_fool_rate.png"))
        except Exception as e:
            print(f"[ERROR] Memory fool rate plots failed: {e}")

        # --- 2.5 Personality Results ---
        try:
            results_df = evaluate_all_fool_rates(personality_df, TARGETS)
            plot_all_fool_rates(results_df, os.path.join(param_dir,  'single-speaker', "personality_fool_rates.png"))
            plot_overall_fool_rate_by_response_type(personality_df, TARGETS, os.path.join(param_dir,  'single-speaker', "personality_overall_fool_rate.png"))
        except Exception as e:
            print(f"[ERROR] Personality fool rate plots failed: {e}")



if __name__ == "__main__":
    main()