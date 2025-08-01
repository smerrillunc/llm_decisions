import os
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def load_models_config(config_path):
    logging.info(f"Loading models config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_records(models_config):
    records = []
    for agent, path in models_config.items():
        try:
            parts = path.split('/')[-2].split('_')
            record = {
                'agent': parts[0],
                'lora_rank': int(parts[1]),
                'lora_dropout': float(parts[2]),
                'learning_rate': float(parts[3]),
                'epochs': int(parts[4]),
                'path': path
            }
            records.append(record)
        except (IndexError, ValueError) as e:
            logging.warning(f"Skipping {agent} due to error parsing path '{path}': {e}")
    return records

def create_summary_df(records, content_words_data):
    df = pd.DataFrame(records)
    df['lora_alpha'] = 2 * df['lora_rank']
    
    content_df = pd.DataFrame([
        {'agent': agent, 'content_words': count}
        for agent, count in content_words_data.items()
    ])
    
    df = df.merge(content_df, on='agent', how='left')
    df = df.sort_values(by='content_words', ascending=False)
    return df

def save_markdown_summary(df, output_path):
    cols = ['agent', 'content_words', 'lora_rank', 'lora_alpha', 'lora_dropout', 'learning_rate', 'epochs', 'path']
    markdown_str = df[cols].to_markdown(index=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_str)
    logging.info(f"Markdown summary saved to {output_path}")

def load_perplexity_data(model_dirs):
    all_dfs = []
    for path in model_dirs:
        csv_path = os.path.join(path.replace('/merged', ''), "perplexity_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["model_path"] = path  # Track source path
            all_dfs.append(df)
        else:
            logging.warning(f"Perplexity CSV not found: {csv_path}")
    if not all_dfs:
        logging.error("No perplexity data loaded; please check your paths.")
        raise FileNotFoundError("No perplexity CSV files found.")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['model'] = combined_df['model'].apply(lambda x: x.split("/")[-1])
    combined_df.sort_values('perplexity', inplace=True)
    return combined_df

def create_perplexity_plot(combined_df, output_path):
    ppl_matrix = combined_df.pivot_table(index="dataset", columns="model", values="perplexity")
    if 'baseline' not in ppl_matrix.columns:
        logging.error("'baseline' model column not found in perplexity data.")
        raise KeyError("'baseline' model not found in data.")

    baseline_ppl = ppl_matrix['baseline']
    other_ppl = ppl_matrix.drop(columns=['baseline'])

    other_ppl = other_ppl.sort_index()
    baseline_ppl = baseline_ppl.sort_index()

    other_ppl.columns = other_ppl.columns.astype(str)
    other_ppl = other_ppl[sorted(other_ppl.columns, key=lambda x: x.lower())]

    normalized_ppl = other_ppl.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)

    short_model_names = [name.split('_')[0] for name in normalized_ppl.columns]
    normalized_ppl.columns = short_model_names
    other_ppl.columns = short_model_names  # For annotations

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 0.7])

    ax1 = plt.subplot(gs[0])
    sns.heatmap(
        normalized_ppl,
        annot=other_ppl.round(2),
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Perplexity'},
        ax=ax1
    )
    ax1.set_title("Perplexity of Each Model (Excl. Baseline)")
    ax1.set_ylabel("Dataset")
    ax1.set_xlabel("Model")

    ax2 = plt.subplot(gs[1])
    sns.heatmap(
        baseline_ppl.to_frame().T,
        annot=baseline_ppl.to_frame().T.round(2),
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': 'Perplexity'},
        ax=ax2
    )
    ax2.set_title("LLaMA‑70B Perplexity")
    ax2.set_ylabel("")
    ax2.set_xlabel("Dataset")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["LLaMA‑70B"], rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"Perplexity plot saved to {output_path}")
    plt.show()

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Paths and data
    models_config_path = args.models_config
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    models_config = load_models_config(models_config_path)

    records = extract_records(models_config)

    content_words_data = {
        'ellenosborne': 5728,
        'judyle': 17462,
        'davidoberg': 19247,
        'kateacuff': 22398,
        'jonnoalcaro': 31320,
        'katrinacallsen': 41910,
        'grahampaige': 45121,
    }

    summary_df = create_summary_df(records, content_words_data)
    markdown_path = os.path.join(output_dir, "model_summary.md")
    save_markdown_summary(summary_df, markdown_path)

    model_dirs = list(models_config.values()) + [
        '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/baseline'
    ]

    combined_df = load_perplexity_data(model_dirs)

    plot_path = os.path.join(output_dir, "combined_perplexity_plot.png")
    create_perplexity_plot(combined_df, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process models config and plot perplexity results.")
    parser.add_argument(
        "--models_config",
        type=str,
        default="/playpen-ssd/smerrill/llm_decisions/configs/models.json",
        help="Path to models.json configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory to save markdown summary and plots"
    )

    args = parser.parse_args()
    main(args)
