"""Plot evaluation curves from CSVs downloaded from Weights & Biases.

Replace --root with the local folder that holds those exported evaluation
CSVs (grouped in experiment subfolders). Folder names should include
tokens such as beta_1.0 or lambda_0.25 if you use --param to split curves.

Example::

    python scripts/plot_evaluation.py --root path/to/wandb_evaluation_csvs --param beta
"""
import argparse
import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def parse_param_value(filename, param_name):
    """Extract the value of ``param_name`` from a folder or file name."""
    pattern = f"{param_name}_([^_]+)"
    match = re.search(pattern, filename)

    if match:
        val_str = match.group(1)
        try:
            return float(val_str)
        except ValueError:
            return val_str
    return None


def plot_auto_comparison(
    root_dir,
    target_param,
    fixed_filters=None,
    smooth_window=1,
    bin_size=None,
    figsize=(10, 6),
    xlim=None,
    palette='viridis'
):
    print(f"Searching for folders with parameter: '{target_param}' in {root_dir}...")

    all_subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    valid_tasks = []

    for folder in all_subdirs:
        val = parse_param_value(folder, target_param)
        if val is None:
            continue

        skip = False
        if fixed_filters:
            for f_key, f_val in fixed_filters.items():
                current_f_val = parse_param_value(folder, f_key)
                if str(current_f_val) != str(f_val):
                    skip = True
                    break
        if skip:
            continue

        valid_tasks.append((val, folder))

    if not valid_tasks:
        print(f"Error: No folders found containing parameter '{target_param}'.")
        return

    # Sort numbers before strings so mixed types do not raise TypeError.
    valid_tasks.sort(key=lambda x: (0, x[0]) if isinstance(x[0], (int, float)) else (1, str(x[0])))

    print(f"Found {len(valid_tasks)} matching experiments:")
    for val, folder in valid_tasks:
        print(f"  [{target_param} = {val}] -> {folder}")

    all_data = []
    for val, folder in valid_tasks:
        full_path = os.path.join(root_dir, folder)
        csv_files = glob.glob(os.path.join(full_path, "*.csv"))

        task_dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                col_mapping = {}
                for c in df.columns:
                    if 'step' in c:
                        col_mapping[c] = 'step'
                    elif 'value' in c:
                        col_mapping[c] = 'value'
                df = df.rename(columns=col_mapping)

                if 'step' not in df or 'value' not in df:
                    continue

                if bin_size:
                    df['step'] = (np.round(df['step'] / bin_size) * bin_size).astype(int)

                if smooth_window > 1:
                    df['value'] = df['value'].rolling(window=smooth_window, min_periods=1).mean()

                task_dfs.append(df)
            except Exception:
                pass

        if task_dfs:
            combined = pd.concat(task_dfs, ignore_index=True)
            combined['Param Value'] = val
            all_data.append(combined)

    if not all_data:
        print("No valid CSV data found.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)

    if xlim:
        final_df = final_df[(final_df['step'] >= xlim[0]) & (final_df['step'] <= xlim[1])]

    is_numeric = all(isinstance(x, (int, float)) for x in final_df['Param Value'].unique())
    current_palette = palette
    if not is_numeric:
        current_palette = "tab10"

    sns.lineplot(
        data=final_df,
        x='step',
        y='value',
        hue='Param Value',
        errorbar='sd',
        palette=current_palette,
        linewidth=2.5,
    )

    title_str = f'Evaluation: Varying {target_param}'
    if fixed_filters:
        subtitle = ", ".join([f"{k}={v}" for k, v in fixed_filters.items()])
        title_str += f'\n({subtitle})'

    plt.title(title_str, fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Evaluation Value', fontsize=12)
    plt.legend(title=target_param, title_fontsize=11)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot evaluation CSVs downloaded from Weights & Biases."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Folder of W&B-exported evaluation CSVs (experiment subfolders). Replace with your local path.",
    )
    parser.add_argument("--param", default="beta", help="Hyperparameter token in the folder name.")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--palette", default="Set2")
    args = parser.parse_args()

    plot_auto_comparison(
        root_dir=args.root,
        target_param=args.param,
        smooth_window=args.smooth_window,
        bin_size=args.bin_size,
        palette=args.palette,
    )
