"""Plot smoothed training-reward curves from CSVs downloaded from Weights & Biases.

Replace --data-dir with the local folder that holds those exported reward
CSVs, and --log-file with a glob matching the file names (tokens such as
lambda_0.25 in the name are used to group curves).

Example::

    python scripts/plot_smooth.py --data-dir path/to/wandb_reward_csvs \\
        --log-file "batch_*_featureDim_512_seed_*_lambda_*_beta_*_extraFeatureStep_1_experiment_*.csv" \\
        --param batch
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import re
import argparse

def parse_lambda_from_filename(filename):
    """Extract lambda from a file name."""
    # Match the lambda token with a regular expression.
    match = re.search(r'lambda_([\d.]+)', filename)
    if match:
        return float(match.group(1))
    return None

def smooth_data(data, smooth_window=1):
    """
    Smooth a 1-D series with a moving average.
    
    Parameters:
    -----------
    data : array-like
        Series to smooth.
    smooth_window : int
        Window length; forced to an odd integer.
    """
    if smooth_window <= 1:
        return data
    
    # Force an odd window length.
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    y = np.ones(smooth_window)
    x = np.asarray(data)
    z = np.ones(len(x))
    
    # Moving average via convolution.
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    
    return smoothed_x

def plot_data_with_smoothing(data_directory, log_file,smooth_window=1, xaxis='episode', value='reward', **kwargs):
    """
    Plot reward curves with optional smoothing.
    """
    
    pattern = os.path.join(data_directory, log_file)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No file found, please check:{data_directory}")
        print(f"Searching in:{pattern}")
        return

    print(f"find {len(csv_files)} files")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")

    # Load and preprocess CSV logs.
    experiment_data = {}
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        
        # Apply smoothing.
        if smooth_window > 1:
            smoothed_rewards = smooth_data(df['Reward'].values, smooth_window)
            experiment_data[f'exp_{i+1}'] = smoothed_rewards
        else:
            experiment_data[f'exp_{i+1}'] = df['Reward'].values

    # One DataFrame column per experiment.
    max_length = max(len(arr) for arr in experiment_data.values())
    wide_df = pd.DataFrame()

    for exp_name, rewards in experiment_data.items():
        # Pad series to a common length.
        padded_rewards = np.pad(rewards, (0, max_length - len(rewards)), 
                               constant_values=np.nan)
        wide_df[exp_name] = padded_rewards

    # Add an episode index column.
    wide_df['episode'] = range(max_length)

    # Convert to long form for seaborn.
    long_df = wide_df.melt(id_vars=['episode'], 
                          var_name='experiment', 
                          value_name='reward')

    # Draw the figure.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Line plot with an error band.
    sns.lineplot(data=long_df, x='episode', y='reward', errorbar='sd', **kwargs)
    
    # Set the title.
    if smooth_window > 1:
        plt.title(f'Reward vs Episode (Smoothed with window={smooth_window})')
    else:
        plt.title('Reward vs Episode (Raw Data)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.show()
    
    return long_df

# Variant with extra smoothing methods.
def plot_data_advanced(data_directory, log_file, smooth_method='moving_average', 
                      smooth_window=5, sigma=2, resample_interval=None,
                      xaxis='episode', value='reward', **kwargs):
    """Plot reward curves with extra smoothing options."""
    def apply_smoothing_advanced(data, method='moving_average', window_size=5, sigma=2):
        """Apply the selected smoother."""
        if method == 'moving_average':
            return smooth_data(data, window_size)
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(data, sigma=sigma)
        elif method == 'exponential':
            # Exponential weighted moving average (pandas).
            series = pd.Series(data)
            return series.ewm(span=window_size).mean().values
        else:
            return data
    
    
    pattern = os.path.join(data_directory, log_file)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No file found, please check:{data_directory}")
        return

    print(f"find {len(csv_files)} files")
    
    experiment_data = {}
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        
        # Apply the chosen smoother.
        smoothed_rewards = apply_smoothing_advanced(
            df['Reward'].values, 
            method=smooth_method,
            window_size=smooth_window,
            sigma=sigma
        )
        
        experiment_data[f'exp_{i+1}'] = smoothed_rewards

    # Build the DataFrame.
    max_length = max(len(arr) for arr in experiment_data.values())
    wide_df = pd.DataFrame()

    for exp_name, rewards in experiment_data.items():
        padded_rewards = np.pad(rewards, (0, max_length - len(rewards)), 
                               constant_values=np.nan)
        wide_df[exp_name] = padded_rewards

    # Optional resampling to thin the series.
    if resample_interval and max_length > resample_interval * 2:
        wide_df = wide_df.iloc[::resample_interval].reset_index(drop=True)
        wide_df['episode'] = range(0, max_length, resample_interval)
    else:
        wide_df['episode'] = range(max_length)

    # Convert to long form for seaborn.
    long_df = wide_df.melt(id_vars=['episode'], 
                          var_name='experiment', 
                          value_name='reward')

    # Draw the figure.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=long_df, x='episode', y='reward', errorbar='sd', **kwargs)
    
    # Title depends on the smoother.
    method_names = {
        'moving_average': f'Moving average (window={smooth_window})',
        'gaussian': f'Gaussian smoothing (sigma={sigma})',
        'exponential': f'Exponential smoothing (span={smooth_window})'
    }
    
    title = method_names.get(smooth_method, 'raw data')
    if resample_interval:
        title += f' (resample interval={resample_interval})'
    
    plt.title(f'Reward vs Episode - {title}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.show()
    
    return long_df

def plot_multiple_lambdas(data_directory, log_file, smooth_window=1, xaxis='episode', value='reward', **kwargs):
    """
    Overlay reward curves for several lambda values.
    
    Parameters:
    -----------
    data_directory : str
        Directory of CSV logs.
    smooth_window : int
        Smoothing window length.
    """
    # Find matching CSV files.
    
    pattern = os.path.join(data_directory, log_file)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No file found, please check:{data_directory}")
        print(f"Searching in:{pattern}")
        return

    print(f"find {len(csv_files)} files")
    
    # Group files by lambda.
    lambda_groups = {}
    for file in csv_files:
        lambda_val = parse_lambda_from_filename(os.path.basename(file))
        if lambda_val is not None:
            if lambda_val not in lambda_groups:
                lambda_groups[lambda_val] = []
            lambda_groups[lambda_val].append(file)
            print(f"  - Lambda {lambda_val}: {os.path.basename(file)}")
    
    # Count experiments in each lambda group.
    for lambda_val, files in lambda_groups.items():
        print(f"Lambda {lambda_val}: {len(files)} experiments")
    
    # Process each lambda group.
    all_lambda_data = []
    
    for lambda_val, files in lambda_groups.items():
        lambda_experiment_data = {}
        
        for i, file in enumerate(files):
            df = pd.read_csv(file)
            
            # Apply smoothing.
            if smooth_window > 1:
                smoothed_rewards = smooth_data(df['Reward'].values, smooth_window)
                lambda_experiment_data[f'exp_{i+1}'] = smoothed_rewards
            else:
                lambda_experiment_data[f'exp_{i+1}'] = df['Reward'].values
        
        # One DataFrame column per experiment.
        max_length = max(len(arr) for arr in lambda_experiment_data.values())
        wide_df = pd.DataFrame()

        for exp_name, rewards in lambda_experiment_data.items():
            # Pad series to a common length.
            padded_rewards = np.pad(rewards, (0, max_length - len(rewards)), 
                                   constant_values=np.nan)
            wide_df[exp_name] = padded_rewards

        # Add an episode index column.
        wide_df['episode'] = range(max_length)
        
        # Convert to long form for seaborn.
        long_df = wide_df.melt(id_vars=['episode'], 
                              var_name='experiment', 
                              value_name='reward')
        
        # Tag rows with lambda.
        long_df['lambda'] = lambda_val
        
        # Append to the combined table.
        all_lambda_data.append(long_df)
    
    # Concatenate all lambda groups.
    combined_df = pd.concat(all_lambda_data, ignore_index=True)
    
    # Draw the figure.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Line plot coloured by lambda.
    sns.lineplot(
        data=combined_df, 
        x='episode', 
        y='reward', 
        hue='lambda',
        errorbar='sd', 
        palette='viridis',  # distinct colours
        **kwargs
    )
    
    # Title and legend.
    if smooth_window > 1:
        plt.title(f'Reward vs Episode - Multiple Lambdas (Smoothed with window={smooth_window})')
    else:
        plt.title('Reward vs Episode - Multiple Lambdas (Raw Data)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Place the legend.
    plt.legend(title='Lambda Value', loc='best')
    plt.tight_layout()
    plt.show()
    
    return combined_df

# Overlay several lambdas with extra options.
def plot_multiple_lambdas_advanced(
    data_directory, 
    log_file, 
    smooth_method='moving_average', 
    smooth_window=5, 
    sigma=2, 
    resample_interval=None,
    xaxis='episode', 
    value='reward',
    errorbar_type='sd',
    palette='viridis',
    figsize=(12, 8),
    **kwargs
):
    """Overlay several lambda values with extra smoothing options."""
    def apply_smoothing_advanced(data, method='moving_average', window_size=5, sigma=2):
        """Apply the selected smoother."""
        if method == 'moving_average':
            return smooth_data(data, window_size)
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(data, sigma=sigma)
        elif method == 'exponential':
            # Exponential weighted moving average (pandas).
            series = pd.Series(data)
            return series.ewm(span=window_size).mean().values
        else:
            return data
    
    # Find matching CSV files.
    
    pattern = os.path.join(data_directory, log_file)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No file found, please check:{data_directory}")
        return

    print(f"find {len(csv_files)} files")
    
    # Group files by lambda.
    lambda_groups = {}
    for file in csv_files:
        lambda_val = parse_lambda_from_filename(os.path.basename(file))
        if lambda_val is not None:
            if lambda_val not in lambda_groups:
                lambda_groups[lambda_val] = []
            lambda_groups[lambda_val].append(file)
            print(f"  - Lambda {lambda_val}: {os.path.basename(file)}")
    
    # Process each lambda group.
    all_lambda_data = []
    
    for lambda_val, files in lambda_groups.items():
        lambda_experiment_data = {}
        
        for i, file in enumerate(files):
            df = pd.read_csv(file)
            
            # Apply smoothing.
            smoothed_rewards = apply_smoothing_advanced(
                df['Reward'].values, 
                method=smooth_method,
                window_size=smooth_window,
                sigma=sigma
            )
            lambda_experiment_data[f'exp_{i+1}'] = smoothed_rewards
        
        # One DataFrame column per experiment.
        max_length = max(len(arr) for arr in lambda_experiment_data.values())
        wide_df = pd.DataFrame()

        for exp_name, rewards in lambda_experiment_data.items():
            # Pad series to a common length.
            padded_rewards = np.pad(rewards, (0, max_length - len(rewards)), 
                                   constant_values=np.nan)
            wide_df[exp_name] = padded_rewards

        # Optional resampling to thin the series.
        if resample_interval and max_length > resample_interval * 2:
            wide_df = wide_df.iloc[::resample_interval].reset_index(drop=True)
            wide_df['episode'] = range(0, max_length, resample_interval)
        else:
            wide_df['episode'] = range(max_length)
        
        # Convert to long form for seaborn.
        long_df = wide_df.melt(id_vars=['episode'], 
                              var_name='experiment', 
                              value_name='reward')
        
        # Tag rows with lambda.
        long_df['lambda'] = lambda_val
        
        # Append to the combined table.
        all_lambda_data.append(long_df)
    
    # Concatenate all lambda groups.
    combined_df = pd.concat(all_lambda_data, ignore_index=True)
    
    # Draw the figure.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)

    # Line plot coloured by lambda.
    sns.lineplot(
        data=combined_df, 
        x='episode', 
        y='reward', 
        hue='lambda',
        errorbar=errorbar_type,
        palette=palette,
        **kwargs
    )
    
    # Title depends on the smoother.
    method_names = {
        'moving_average': f'Moving Average (Window={smooth_window})',
        'gaussian': f'Gaussian Smooth (σ={sigma})',
        'exponential': f'Exponential Smooth (Width={smooth_window})'
    }
    
    title = f'Reward vs Episode - Multiple Lambdas - {method_names.get(smooth_method, "original data")}'
    if resample_interval:
        title += f' (resample_interval={resample_interval})'
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Place the legend.
    plt.legend(title='Lambda Value', loc='best')
    plt.tight_layout()
    plt.show()
    
    return combined_df

# Overlay curves grouped by an arbitrary hyperparameter.
def plot_multiple_params_advanced(
    data_directory, 
    log_file, 
    param_name='beta',  # Hyperparameter to compare: beta, lambda, seed, ...
    smooth_method='moving_average', 
    smooth_window=5, 
    sigma=2, 
    resample_interval=None,
    xaxis='episode', 
    value='reward',
    errorbar_type='sd',
    palette='viridis',
    figsize=(12, 8),
    xlim=None,
    hue_order=None,
    sort_ascending=True,
    margin_ratio=0.05,
    **kwargs
):
    """
    Overlay curves grouped by an arbitrary hyperparameter.
    
    Parameters:
    -----------
    param_name : str
        Hyperparameter to compare, e.g. beta, lambda, seed.
    """
    def parse_param_from_filename(filename, param):
        """Parse a named hyperparameter from a file stem (number or string)."""
        # Strip the suffix so .csv is not parsed as a value.
        clean_name = os.path.splitext(filename)[0]
        
        # Match letters, digits, and dots after param_.
        # Read the token after param_ until the next underscore or end of string.
        pattern = f"{param}_([^_]+)"
        
        match = re.search(pattern, clean_name)
        if match:
            val_str = match.group(1)
            # Cast to float when possible; otherwise keep the string (e.g. median).
            try:
                return float(val_str)
            except ValueError:
                return val_str
        return None
    
    def apply_smoothing_advanced(data, method='moving_average', window_size=5, sigma=2):
        """Apply the selected smoother."""
        if method == 'moving_average':
            return smooth_data(data, window_size)
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(data, sigma=sigma)
        elif method == 'exponential':
            # Exponential weighted moving average (pandas).
            series = pd.Series(data)
            return series.ewm(span=window_size).mean().values
        else:
            return data
    
    # Find matching CSV files.
    pattern = os.path.join(data_directory, log_file)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No file found, please check:{data_directory}")
        return

    print(f"find {len(csv_files)} files")
    
    # Group files by that hyperparameter.
    param_groups = {}
    for file in csv_files:
        param_val = parse_param_from_filename(os.path.basename(file), param_name)
        if param_val is not None:
            if param_val not in param_groups:
                param_groups[param_val] = []
            param_groups[param_val].append(file)
            print(f"  - {param_name} {param_val}: {os.path.basename(file)}")
    
    min_length = float('inf')
    for files in param_groups.values():
        for file in files:
            df = pd.read_csv(file)
            min_length = min(min_length, len(df))
    
    print(f"Shortest series length: {min_length}")

    # Process each hyperparameter group.
    all_param_data = []
    
    for param_val, files in param_groups.items():
        param_experiment_data = {}
        
        for i, file in enumerate(files):
            df = pd.read_csv(file)
            
            if len(df) > min_length:
                df = df.head(min_length)

            # Apply smoothing.
            smoothed_rewards = apply_smoothing_advanced(
                df['Reward'].values, 
                method=smooth_method,
                window_size=smooth_window,
                sigma=sigma
            )
            param_experiment_data[f'exp_{i+1}'] = smoothed_rewards
        
        # One DataFrame column per experiment.
        #max_length = max(len(arr) for arr in param_experiment_data.values())
        #wide_df = pd.DataFrame()
        wide_df = pd.DataFrame(index=range(min_length))

        for exp_name, rewards in param_experiment_data.items():
            # Trim or pad rewards to min_length.
            if len(rewards) > min_length:
                rewards = rewards[:min_length]
            elif len(rewards) < min_length:
                # Pad with NaN if the series is shorter.
                rewards = np.pad(rewards, (0, min_length - len(rewards)), constant_values=np.nan)
            
            wide_df[exp_name] = rewards

        #max_length = max(len(arr) for arr in param_experiment_data.values())
        #wide_df = pd.DataFrame()

        #for exp_name, rewards in param_experiment_data.items():
            # Pad series to a common length.
            #padded_rewards = np.pad(rewards, (0, max_length - len(rewards)), constant_values=np.nan)
            #wide_df[exp_name] = padded_rewards

        # Optional resampling to thin the series.
        #if resample_interval and max_length > resample_interval * 2:
            #wide_df = wide_df.iloc[::resample_interval].reset_index(drop=True)
            #wide_df['episode'] = range(0, max_length, resample_interval)
        #else:
            #wide_df['episode'] = range(max_length)
        if resample_interval and min_length > resample_interval * 2:
            wide_df = wide_df.iloc[::resample_interval].reset_index(drop=True)
            wide_df['episode'] = range(0, min_length, resample_interval)
        else:
            wide_df['episode'] = range(min_length)
        
        # Convert to long form for seaborn.
        long_df = wide_df.melt(id_vars=['episode'], 
                              var_name='experiment', 
                              value_name='reward')
        
        # Tag rows with the hyperparameter value.
        long_df[param_name] = param_val
        
        # Append to the combined table.
        all_param_data.append(long_df)
    
    # Concatenate all groups.
    combined_df = pd.concat(all_param_data, ignore_index=True)

    if xlim is not None:
        min_x, max_x = xlim
        combined_df = combined_df[(combined_df['episode'] >= min_x) & (combined_df['episode'] <= max_x)]
    
    if hue_order is None:
        unique_params = combined_df[param_name].unique()
        try:
            # Sort numerically when possible; reverse follows sort_ascending.
            # sort_ascending=False means descending order.
            hue_order = sorted(unique_params, key=float, reverse=not sort_ascending)
        except:
            # Fall back to string sort when values are not numeric.
            hue_order = sorted(unique_params, key=str, reverse=not sort_ascending)

    # Draw the figure.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)

    # Line plot coloured by the hyperparameter.
    sns.lineplot(
        data=combined_df, 
        x='episode', 
        y='reward', 
        hue=param_name,
        errorbar=errorbar_type,
        palette=palette,
        #legend=False,
        **kwargs
    )
    
    # Title depends on the smoother.
    method_names = {
        'moving_average': f'Moving Average (Window={smooth_window})',
        'gaussian': f'Gaussian Smooth (σ={sigma})',
        'exponential': f'Exponential Smooth (Width={smooth_window})'
    }
    
    title = f'Reward vs Episode - Multiple {param_name.capitalize()}s - {method_names.get(smooth_method, "original data")}'
    #title = f'Reward of Vanilla SAC'
    if resample_interval:
        title += f' (resample_interval={resample_interval})'
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Set x-limits with a margin.
    if xlim is not None:
        min_x, max_x = xlim
        padding = (max_x - min_x) * margin_ratio
        plt.xlim(min_x - padding, max_x + padding)
    else:
        # If xlim is omitted, use the shortest series plus a margin.
        padding = min_length * margin_ratio
        plt.xlim(-padding, min_length + padding)
    
    # Place the legend.
    plt.legend(title=f'{param_name.capitalize()} Value', loc='best')
    #plt.margins(x=0.1)
    plt.tight_layout()
    plt.show()
    
    return combined_df




# Example: pass your local CSV directory on the command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training-reward CSVs downloaded from Weights & Biases."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder of W&B-exported reward CSVs. Replace with your local path.",
    )
    parser.add_argument(
        "--log-file",
        default="batch_*_featureDim_512_seed_*_lambda_*_beta_*_extraFeatureStep_1_experiment_*.csv",
        help="Glob for CSV names inside --data-dir.",
    )
    parser.add_argument("--param", default="batch", help="Hyperparameter token to group by.")
    parser.add_argument("--smooth-window", type=int, default=35)
    parser.add_argument("--resample-interval", type=int, default=2)
    args = parser.parse_args()

    plot_multiple_params_advanced(
        args.data_dir,
        args.log_file,
        param_name=args.param,
        smooth_method="moving_average",
        smooth_window=args.smooth_window,
        resample_interval=args.resample_interval,
        errorbar_type=("ci", 95),
        palette="Set2",
        figsize=(14, 9),
        sort_ascending=False,
        margin_ratio=0.1,
    )
