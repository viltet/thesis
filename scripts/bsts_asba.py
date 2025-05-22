import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- 1. Configuration ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent # Directory of the current script (scripts/)
    PROJECT_ROOT_DIR = SCRIPT_DIR.parent        # Go up one level to the main 'thesis' folder
except NameError:
    # Fallback for interactive environments - assumes you run from project root or scripts folder
    PROJECT_ROOT_DIR = Path.cwd()
    if PROJECT_ROOT_DIR.name == "scripts": # If cwd is scripts, go up
        PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

RESULTS_DIR = PROJECT_ROOT_DIR / "results"  # Now points to thesis/results/
BSTS_OUTPUT_DIR = RESULTS_DIR / "bsts_outputs" / "absa_sentiment" # This will now be thesis/results/bsts_outputs/absa_sentiment/
BSTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which assistant to analyze: "alexa" or "google"
ASSISTANT_NAME_TO_ANALYZE = "alexa"  # Change this to "google" to analyze Google Assistant

# Choose which aspect to analyze (or set to None to analyze all aspects found in the file)
# Example: "Functionality & Performance", "Voice Recognition", "Privacy & Security", etc.
# These names must exactly match the values in your 'identified_aspect' column.
ASPECT_TO_ANALYZE = None # "Functionality & Performance" # Example: Set to a specific aspect for testing


# --- ABSA Data File Mapping ---
ABSA_FILE_MAPPING = {
    "alexa": "absa_full_results_colab/alexa_full_absa_sentiments_colab.csv", # <-- Include the subdirectory here
    "google": "absa_full_results_colab/google_full_absa_sentiments_colab_COMBINED.csv" # <-- Include the subdirectory here
}

# --- Intervention Events ---
if ASSISTANT_NAME_TO_ANALYZE == "alexa":
    INTERVENTION_EVENTS = [
        {"name": "Alexa_Hunches_Introduction_Sep2018", "date": "2018-09-20"},
        {"name": "Alexa_Wolfram_Alpha_Integration_Dec2018", "date": "2018-12-20"},
        {"name": "Alexa_Privacy_Hub_Launched_Sep2019", "date": "2019-09-25"},
        {"name": "Alexa_Reminders_Across_Devices_Jun2020", "date": "2020-06-15"},
        {"name": "Alexa_Proactive_Hunches_Guard_Plus_Jan2021", "date": "2021-01-25"},
        {"name": "Alexa_Smarter_Alexa_New_Echo_Show_Sep2023", "date": "2023-09-20"},
        {"name": "Alexa_Plus_Launch_Feb2025", "date": "2025-02-26"}
    ]
elif ASSISTANT_NAME_TO_ANALYZE == "google":
    INTERVENTION_EVENTS = [
        {"name": "Google_Additional_Languages_Routines_Feb2018", "date": "2018-02-07"},
        {"name": "Google_Duplex_Announced_May2018", "date": "2018-05-08"},
        {"name": "Google_New_Features_Update_Nov2018", "date": "2018-11-15"},
        {"name": "Google_Interpreter_Mode_Dec2019", "date": "2019-12-12"},
        {"name": "Google_Voice_Match_Expansion_Jun2020", "date": "2020-06-15"},
        {"name": "Google_Simple_Nest_Hub_Features_Jun2020", "date": "2020-06-20"},
        {"name": "Google_iOS_Fixes_Stability_Dec2021", "date": "2021-12-15"},
        {"name": "Google_Shortcuts_Plus_Expanded_Support_May2022", "date": "2022-05-03"},
        {"name": "Google_Smart_Assistant_Improvements_Mar2023", "date": "2023-03-15"},
        {"name": "Google_Gemini_Assistant_Nest_Dec2024", "date": "2024-12-13"}
    ]

AGGREGATION_PERIOD = "W-MON"
SENTIMENT_AGGREGATION_METRIC = 'mean_score'
USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE = True
USE_OVERALL_SENTIMENT_AS_COVARIATE = True

def load_and_prepare_absa_data(assistant_name_func, data_dir_func, agg_period_func, sentiment_metric_func, aspect_filter=None):
    input_filename = ABSA_FILE_MAPPING.get(assistant_name_func)
    if not input_filename:
        print(f"Error: No ABSA file mapping found for assistant '{assistant_name_func}'")
        return None
    input_file = data_dir_func / input_filename
    
    if not input_file.exists():
        print(f"Error: ABSA input file not found at {input_file}")
        return None
    
    print(f"Loading ABSA data from {input_file} for {assistant_name_func}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading CSV {input_file}: {e}")
        return None

    required_cols = ['timestamp', 'identified_aspect', 'aspect_sentiment'] 
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in {input_file}.")
        print(f"Available columns: {df.columns.tolist()}")
        return None


    sentiment_mapping = {
        'Positive': 1,
        'Neutral': 0,
        'Negative': -1
        # Add other mappings if needed, e.g., 'Very Positive': 2
    }
    
    # Create a copy to avoid SettingWithCopyWarning if df is a slice
    df = df.copy()
    
    # Apply mapping. Unmapped values will become NaN.
    # Ensure 'aspect_sentiment' is string type before mapping if it's mixed
    df['aspect_sentiment_numeric'] = df['aspect_sentiment'].astype(str).map(sentiment_mapping)
    
    # Check for any values that didn't map (became NaN but weren't NaN originally)
    # and also for the specific problematic concatenated string.
    unmapped_mask = df['aspect_sentiment_numeric'].isnull() & df['aspect_sentiment'].notnull()
    problematic_strings = df.loc[unmapped_mask, 'aspect_sentiment'].unique()
    if len(problematic_strings) > 0:
        print(f"Warning: Found unmapped or problematic sentiment strings in 'aspect_sentiment' column for {input_file}: {problematic_strings}")
        print("These will be treated as NaN and likely dropped. Review your input CSV for data quality.")


    # Use the new numeric column for aggregation
    sentiment_col_to_aggregate = 'aspect_sentiment_numeric'


    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    if aspect_filter:
        print(f"Filtering for aspect: '{aspect_filter}'")
        df_filtered = df[df['identified_aspect'] == aspect_filter]
        if df_filtered.empty:
            print(f"Warning: No data found for aspect '{aspect_filter}' for {assistant_name_func} in {input_file}")
            return None

        print(f"Aggregating {assistant_name_func} - {aspect_filter} sentiment to '{agg_period_func}' frequency using '{sentiment_metric_func}'...")
        if sentiment_metric_func == 'mean_score':
            ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).mean()
        elif sentiment_metric_func == 'prop_positive':
            ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'prop_negative':
            ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).apply(lambda x: (x == -1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'net_sentiment':
            ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).apply(
                lambda x: ((x == 1).sum() - (x == -1).sum()) / len(x) if len(x) > 0 else 0
            )
        else:
            print(f"Error: Unknown sentiment_aggregation_metric: {sentiment_metric_func}")
            return None
            
        ts_data = ts_data.dropna()
        if ts_data.empty:
            print(f"Warning: No data remaining after aggregation for {assistant_name_func} - {aspect_filter}.")
            return None
        else:
            print(f"{assistant_name_func} - {aspect_filter} time series prepared: {len(ts_data)} points, from {ts_data.index.min().date()} to {ts_data.index.max().date()}.")
        aspect_key = aspect_filter.replace(' ', '_').replace('&', 'and').replace('/', '_').lower()
        return ts_data.rename(f"{assistant_name_func}_{aspect_key}_sentiment")
    
    else:
        aspects = df['identified_aspect'].unique()
        print(f"Found aspects for {assistant_name_func}: {aspects}. Aggregating all...")
        aspect_ts_dict = {}
        
        for current_aspect in aspects:
            df_filtered = df[df['identified_aspect'] == current_aspect]
            if df_filtered.empty:
                print(f"Warning: No data for aspect '{current_aspect}' during iteration for {assistant_name_func}")
                continue

            if sentiment_metric_func == 'mean_score':
                ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).mean()
            elif sentiment_metric_func == 'prop_positive':
                ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
            elif sentiment_metric_func == 'prop_negative':
                ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).apply(lambda x: (x == -1).sum() / len(x) if len(x) > 0 else 0)
            elif sentiment_metric_func == 'net_sentiment':
                ts_data = df_filtered[sentiment_col_to_aggregate].resample(agg_period_func).apply(
                    lambda x: ((x == 1).sum() - (x == -1).sum()) / len(x) if len(x) > 0 else 0
                )
            else:
                print(f"Error: Unknown sentiment_aggregation_metric: {sentiment_metric_func}")
                continue

            ts_data = ts_data.dropna()
            if not ts_data.empty:
                aspect_key_sanitized = current_aspect.replace(' ', '_').replace('&', 'and').replace('/', '_').lower()
                aspect_ts_dict[current_aspect] = ts_data.rename(f"{assistant_name_func}_{aspect_key_sanitized}_sentiment")
                print(f"{assistant_name_func} - {current_aspect} time series prepared: {len(ts_data)} points.")
            else:
                print(f"Warning: No data remaining after aggregation for {assistant_name_func} - {current_aspect}.")
        
        return aspect_ts_dict


# (load_overall_sentiment_data, run_bsts_for_one_aspect, main processing logic)

def load_overall_sentiment_data(assistant_name_func, data_dir_func, agg_period_func, sentiment_metric_func):
    """Loads overall sentiment data for use as covariate.
       This function assumes the overall sentiment CSV has 'at' and 'sentiment_score' columns."""
    input_file = data_dir_func / f"{assistant_name_func}_sentiment.csv" 
    if not input_file.exists():
        print(f"Warning: Overall sentiment file for {assistant_name_func} not found at {input_file} (for covariate use).")
        return None
    
    print(f"Loading overall sentiment data for {assistant_name_func} (covariate) from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        if 'at' not in df.columns or 'sentiment_score' not in df.columns:
            print(f"Error: Overall sentiment file {input_file} missing 'at' or 'sentiment_score' column.")
            return None

        df['at'] = pd.to_datetime(df['at'])
        df = df.set_index('at').sort_index()
        
        # **MODIFICATION: Ensure overall sentiment is also numeric if it comes from string labels**
        # This assumes 'sentiment_score' in the overall file might also be string-based initially
        # If 'sentiment_score' in overall_sentiment.csv is ALREADY numeric (e.g., -1,0,1), this mapping isn't strictly needed for it.
        # However, to be safe and consistent with ABSA data handling:
        sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1} # Or however it's coded
        # If 'sentiment_score' is already numeric, .map won't change it if it's not in keys.
        # A more robust way if it could be mixed:
        if df['sentiment_score'].dtype == 'object': # Check if it's string-like
            df_overall_sentiment_numeric = df['sentiment_score'].astype(str).map(sentiment_mapping)
            if df_overall_sentiment_numeric.isnull().any() and df['sentiment_score'].notnull().any():
                 print(f"Warning: Unmapped sentiment strings in overall sentiment file {input_file}. Check data.")
        else: # Assume it's already numeric
            df_overall_sentiment_numeric = df['sentiment_score']

        if sentiment_metric_func == 'mean_score':
            ts_data = df_overall_sentiment_numeric.resample(agg_period_func).mean()
        elif sentiment_metric_func == 'prop_positive':
            ts_data = df_overall_sentiment_numeric.resample(agg_period_func).apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'prop_negative':
            ts_data = df_overall_sentiment_numeric.resample(agg_period_func).apply(lambda x: (x == -1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'net_sentiment':
            ts_data = df_overall_sentiment_numeric.resample(agg_period_func).apply(
                lambda x: ((x == 1).sum() - (x == -1).sum()) / len(x) if len(x) > 0 else 0
            )
        else:
            print(f"Error: Unknown sentiment_aggregation_metric for overall sentiment: {sentiment_metric_func}")
            return None
            
        ts_data = ts_data.dropna()
        if ts_data.empty:
             print(f"Warning: No overall sentiment data after aggregation for {assistant_name_func}.")
             return None
        return ts_data.rename(f"{assistant_name_func}_overall_sentiment_cov")
    except Exception as e:
        print(f"Error loading overall sentiment for {assistant_name_func}: {e}")
        return None

def run_bsts_for_one_aspect(current_aspect_name, main_aspect_ts, other_assistant_aspect_ts, current_assistant_overall_ts, events_list):
    aspect_dir_name = current_aspect_name.replace(' ', '_').replace('&', 'and').replace('/', '_').lower()
    aspect_specific_output_dir = BSTS_OUTPUT_DIR / aspect_dir_name
    aspect_specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    for event in events_list:
        event_name = event["name"]
        intervention_date_str = event["date"]

        print(f"\n--- Analyzing Event for {ASSISTANT_NAME_TO_ANALYZE.title()} - Aspect '{current_aspect_name}': {event_name} (Date: {intervention_date_str}) ---")

        y = main_aspect_ts.copy()
        impact_data = y.to_frame() 
        covariate_names_used = []
        
        # --- Covariate Alignment Logic (simplified for brevity, use your existing robust logic) ---
        if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE and other_assistant_aspect_ts is not None and not other_assistant_aspect_ts.empty:
            other_assistant_label = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
            cov_name = f'{other_assistant_label}_{aspect_dir_name}_cov'
            current_impact_data_index = impact_data.index
            if not current_impact_data_index.equals(other_assistant_aspect_ts.index):
                common_index = current_impact_data_index.union(other_assistant_aspect_ts.index)
                impact_data = impact_data.reindex(common_index).ffill().bfill()
                aligned_covariate = other_assistant_aspect_ts.reindex(common_index).ffill().bfill()
                impact_data[cov_name] = aligned_covariate.loc[impact_data.index.intersection(aligned_covariate.index)] # Intersect to be safe
            else:
                impact_data[cov_name] = other_assistant_aspect_ts
            if cov_name in impact_data.columns: covariate_names_used.append(cov_name)

        if USE_OVERALL_SENTIMENT_AS_COVARIATE and current_assistant_overall_ts is not None and not current_assistant_overall_ts.empty:
            cov_name = f'{ASSISTANT_NAME_TO_ANALYZE}_overall_sentiment_cov'
            current_impact_data_index = impact_data.index
            if not current_impact_data_index.equals(current_assistant_overall_ts.index):
                common_index = current_impact_data_index.union(current_assistant_overall_ts.index)
                impact_data = impact_data.reindex(common_index).ffill().bfill()
                aligned_covariate = current_assistant_overall_ts.reindex(common_index).ffill().bfill()
                impact_data[cov_name] = aligned_covariate.loc[impact_data.index.intersection(aligned_covariate.index)] # Intersect
            else:
                impact_data[cov_name] = current_assistant_overall_ts
            if cov_name in impact_data.columns: covariate_names_used.append(cov_name)
        
        # Ensure first column is response, rest are covariates for CausalImpact
        response_col_name = main_aspect_ts.name 
        if response_col_name not in impact_data.columns and impact_data.columns[0] == 0 : # Fallback if name was lost
             impact_data.rename(columns={impact_data.columns[0]: response_col_name}, inplace=True)
        
        final_columns_for_ci = [response_col_name] + [c for c in covariate_names_used if c in impact_data.columns and c != response_col_name]
        # Drop rows with NaN that might have been introduced by reindexing covariates with non-overlapping periods
        ci_input_df = impact_data[final_columns_for_ci].dropna() 

        if ci_input_df.empty or response_col_name not in ci_input_df.columns:
            print(f"Skipping event {event_name} for aspect {current_aspect_name} due to empty data after covariate alignment or missing response column.")
            continue
        # --- End of Simplified Covariate Logic Placeholder ---
        
        # --- [Your existing pre-period/post-period definition and checks remain here] ---
        intervention_timestamp = pd.to_datetime(intervention_date_str)

        if not (ci_input_df.index.min() <= intervention_timestamp <= ci_input_df.index.max()):
            print(f"Intervention date {intervention_date_str} is outside the data range ({ci_input_df.index.min().date()} to {ci_input_df.index.max().date()}) for aspect {current_aspect_name}. Skipping event.")
            continue

        pre_period_end_loc = ci_input_df.index.get_indexer([intervention_timestamp], method='ffill')[0]
        if ci_input_df.index[pre_period_end_loc] >= intervention_timestamp and pre_period_end_loc > 0:
            pre_period_end_loc -= 1
        
        post_period_start_loc = ci_input_df.index.get_indexer([intervention_timestamp], method='bfill')[0]

        # Ensure pre_period_end_loc and post_period_start_loc are valid relative to ci_input_df
        if pre_period_end_loc < 0 or post_period_start_loc >= len(ci_input_df.index) or post_period_start_loc <= pre_period_end_loc :
            print(f"Cannot define valid pre/post periods for intervention {intervention_date_str} on aspect {current_aspect_name} after data preparation.")
            print(f"Details: Min data date: {ci_input_df.index.min().date()}, Max data date: {ci_input_df.index.max().date()}, Pre-end index: {pre_period_end_loc}, Post-start index: {post_period_start_loc}, DF length: {len(ci_input_df.index)}")
            print("Skipping.")
            continue
            
        pre_period_start_date = ci_input_df.index[0]
        pre_period_end_date = ci_input_df.index[pre_period_end_loc]
        post_period_start_date = ci_input_df.index[post_period_start_loc]
        post_period_end_date = ci_input_df.index[-1]

        if post_period_start_date <= pre_period_end_date:
            if post_period_start_loc + 1 < len(ci_input_df.index):
                post_period_start_loc += 1
                post_period_start_date = ci_input_df.index[post_period_start_loc]
                if post_period_start_date <= pre_period_end_date:
                    print("Adjustment failed to create valid separation. Skipping.")
                    continue
            else:
                print("Not enough data for distinct post-period after adjustment. Skipping.")
                continue
        
        min_period_points = 8 
        if (pre_period_end_loc + 1) < min_period_points or (len(ci_input_df.index) - post_period_start_loc) < min_period_points:
            print(f"Insufficient data points for robust analysis on aspect {current_aspect_name}. Pre: {pre_period_end_loc + 1}, Post: {len(ci_input_df.index) - post_period_start_loc}. Need {min_period_points}. Skipping event.")
            continue

        pre_period_for_ci = [str(pre_period_start_date.date()), str(pre_period_end_date.date())]
        post_period_for_ci = [str(post_period_start_date.date()), str(post_period_end_date.date())]
        
        if pd.to_datetime(pre_period_for_ci[1]) < pd.to_datetime(pre_period_for_ci[0]) or \
           pd.to_datetime(post_period_for_ci[1]) < pd.to_datetime(post_period_for_ci[0]) or \
           pd.to_datetime(post_period_for_ci[0]) <= pd.to_datetime(pre_period_for_ci[1]):
            print("Invalid period definition. Skipping.")
            continue
        # --- [End of pre/post period definition] ---
        
        try:
            print(f"Running CausalImpact for {event_name} on {ASSISTANT_NAME_TO_ANALYZE} - {current_aspect_name}...")
            print(f"CausalImpact input DF columns: {ci_input_df.columns.tolist()}")
            print(f"CausalImpact input DF shape: {ci_input_df.shape}")
            print(f"Pre-period: {pre_period_for_ci}, Post-period: {post_period_for_ci}")

            ci = CausalImpact(ci_input_df, pre_period_for_ci, post_period_for_ci)

            # --- [Your summary saving logic remains here] ---
            summary_filename = aspect_specific_output_dir / f"{ASSISTANT_NAME_TO_ANALYZE}_{aspect_dir_name}_{event_name}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write(f"Causal Impact Analysis Summary for Event: {event_name}\n")
                f.write(f"Intervention Date: {intervention_date_str}\n")
                f.write(f"Assistant Analyzed: {ASSISTANT_NAME_TO_ANALYZE.upper()}\n")
                f.write(f"Aspect Analyzed: {current_aspect_name}\n")
                f.write(f"Aggregation Period: {AGGREGATION_PERIOD}\n")
                f.write(f"Sentiment Metric: {SENTIMENT_AGGREGATION_METRIC}\n")
                f.write(f"Covariates Used: {', '.join(c for c in final_columns_for_ci[1:]) if len(final_columns_for_ci) > 1 else 'None'}\n")
                f.write("-" * 50 + "\n")
                f.write(ci.summary())
                f.write("\n" + "-" * 50 + "\n")
                f.write("Full Report:\n")
                f.write(ci.summary(output='report'))
            print(f"Summary saved to {summary_filename}")
            # --- [End of summary saving logic] ---

            plot_filename = aspect_specific_output_dir / f"{ASSISTANT_NAME_TO_ANALYZE}_{aspect_dir_name}_{event_name}_plot.png"
            print(f"Generating and saving plot to {plot_filename}...")
            
            custom_plot_successful = False
            try:
                fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
                plot_data = ci.data 
                
                axes[0].plot(plot_data.index, plot_data.iloc[:, 0], 'k-', label='Original (Observed)') 
                axes[0].plot(plot_data.index, plot_data.iloc[:, 1], 'b--', label='Counterfactual Prediction') 
                if plot_data.shape[1] >=4: 
                     axes[0].fill_between(plot_data.index, plot_data.iloc[:, 2], plot_data.iloc[:, 3], color='blue', alpha=0.2, label='95% Prediction Interval')
                axes[0].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[0].legend()
                axes[0].set_title(f'{current_aspect_name}: Observed vs. Counterfactual Prediction')
                
                pointwise = plot_data.iloc[:, 0] - plot_data.iloc[:, 1] 
                axes[1].plot(plot_data.index, pointwise, 'b-')
                if plot_data.shape[1] >=4: 
                    pointwise_lower = plot_data.iloc[:, 0] - plot_data.iloc[:, 3] 
                    pointwise_upper = plot_data.iloc[:, 0] - plot_data.iloc[:, 2] 
                    axes[1].fill_between(plot_data.index, pointwise_lower, pointwise_upper, color='blue', alpha=0.2)
                axes[1].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[1].axhline(y=0, color='gray', linestyle='-')
                axes[1].set_title('Pointwise Effect (Observed - Counterfactual)')
                
                # **MODIFIED CUMULATIVE PLOT SECTION**
                if ci.inferences is not None and \
                   all(col in ci.inferences.columns for col in ['post_cum_effect', 'post_cum_effect_lower', 'post_cum_effect_upper']):
                    cumulative_inferences = ci.inferences[['post_cum_effect', 'post_cum_effect_lower', 'post_cum_effect_upper']]
                    post_period_mask = plot_data.index >= pd.to_datetime(post_period_for_ci[0])
                    
                    axes[2].plot(plot_data.index[post_period_mask], cumulative_inferences['post_cum_effect'][post_period_mask], 'b-')
                    axes[2].fill_between(plot_data.index[post_period_mask], 
                                         cumulative_inferences['post_cum_effect_lower'][post_period_mask], 
                                         cumulative_inferences['post_cum_effect_upper'][post_period_mask], 
                                         color='blue', alpha=0.2)
                    axes[2].set_title('Cumulative Effect with 95% CI (Post-Intervention)')
                else:
                    # Fallback: Plot cumulative effect calculated from pointwise, without CI
                    print(f"Warning: Full cumulative inferences not available for {event_name} - {current_aspect_name}. Plotting cumulative sum without CI.")
                    cumulative_pointwise = pointwise.cumsum()
                    axes[2].plot(plot_data.index, cumulative_pointwise, 'b-')
                    axes[2].set_title('Cumulative Effect (Calculated from Pointwise, No CI)')

                axes[2].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[2].axhline(y=0, color='gray', linestyle='-')

                
                fig.suptitle(f"Causal Impact: {event_name} on {ASSISTANT_NAME_TO_ANALYZE.upper()} - {current_aspect_name} Sentiment ({SENTIMENT_AGGREGATION_METRIC})", 
                             fontsize=14)
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(plot_filename)
                plt.close(fig)
                print(f"Plot saved to {plot_filename}")
                custom_plot_successful = True
                
            except Exception as plot_e:
                print(f"Error during custom plot generation for {event_name} - {current_aspect_name}: {plot_e}")
            
            if not custom_plot_successful: # Try default plot if custom failed
                try:
                    print("Attempting default CausalImpact plot as fallback...")
                    # The `plot` method of CausalImpact object itself returns a matplotlib Figure
                    fig_default = ci.plot(panels=['original', 'pointwise', 'cumulative'], figsize=(15, 12))
                    if fig_default is not None: # Check if a figure object was returned
                        fig_default.suptitle(f"Causal Impact (Default Plot): {event_name} on {ASSISTANT_NAME_TO_ANALYZE.upper()} - {current_aspect_name} Sentiment ({SENTIMENT_AGGREGATION_METRIC})", 
                                         fontsize=14)
                        # Adjust layout for suptitle if needed
                        fig_default.tight_layout(rect=[0, 0.03, 1, 0.96])
                        default_plot_filename = plot_filename.with_name(plot_filename.stem + "_default.png")
                        plt.savefig(default_plot_filename)
                        plt.close(fig_default)
                        print(f"Default plot saved to {default_plot_filename}")
                    else:
                        print(f"Default plot could not be generated for {event_name} - {current_aspect_name} (ci.plot() returned None).")
                except Exception as default_plot_e:
                    print(f"Error during default plot generation: {default_plot_e}")
            
            print(f"Successfully processed event: {event_name} for aspect: {current_aspect_name}")

        except Exception as e:
            print(f"Error during CausalImpact analysis for {event_name} - {current_aspect_name}: {e}")


# --- Main Processing Logic ---
plt.switch_backend('agg') # Ensure this is called before any plotting

print(f"--- Starting ABSA BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} ---")
print(f"Project Root directory: {PROJECT_ROOT_DIR}")  
print(f"Results directory (input for sentiment data): {RESULTS_DIR}")
print(f"ABSA BSTS Output directory (for plots/summaries): {BSTS_OUTPUT_DIR}\n")

main_assistant_absa_data_collection = load_and_prepare_absa_data(
    ASSISTANT_NAME_TO_ANALYZE, 
    RESULTS_DIR, 
    AGGREGATION_PERIOD, 
    SENTIMENT_AGGREGATION_METRIC, 
    ASPECT_TO_ANALYZE 
)

if main_assistant_absa_data_collection is None:
    print(f"Critical Error: No ABSA data could be loaded for {ASSISTANT_NAME_TO_ANALYZE}. Aborting analysis.")
    exit()

aspects_for_loop = []
if ASPECT_TO_ANALYZE:
    if isinstance(main_assistant_absa_data_collection, pd.Series):
        aspects_for_loop = [ASPECT_TO_ANALYZE]
        main_assistant_absa_data_collection = {ASPECT_TO_ANALYZE: main_assistant_absa_data_collection}
else:
    if isinstance(main_assistant_absa_data_collection, dict):
        aspects_for_loop = list(main_assistant_absa_data_collection.keys())

other_assistant_name = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
other_assistant_absa_covariate_collection = None
if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE:
    other_assistant_absa_covariate_collection = load_and_prepare_absa_data(
        other_assistant_name, 
        RESULTS_DIR, 
        AGGREGATION_PERIOD, 
        SENTIMENT_AGGREGATION_METRIC,
        ASPECT_TO_ANALYZE 
    )

current_assistant_overall_covariate = None
if USE_OVERALL_SENTIMENT_AS_COVARIATE:
    current_assistant_overall_covariate = load_overall_sentiment_data(
        ASSISTANT_NAME_TO_ANALYZE, 
        RESULTS_DIR, 
        AGGREGATION_PERIOD, 
        SENTIMENT_AGGREGATION_METRIC
    )

if not aspects_for_loop:
    print(f"No aspects to analyze for {ASSISTANT_NAME_TO_ANALYZE}. Check data loading and ASPECT_TO_ANALYZE setting.")
else:
    print(f"\nStarting BSTS analysis for {len(aspects_for_loop)} aspect(s): {', '.join(aspects_for_loop)}")
    for aspect_name_loop in aspects_for_loop:
        print(f"\n==================== Processing Aspect: {aspect_name_loop} ====================")
        
        current_main_aspect_ts = main_assistant_absa_data_collection.get(aspect_name_loop)
        if current_main_aspect_ts is None or current_main_aspect_ts.empty:
            print(f"Skipping aspect '{aspect_name_loop}' for {ASSISTANT_NAME_TO_ANALYZE} - no main time series data.")
            continue

        current_other_assistant_aspect_ts_cov = None
        if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE and other_assistant_absa_covariate_collection:
            if ASPECT_TO_ANALYZE: 
                 current_other_assistant_aspect_ts_cov = other_assistant_absa_covariate_collection if isinstance(other_assistant_absa_covariate_collection, pd.Series) else None
            else: 
                 current_other_assistant_aspect_ts_cov = other_assistant_absa_covariate_collection.get(aspect_name_loop) if isinstance(other_assistant_absa_covariate_collection, dict) else None
            
            if current_other_assistant_aspect_ts_cov is None or current_other_assistant_aspect_ts_cov.empty :
                 print(f"Warning: No covariate data for aspect '{aspect_name_loop}' from {other_assistant_name}.")

        run_bsts_for_one_aspect(
            aspect_name_loop, 
            current_main_aspect_ts, 
            current_other_assistant_aspect_ts_cov, 
            current_assistant_overall_covariate, 
            INTERVENTION_EVENTS
        )

print(f"\n--- ABSA BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} Complete ---")
print(f"Results saved to: {BSTS_OUTPUT_DIR}")