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
ASPECT_TO_ANALYZE = None

# --- ABSA Data File Mapping ---
# Ensure these filenames are correct and exist in your RESULTS_DIR
ABSA_FILE_MAPPING = {
    "alexa": "alexa_full_absa_sentiments_colab.csv",
    "google": "google_full_absa_sentiments_colab_COMBINED.csv"
}

# --- Intervention Events ---
if ASSISTANT_NAME_TO_ANALYZE == "alexa":
    INTERVENTION_EVENTS = [
        {"name": "Alexa_Hunches_Introduction_Sep2018", "date": "2018-09-20"},
        {"name": "Alexa_Wolfram_Alpha_Integration_Dec2018", "date": "2018-12-20"},
        {"name": "Alexa_Privacy_Hub_Launched_Sep2019", "date": "2019-09-25"},
        {"name": "Alexa_Reminders_Across_Devices_Jun2020", "date": "2020-06-15"}, # Verify exact date
        {"name": "Alexa_Proactive_Hunches_Guard_Plus_Jan2021", "date": "2021-01-25"},
        {"name": "Alexa_Smarter_Alexa_New_Echo_Show_Sep2023", "date": "2023-09-20"},
        {"name": "Alexa_Plus_Launch_Feb2025", "date": "2025-02-26"} # Note: Data ends Mar 2025
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

# Time series aggregation period
AGGREGATION_PERIOD = "W-MON"  # Weekly, starting Mondays

# Sentiment metric for aggregation (ensure aspect_sentiment is coded appropriately, e.g., 1, 0, -1)
SENTIMENT_AGGREGATION_METRIC = 'mean_score' # Options: 'mean_score', 'prop_positive', 'prop_negative', 'net_sentiment'

# Use the other assistant's SAME aspect sentiment as a covariate?
USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE = True

# Use the CURRENT assistant's OVERALL sentiment as a covariate?
USE_OVERALL_SENTIMENT_AS_COVARIATE = True
# --- End of Configuration ---

def load_and_prepare_absa_data(assistant_name_func, data_dir_func, agg_period_func, sentiment_metric_func, aspect_filter=None):
    """Loads ABSA sentiment data, filters by aspect, converts to time series, and aggregates."""
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

    # Your specified columns: reviewId,sentence_text,identified_aspect,matched_keyword,timestamp,aspect_sentiment
    required_cols = ['timestamp', 'identified_aspect', 'aspect_sentiment'] 
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in {input_file}.")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    # Standardize aspect names for filtering and dictionary keys (e.g. from thesis Appendix A)
    # Example: 'Functionality & Performance'
    # This step depends on how aspects are named in your 'identified_aspect' column.
    # If they already match the desired names (like those in ASPECT_TO_ANALYZE or thesis), this might not be needed.
    # df['identified_aspect'] = df['identified_aspect'].str.strip() # Example cleanup

    if aspect_filter:
        # Analyzing a single, specified aspect
        print(f"Filtering for aspect: '{aspect_filter}'")
        df_filtered = df[df['identified_aspect'] == aspect_filter]
        if df_filtered.empty:
            print(f"Warning: No data found for aspect '{aspect_filter}' for {assistant_name_func} in {input_file}")
            return None

        print(f"Aggregating {assistant_name_func} - {aspect_filter} sentiment to '{agg_period_func}' frequency using '{sentiment_metric_func}'...")
        if sentiment_metric_func == 'mean_score':
            ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).mean()
        elif sentiment_metric_func == 'prop_positive': # Assumes aspect_sentiment is 1 for positive
            ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'prop_negative': # Assumes aspect_sentiment is -1 for negative
            ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).apply(lambda x: (x == -1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'net_sentiment': # Assumes 1 for positive, -1 for negative
            ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).apply(
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
        # Sanitize aspect_filter for use in series name
        aspect_key = aspect_filter.replace(' ', '_').replace('&', 'and').replace('/', '_').lower()
        return ts_data.rename(f"{assistant_name_func}_{aspect_key}_sentiment")
    
    else:
        # Analyzing all aspects found in the file
        aspects = df['identified_aspect'].unique()
        print(f"Found aspects for {assistant_name_func}: {aspects}. Aggregating all...")
        aspect_ts_dict = {}
        
        for current_aspect in aspects:
            df_filtered = df[df['identified_aspect'] == current_aspect]
            if df_filtered.empty:
                print(f"Warning: No data for aspect '{current_aspect}' during iteration for {assistant_name_func}")
                continue

            if sentiment_metric_func == 'mean_score':
                ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).mean()
            elif sentiment_metric_func == 'prop_positive':
                ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
            elif sentiment_metric_func == 'prop_negative':
                ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).apply(lambda x: (x == -1).sum() / len(x) if len(x) > 0 else 0)
            elif sentiment_metric_func == 'net_sentiment':
                ts_data = df_filtered['aspect_sentiment'].resample(agg_period_func).apply(
                    lambda x: ((x == 1).sum() - (x == -1).sum()) / len(x) if len(x) > 0 else 0
                )
            else:
                print(f"Error: Unknown sentiment_aggregation_metric: {sentiment_metric_func}")
                # Skip this aspect if metric is unknown
                continue

            ts_data = ts_data.dropna()
            if not ts_data.empty:
                # Sanitize current_aspect for use in dictionary key and series name
                aspect_key_sanitized = current_aspect.replace(' ', '_').replace('&', 'and').replace('/', '_').lower()
                aspect_ts_dict[current_aspect] = ts_data.rename(f"{assistant_name_func}_{aspect_key_sanitized}_sentiment")
                print(f"{assistant_name_func} - {current_aspect} time series prepared: {len(ts_data)} points.")
            else:
                print(f"Warning: No data remaining after aggregation for {assistant_name_func} - {current_aspect}.")
        
        return aspect_ts_dict


def load_overall_sentiment_data(assistant_name_func, data_dir_func, agg_period_func, sentiment_metric_func):
    """Loads overall sentiment data for use as covariate.
       This function assumes the overall sentiment CSV has 'at' and 'sentiment_score' columns."""
    # Assuming overall sentiment files are named like 'alexa_sentiment.csv' or 'google_sentiment.csv'
    # This part is from your previous script for overall sentiment.
    input_file = data_dir_func / f"{assistant_name_func}_sentiment.csv" # Make sure this file name is correct
    if not input_file.exists():
        print(f"Warning: Overall sentiment file for {assistant_name_func} not found at {input_file} (for covariate use).")
        return None
    
    print(f"Loading overall sentiment data for {assistant_name_func} (covariate) from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        # Ensure 'at' and 'sentiment_score' columns exist for overall sentiment file
        if 'at' not in df.columns or 'sentiment_score' not in df.columns:
            print(f"Error: Overall sentiment file {input_file} missing 'at' or 'sentiment_score' column.")
            return None

        df['at'] = pd.to_datetime(df['at'])
        df = df.set_index('at').sort_index()
        
        if sentiment_metric_func == 'mean_score':
            ts_data = df['sentiment_score'].resample(agg_period_func).mean()
        elif sentiment_metric_func == 'prop_positive':
            ts_data = df['sentiment_score'].resample(agg_period_func).apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'prop_negative':
            ts_data = df['sentiment_score'].resample(agg_period_func).apply(lambda x: (x == -1).sum() / len(x) if len(x) > 0 else 0)
        elif sentiment_metric_func == 'net_sentiment':
            ts_data = df['sentiment_score'].resample(agg_period_func).apply(
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
    """Run BSTS analysis for a single, specific aspect."""
    
    # Sanitize aspect name for directory creation
    aspect_dir_name = current_aspect_name.replace(' ', '_').replace('&', 'and').replace('/', '_').lower()
    aspect_specific_output_dir = BSTS_OUTPUT_DIR / aspect_dir_name
    aspect_specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    for event in events_list:
        event_name = event["name"]
        intervention_date_str = event["date"]

        print(f"\n--- Analyzing Event for {ASSISTANT_NAME_TO_ANALYZE.title()} - Aspect '{current_aspect_name}': {event_name} (Date: {intervention_date_str}) ---")

        y = main_aspect_ts.copy()
        impact_data = y.to_frame() # First column is the response variable
        covariate_names_used = []
        
        # Temporary y for alignment if covariates are added
        y_for_alignment = y 

        # 1. Add other assistant's SAME aspect sentiment as covariate
        if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE and other_assistant_aspect_ts is not None and not other_assistant_aspect_ts.empty:
            other_assistant_label = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
            cov_name = f'{other_assistant_label}_{aspect_dir_name}_cov'
            
            if not y_for_alignment.index.equals(other_assistant_aspect_ts.index):
                print(f"Aligning {other_assistant_label} '{current_aspect_name}' covariate index...")
                common_index = y_for_alignment.index.union(other_assistant_aspect_ts.index)
                # Reindex y_for_alignment if it's the first covariate added to impact_data
                if impact_data.shape[1] == 1: # Only response variable present so far
                     impact_data = y_for_alignment.reindex(common_index).ffill().bfill().to_frame()
                else: # impact_data already has y (and maybe other covariates) on common_index
                     impact_data = impact_data.reindex(common_index).ffill().bfill()


                aligned_covariate = other_assistant_aspect_ts.reindex(common_index).ffill().bfill()
                # Ensure covariate only covers the range of the (potentially reindexed) response
                impact_data[cov_name] = aligned_covariate[impact_data.index.min():impact_data.index.max()]
            else:
                impact_data[cov_name] = other_assistant_aspect_ts
            
            covariate_names_used.append(cov_name)
            print(f"Using {other_assistant_label} '{current_aspect_name}' sentiment as a covariate.")
            y_for_alignment = impact_data.iloc[:,0] # Update y_for_alignment to the current index of impact_data's first column

        # 2. Add CURRENT assistant's OVERALL sentiment as covariate
        if USE_OVERALL_SENTIMENT_AS_COVARIATE and current_assistant_overall_ts is not None and not current_assistant_overall_ts.empty:
            cov_name = f'{ASSISTANT_NAME_TO_ANALYZE}_overall_sentiment_cov'
            
            if not y_for_alignment.index.equals(current_assistant_overall_ts.index):
                print(f"Aligning {ASSISTANT_NAME_TO_ANALYZE} overall sentiment covariate index...")
                common_index = y_for_alignment.index.union(current_assistant_overall_ts.index)
                if impact_data.shape[1] == 1:  # Only response variable present so far
                    impact_data = y_for_alignment.reindex(common_index).ffill().bfill().to_frame()
                else: # impact_data already has y (and maybe other covariates) on common_index
                    impact_data = impact_data.reindex(common_index).ffill().bfill()

                aligned_covariate = current_assistant_overall_ts.reindex(common_index).ffill().bfill()
                impact_data[cov_name] = aligned_covariate[impact_data.index.min():impact_data.index.max()]
            else:
                impact_data[cov_name] = current_assistant_overall_ts

            covariate_names_used.append(cov_name)
            print(f"Using {ASSISTANT_NAME_TO_ANALYZE} overall sentiment as a covariate.")
            # y_for_alignment = impact_data.iloc[:,0] # Not strictly needed to update again if this is the last covariate

        # Ensure the first column of impact_data is the one named after main_aspect_ts
        # This happens naturally if y.to_frame() was the start and covariates were appended.
        # If reindexing happened, impact_data.iloc[:,0] is still the response.

        intervention_timestamp = pd.to_datetime(intervention_date_str)

        if not (impact_data.index.min() <= intervention_timestamp <= impact_data.index.max()):
            print(f"Intervention date {intervention_date_str} is outside the data range ({impact_data.index.min().date()} to {impact_data.index.max().date()}) for aspect {current_aspect_name}. Skipping event.")
            continue

        pre_period_end_loc = impact_data.index.get_indexer([intervention_timestamp], method='ffill')[0]
        if impact_data.index[pre_period_end_loc] >= intervention_timestamp and pre_period_end_loc > 0:
            pre_period_end_loc -= 1
        
        post_period_start_loc = impact_data.index.get_indexer([intervention_timestamp], method='bfill')[0]

        if pre_period_end_loc < 0 or post_period_start_loc >= len(impact_data.index) or post_period_start_loc <= pre_period_end_loc:
            print(f"Cannot define valid pre/post periods for intervention {intervention_date_str} on aspect {current_aspect_name}.")
            print(f"Details: Min data date: {impact_data.index.min().date()}, Max data date: {impact_data.index.max().date()}, Pre-end index: {pre_period_end_loc}, Post-start index: {post_period_start_loc}")
            print("Skipping. Check data availability and aggregation.")
            continue

        pre_period_start_date = impact_data.index[0]
        pre_period_end_date = impact_data.index[pre_period_end_loc]
        post_period_start_date = impact_data.index[post_period_start_loc]
        post_period_end_date = impact_data.index[-1]

        if post_period_start_date <= pre_period_end_date:
            print(f"Warning for aspect {current_aspect_name}: Post-period start ({post_period_start_date.date()}) is not sufficiently after pre-period end ({pre_period_end_date.date()}).")
            if post_period_start_loc + 1 < len(impact_data.index):
                post_period_start_loc += 1
                post_period_start_date = impact_data.index[post_period_start_loc]
                print(f"Adjusted post-period start to: {post_period_start_date.date()}")
                if post_period_start_date <= pre_period_end_date:
                    print("Adjustment failed to create valid separation. Skipping event for this aspect.")
                    continue
            else:
                print("Not enough data points for a distinct post-period after adjustment. Skipping event for this aspect.")
                continue
        
        min_period_points = 8 # Increased minimum points for weekly data
        if (pre_period_end_loc + 1) < min_period_points or (len(impact_data.index) - post_period_start_loc) < min_period_points:
            print(f"Insufficient data points for robust analysis on aspect {current_aspect_name}. Pre: {pre_period_end_loc + 1}, Post: {len(impact_data.index) - post_period_start_loc}. Need {min_period_points}. Skipping event.")
            continue

        pre_period_for_ci = [str(pre_period_start_date.date()), str(pre_period_end_date.date())]
        post_period_for_ci = [str(post_period_start_date.date()), str(post_period_end_date.date())]

        print(f"Pre-period for CausalImpact: {pre_period_for_ci}")
        print(f"Post-period for CausalImpact: {post_period_for_ci}")

        if pd.to_datetime(pre_period_for_ci[1]) < pd.to_datetime(pre_period_for_ci[0]) or \
           pd.to_datetime(post_period_for_ci[1]) < pd.to_datetime(post_period_for_ci[0]) or \
           pd.to_datetime(post_period_for_ci[0]) <= pd.to_datetime(pre_period_for_ci[1]):
            print("Invalid period definition (overlap or end before start). Skipping event for this aspect.")
            continue
        
        try:
            print(f"Running CausalImpact for {event_name} on {ASSISTANT_NAME_TO_ANALYZE} - {current_aspect_name}...")
            # Ensure the first column is the response, others are covariates
            ci_input_df = impact_data[ [main_aspect_ts.name] + [col for col in covariate_names_used if col in impact_data.columns] ]

            ci = CausalImpact(ci_input_df, pre_period_for_ci, post_period_for_ci)

            summary_filename = aspect_specific_output_dir / f"{ASSISTANT_NAME_TO_ANALYZE}_{aspect_dir_name}_{event_name}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write(f"Causal Impact Analysis Summary for Event: {event_name}\n")
                f.write(f"Intervention Date: {intervention_date_str}\n")
                f.write(f"Assistant Analyzed: {ASSISTANT_NAME_TO_ANALYZE.upper()}\n")
                f.write(f"Aspect Analyzed: {current_aspect_name}\n")
                f.write(f"Aggregation Period: {AGGREGATION_PERIOD}\n")
                f.write(f"Sentiment Metric: {SENTIMENT_AGGREGATION_METRIC}\n")
                f.write(f"Covariates Used: {', '.join(covariate_names_used) if covariate_names_used else 'None'}\n")
                f.write("-" * 50 + "\n")
                f.write(ci.summary())
                f.write("\n" + "-" * 50 + "\n")
                f.write("Full Report:\n")
                f.write(ci.summary(output='report'))
            print(f"Summary saved to {summary_filename}")

            plot_filename = aspect_specific_output_dir / f"{ASSISTANT_NAME_TO_ANALYZE}_{aspect_dir_name}_{event_name}_plot.png"
            print(f"Generating and saving plot to {plot_filename}...")
            
            try:
                fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
                plot_data = ci.data # Access data from CausalImpact object
                
                axes[0].plot(plot_data.index, plot_data.iloc[:, 0], 'k-', label='Original (Observed)') # Observed
                axes[0].plot(plot_data.index, plot_data.iloc[:, 1], 'b--', label='Counterfactual Prediction') # Predicted
                # Access confidence interval if available and plot it
                if plot_data.shape[1] >=4: # Check if ci.data has CIs (Original, Pred, Pred_lower, Pred_upper)
                     axes[0].fill_between(plot_data.index, plot_data.iloc[:, 2], plot_data.iloc[:, 3], color='blue', alpha=0.2, label='95% Prediction Interval')

                axes[0].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[0].legend()
                axes[0].set_title(f'{current_aspect_name}: Observed vs. Counterfactual Prediction')
                
                pointwise = plot_data.iloc[:, 0] - plot_data.iloc[:, 1] # Observed - Predicted
                axes[1].plot(plot_data.index, pointwise, 'b-')
                if plot_data.shape[1] >=4: # If CIs are present, plot pointwise CIs
                    pointwise_lower = plot_data.iloc[:, 0] - plot_data.iloc[:, 3] # Observed - Pred_upper
                    pointwise_upper = plot_data.iloc[:, 0] - plot_data.iloc[:, 2] # Observed - Pred_lower
                    axes[1].fill_between(plot_data.index, pointwise_lower, pointwise_upper, color='blue', alpha=0.2)

                axes[1].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[1].axhline(y=0, color='gray', linestyle='-')
                axes[1].set_title('Pointwise Effect (Observed - Counterfactual)')
                
                # Use ci.inferences.cum_effect for cumulative effect and its CIs
                cumulative_inferences = ci.inferences[['post_cum_effect', 'post_cum_effect_lower', 'post_cum_effect_upper']]
                # Only plot cumulative effects for the post-period
                post_period_mask = plot_data.index >= pd.to_datetime(post_period_for_ci[0])
                
                axes[2].plot(plot_data.index[post_period_mask], cumulative_inferences['post_cum_effect'][post_period_mask], 'b-')
                axes[2].fill_between(plot_data.index[post_period_mask], 
                                     cumulative_inferences['post_cum_effect_lower'][post_period_mask], 
                                     cumulative_inferences['post_cum_effect_upper'][post_period_mask], 
                                     color='blue', alpha=0.2)

                axes[2].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[2].axhline(y=0, color='gray', linestyle='-')
                axes[2].set_title('Cumulative Effect (Post-Intervention)')
                
                fig.suptitle(f"Causal Impact: {event_name} on {ASSISTANT_NAME_TO_ANALYZE.upper()} - {current_aspect_name} Sentiment ({SENTIMENT_AGGREGATION_METRIC})", 
                             fontsize=14)
                
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(plot_filename)
                plt.close(fig)
                print(f"Plot saved to {plot_filename}")
                
            except Exception as plot_e:
                print(f"Error during custom plot generation for {event_name} - {current_aspect_name}: {plot_e}")
                # Attempt default plot as fallback if manual fails
                try:
                    fig = ci.plot(panels=['original', 'pointwise', 'cumulative'], figsize=(15, 12))
                    fig.suptitle(f"Causal Impact (Default Plot): {event_name} on {ASSISTANT_NAME_TO_ANALYZE.upper()} - {current_aspect_name} Sentiment ({SENTIMENT_AGGREGATION_METRIC})", 
                                 fontsize=14)
                    plt.savefig(plot_filename.with_name(plot_filename.stem + "_default.png"))
                    plt.close(fig)
                    print(f"Default plot saved to {plot_filename.with_name(plot_filename.stem + '_default.png')}")
                except Exception as default_plot_e:
                    print(f"Error during default plot generation: {default_plot_e}")
            
            print(f"Successfully processed event: {event_name} for aspect: {current_aspect_name}")

        except Exception as e:
            print(f"Error during CausalImpact analysis for {event_name} - {current_aspect_name}: {e}")


# Set the Matplotlib backend to a non-interactive one
plt.switch_backend('agg')

# --- Main Processing Logic ---
print(f"--- Starting ABSA BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} ---")
print(f"Project Root directory: {PROJECT_ROOT_DIR}")  
print(f"Results directory (input for sentiment data): {RESULTS_DIR}")
print(f"ABSA BSTS Output directory (for plots/summaries): {BSTS_OUTPUT_DIR}\n")

# --- 1. Load Data for the Main Assistant to be Analyzed ---
main_assistant_absa_data_collection = load_and_prepare_absa_data(
    ASSISTANT_NAME_TO_ANALYZE, 
    RESULTS_DIR, 
    AGGREGATION_PERIOD, 
    SENTIMENT_AGGREGATION_METRIC, 
    ASPECT_TO_ANALYZE # Pass None to load all aspects, or a specific aspect name
)

if main_assistant_absa_data_collection is None:
    print(f"Critical Error: No ABSA data could be loaded for {ASSISTANT_NAME_TO_ANALYZE}. Aborting analysis.")
    exit()

# Determine which aspects to analyze based on what was loaded
if ASPECT_TO_ANALYZE:
    # Single aspect mode: main_assistant_absa_data_collection is a Series if loaded successfully
    aspects_for_loop = [ASPECT_TO_ANALYZE] if isinstance(main_assistant_absa_data_collection, pd.Series) else []
    # Store the single Series in a dictionary-like structure for consistency in the loop
    if aspects_for_loop:
        temp_main_data = {ASPECT_TO_ANALYZE: main_assistant_absa_data_collection}
        main_assistant_absa_data_collection = temp_main_data
else:
    # Multi-aspect mode: main_assistant_absa_data_collection is a dictionary of Series
    aspects_for_loop = list(main_assistant_absa_data_collection.keys())


# --- 2. Load Covariate Data (Other Assistant's Aspect Sentiment) ---
other_assistant_name = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
other_assistant_absa_covariate_collection = None
if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE:
    other_assistant_absa_covariate_collection = load_and_prepare_absa_data(
        other_assistant_name, 
        RESULTS_DIR, 
        AGGREGATION_PERIOD, 
        SENTIMENT_AGGREGATION_METRIC,
        ASPECT_TO_ANALYZE # Load same specific aspect or all for the other assistant
    )

# --- 3. Load Covariate Data (Current Assistant's Overall Sentiment) ---
current_assistant_overall_covariate = None
if USE_OVERALL_SENTIMENT_AS_COVARIATE:
    current_assistant_overall_covariate = load_overall_sentiment_data(
        ASSISTANT_NAME_TO_ANALYZE, 
        RESULTS_DIR, 
        AGGREGATION_PERIOD, 
        SENTIMENT_AGGREGATION_METRIC
    )

# --- 4. Loop Through Aspects and Events for Analysis ---
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

        # Get corresponding covariate data for this aspect
        current_other_assistant_aspect_ts_cov = None
        if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE and other_assistant_absa_covariate_collection:
            if ASPECT_TO_ANALYZE: # Single aspect mode for covariate
                 current_other_assistant_aspect_ts_cov = other_assistant_absa_covariate_collection if isinstance(other_assistant_absa_covariate_collection, pd.Series) else None
            else: # Multi-aspect mode for covariate
                 current_other_assistant_aspect_ts_cov = other_assistant_absa_covariate_collection.get(aspect_name_loop)
            
            if current_other_assistant_aspect_ts_cov is None or current_other_assistant_aspect_ts_cov.empty :
                 print(f"Warning: No covariate data for aspect '{aspect_name_loop}' from {other_assistant_name}.")


        run_bsts_for_one_aspect(
            aspect_name_loop, 
            current_main_aspect_ts, 
            current_other_assistant_aspect_ts_cov, 
            current_assistant_overall_covariate, # This is the overall sentiment, same for all aspects of the current assistant
            INTERVENTION_EVENTS
        )

print(f"\n--- ABSA BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} Complete ---")
print(f"Results saved to: {BSTS_OUTPUT_DIR}")