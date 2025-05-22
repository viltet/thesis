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
    # This might need adjustment if running interactively from an unexpected location
    PROJECT_ROOT_DIR = Path.cwd()
    if PROJECT_ROOT_DIR.name == "scripts": # If cwd is scripts, go up
        PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

RESULTS_DIR = PROJECT_ROOT_DIR / "results"  # Now points to thesis/results/
BSTS_OUTPUT_DIR = RESULTS_DIR / "bsts_outputs" / "overall_sentiment" # This will now be thesis/results/bsts_outputs/overall_sentiment/
BSTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose which assistant to analyze: "alexa" or "google"
ASSISTANT_NAME_TO_ANALYZE = "google"  

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
        # From your thesis draft Figure 2 (ensure dates are as precise as possible)
        # Data collection starts Oct 2017, so May 2016 event is out of range.
        {"name": "Google_Additional_Languages_Routines_Feb2018", "date": "2018-02-07"}, # Source: Devicebase, 2023a
        {"name": "Google_Duplex_Announced_May2018", "date": "2018-05-08"},       # Source: Google, 2018
        {"name": "Google_New_Features_Update_Nov2018", "date": "2018-11-15"},    # Source: Devicebase, 2023b (Using mid-month for Nov)
        {"name": "Google_Interpreter_Mode_Dec2019", "date": "2019-12-12"},       # Source: Byford, 2019
        {"name": "Google_Voice_Match_Expansion_Jun2020", "date": "2020-06-15"},  # Source: Roland-C, 2020b (Using mid-month for June)
        {"name": "Google_Simple_Nest_Hub_Features_Jun2020", "date": "2020-06-20"},# Source: Roland-C, 2020c (Using a slightly later date in June to differentiate)
        {"name": "Google_iOS_Fixes_Stability_Dec2021", "date": "2021-12-15"},    # Source: Roland-C, 2022a (Using mid-month for Dec)
        {"name": "Google_Shortcuts_Plus_Expanded_Support_May2022", "date": "2022-05-03"}, # Source: Devicebase, 2022
        {"name": "Google_Smart_Assistant_Improvements_Mar2023", "date": "2023-03-15"},# Source: Devicebase, 2023c (Using mid-month for Mar)
        {"name": "Google_Gemini_Assistant_Nest_Dec2024", "date": "2024-12-13"}  # Source: Tuohy, 2024
        # Data collection ends March 2025.
    ]

# Time series aggregation period
AGGREGATION_PERIOD = "W-MON"  # Weekly, starting Mondays

# Sentiment metric for aggregation
SENTIMENT_AGGREGATION_METRIC = 'mean_score' # Options: 'mean_score', 'prop_positive', 'prop_negative', 'net_sentiment'

# Use the other assistant's sentiment as a covariate?
USE_OTHER_ASSISTANT_AS_COVARIATE = True
# --- End of Configuration ---

def load_and_prepare_data(assistant_name_func, data_dir_func, agg_period_func, sentiment_metric_func):
    """Loads sentiment data, converts to time series, and aggregates."""
    input_file = data_dir_func / f"{assistant_name_func}_sentiment.csv"
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return None
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading CSV {input_file}: {e}")
        return None

    required_cols = ['at', 'sentiment_score', 'sentiment']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in {input_file}.")
        return None

    df['at'] = pd.to_datetime(df['at'])
    df = df.set_index('at')
    df = df.sort_index()

    print(f"Aggregating {assistant_name_func} sentiment to '{agg_period_func}' frequency using '{sentiment_metric_func}'...")
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
        print(f"Error: Unknown sentiment_aggregation_metric: {sentiment_metric_func}")
        return None

    ts_data = ts_data.dropna()
    if ts_data.empty:
        print(f"Warning: No data remaining after aggregation for {assistant_name_func}. Check input data and date ranges.")
    else:
        print(f"{assistant_name_func} time series prepared: {len(ts_data)} points, from {ts_data.index.min().date()} to {ts_data.index.max().date()}.")
    return ts_data.rename(f"{assistant_name_func}_sentiment")


# Set the Matplotlib backend to a non-interactive one
plt.switch_backend('agg')

# --- Main Processing Logic ---
print(f"--- Starting BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} ---")
print(f"Project Root directory: {PROJECT_ROOT_DIR}") 
print(f"Results directory (input for sentiment data): {RESULTS_DIR}")
print(f"BSTS Output directory (for plots/summaries): {BSTS_OUTPUT_DIR}\n")

# Load data for the assistant being analyzed
main_ts_data_full = load_and_prepare_data(ASSISTANT_NAME_TO_ANALYZE, RESULTS_DIR, AGGREGATION_PERIOD, SENTIMENT_AGGREGATION_METRIC)

# Load data for the other assistant as covariate if requested
other_assistant_name = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
other_ts_data_full = None

if USE_OTHER_ASSISTANT_AS_COVARIATE:
    other_ts_data_full = load_and_prepare_data(other_assistant_name, RESULTS_DIR, AGGREGATION_PERIOD, SENTIMENT_AGGREGATION_METRIC)

if main_ts_data_full is None or main_ts_data_full.empty:
    print(f"Critical Error: {ASSISTANT_NAME_TO_ANALYZE.title()} time series data could not be loaded or is empty. Aborting analysis.")
else:
    for event in INTERVENTION_EVENTS:
        event_name = event["name"]
        intervention_date_str = event["date"]

        print(f"\n--- Analyzing Event for {ASSISTANT_NAME_TO_ANALYZE.title()}: {event_name} (Date: {intervention_date_str}) ---")

        y = main_ts_data_full.copy()
        impact_data = y.to_frame()
        covariate_names = []

        if USE_OTHER_ASSISTANT_AS_COVARIATE and other_ts_data_full is not None and not other_ts_data_full.empty:
            # Align other assistant's data with main assistant's timeline using reindex and fill missing values
            # Ensure indices are compatible.
            if not y.index.equals(other_ts_data_full.index):
                 print(f"Aligning {other_assistant_name} covariate index with {ASSISTANT_NAME_TO_ANALYZE} outcome variable index...")
                 # Create a union of both indices to ensure full coverage
                 # Then reindex both series to this union index before further processing
                 common_index = y.index.union(other_ts_data_full.index)
                 y_aligned = y.reindex(common_index).ffill().bfill() # Reindex y first
                 impact_data = y_aligned.to_frame() # Use aligned y for impact_data
                 aligned_covariate = other_ts_data_full.reindex(common_index).ffill().bfill()
                 # Only keep covariate data for the range of y_aligned
                 aligned_covariate = aligned_covariate[y_aligned.index]
            else: # Indices are already the same
                 aligned_covariate = other_ts_data_full
                 impact_data = y.to_frame() # y is already correctly indexed

            impact_data[f'{other_assistant_name}_sentiment_covariate'] = aligned_covariate
            covariate_names.append(f'{other_assistant_name}_sentiment_covariate')
            print(f"Using {other_assistant_name.title()} sentiment as a covariate.")
        elif USE_OTHER_ASSISTANT_AS_COVARIATE:
             print(f"{other_assistant_name.title()} sentiment data not available or empty. Proceeding without covariate for this event.")

        intervention_timestamp = pd.to_datetime(intervention_date_str)

        if not (impact_data.index.min() <= intervention_timestamp <= impact_data.index.max()):
            print(f"Intervention date {intervention_date_str} is outside the data range ({impact_data.index.min().date()} to {impact_data.index.max().date()}). Skipping event.")
            continue

        pre_period_end_loc = impact_data.index.get_indexer([intervention_timestamp], method='ffill')[0]
        if impact_data.index[pre_period_end_loc] >= intervention_timestamp and pre_period_end_loc > 0:
            pre_period_end_loc -= 1
        
        post_period_start_loc = impact_data.index.get_indexer([intervention_timestamp], method='bfill')[0]

        if pre_period_end_loc < 0 or post_period_start_loc >= len(impact_data.index) or post_period_start_loc <= pre_period_end_loc:
            print(f"Cannot define valid pre/post periods for intervention {intervention_date_str}.")
            print(f"Min data date: {impact_data.index.min().date()}, Max data date: {impact_data.index.max().date()}, Pre-end index: {pre_period_end_loc}, Post-start index: {post_period_start_loc}")
            print("Skipping. Check data range (min 2-3 periods pre/post often needed) and aggregation.")
            continue

        pre_period_start_date = impact_data.index[0]
        pre_period_end_date = impact_data.index[pre_period_end_loc]
        post_period_start_date = impact_data.index[post_period_start_loc]
        post_period_end_date = impact_data.index[-1]

        if post_period_start_date <= pre_period_end_date:
            print(f"Warning: Post-period start ({post_period_start_date.date()}) is not sufficiently after pre-period end ({pre_period_end_date.date()}).")
            if post_period_start_loc + 1 < len(impact_data.index): # Try to advance post period by one step
                post_period_start_loc +=1
                post_period_start_date = impact_data.index[post_period_start_loc]
                print(f"Adjusted post-period start to: {post_period_start_date.date()}")
                if post_period_start_date <= pre_period_end_date:
                    print("Adjustment failed to create valid separation. Skipping event.")
                    continue
            else: # Cannot adjust further
                print("Not enough data points for a distinct post-period after adjustment. Skipping event.")
                continue
        
        # Final check for minimum data points in pre/post period (e.g., at least 3)
        min_period_points = 3
        if (pre_period_end_loc + 1) < min_period_points or (len(impact_data.index) - post_period_start_loc) < min_period_points:
            print(f"Insufficient data points for robust analysis. Pre-period has {pre_period_end_loc + 1}, Post-period has {len(impact_data.index) - post_period_start_loc}. Need at least {min_period_points}. Skipping event.")
            continue

        pre_period_for_ci = [str(pre_period_start_date.date()), str(pre_period_end_date.date())]
        post_period_for_ci = [str(post_period_start_date.date()), str(post_period_end_date.date())]

        print(f"Pre-period for CausalImpact: {pre_period_for_ci}")
        print(f"Post-period for CausalImpact: {post_period_for_ci}")

        if pd.to_datetime(pre_period_for_ci[1]) < pd.to_datetime(pre_period_for_ci[0]) or \
           pd.to_datetime(post_period_for_ci[1]) < pd.to_datetime(post_period_for_ci[0]) or \
           pd.to_datetime(post_period_for_ci[0]) <= pd.to_datetime(pre_period_for_ci[1]):
            print("Invalid period definition (overlap or end before start). Skipping event.")
            continue
        
        try:
            print(f"Running CausalImpact for {event_name}...")
            # Use the DataFrame 'impact_data' which holds your outcome and covariates
            ci = CausalImpact(impact_data, pre_period_for_ci, post_period_for_ci)

            # --- Save Summary FIRST ---
            summary_filename = BSTS_OUTPUT_DIR / f"{ASSISTANT_NAME_TO_ANALYZE}_{event_name}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write(f"Causal Impact Analysis Summary for Event: {event_name}\n")
                f.write(f"Intervention Date: {intervention_date_str}\n")
                f.write(f"Assistant Analyzed: {ASSISTANT_NAME_TO_ANALYZE.upper()}\n")
                f.write(f"Aggregation Period: {AGGREGATION_PERIOD}\n")
                f.write(f"Sentiment Metric: {SENTIMENT_AGGREGATION_METRIC}\n")
                f.write(f"Covariates Used: {', '.join(covariate_names) if covariate_names else 'None'}\n")
                f.write("-" * 50 + "\n")
                f.write(ci.summary())
                f.write("\n" + "-" * 50 + "\n")
                f.write("Full Report:\n")
                f.write(ci.summary(output='report'))
            print(f"Summary saved to {summary_filename}")

            # --- Handle Plotting Manually --- 
            plot_filename = BSTS_OUTPUT_DIR / f"{ASSISTANT_NAME_TO_ANALYZE}_{event_name}_plot.png"
            print(f"Generating and saving plot to {plot_filename}...")
            
            try:
                # Create a new figure with subplots manually
                fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
                
                # Get the data from CausalImpact result
                data = ci.data
                
                # 1. Original and Counterfactual Predictions
                axes[0].plot(data.index, data.iloc[:, 0], 'k-', label='Original')
                axes[0].plot(data.index, data.iloc[:, 1], 'b--', label='Counterfactual Prediction')
                axes[0].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[0].legend()
                axes[0].set_title('Original and Counterfactual Prediction')
                
                # 2. Pointwise Effects
                pointwise = data.iloc[:, 0] - data.iloc[:, 1]
                axes[1].plot(data.index, pointwise, 'b-')
                axes[1].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[1].axhline(y=0, color='gray', linestyle='-')
                axes[1].set_title('Pointwise Effects')
                
                # 3. Cumulative Effect
                cumulative = pointwise.cumsum()
                axes[2].plot(data.index, cumulative, 'b-')
                axes[2].axvline(x=pd.to_datetime(pre_period_for_ci[1]), color='gray', linestyle='--')
                axes[2].axhline(y=0, color='gray', linestyle='-')
                axes[2].set_title('Cumulative Effect')
                
                # Add overall title
                fig.suptitle(f"Causal Impact: {event_name} on {ASSISTANT_NAME_TO_ANALYZE.upper()} Overall Sentiment ({SENTIMENT_AGGREGATION_METRIC})", 
                             fontsize=16)
                
                # Adjust layout and save
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(plot_filename)
                plt.close(fig)
                print(f"Plot saved to {plot_filename}")
                
            except Exception as plot_e:
                print(f"Error during custom plot generation for {event_name}: {plot_e}")
                
            print(f"Successfully processed event: {event_name}")

        except Exception as e:
            print(f"Error during CausalImpact analysis for {event_name}: {e}")

print(f"\n--- BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} Overall Sentiment Complete ---")
print(f"Results saved to: {BSTS_OUTPUT_DIR}")
