import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- 1. Configuration ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd() # Fallback for interactive environments

RESULTS_DIR = BASE_DIR / "results"
ALEXA_BSTS_OUTPUT_DIR = RESULTS_DIR / "bsts_outputs" / "alexa"
ALEXA_BSTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSISTANT_NAME = "alexa"

# --- Alexa's Intervention Events (from Thesis Figure 1, within approx. Oct 2017 - Mar 2025 data range) ---
# Please VERIFY these dates against your primary sources.
# If a date is an estimate (e.g., mid-month), try to find the exact announcement/rollout day.
ALEXA_INTERVENTION_EVENTS = [
    {
        # Original: October 2018 (Liao, 2018 -> The Verge, Sept 20, 2018 announcement)
        "name": "Alexa_Hunches_Introduction_Sep2018",
        "date": "2018-09-20"
    },
    {
        # Original: December 2018 (Heater, 2018 -> TechCrunch, Dec 20, 2018)
        "name": "Alexa_Wolfram_Alpha_Integration_Dec2018",
        "date": "2018-12-20"
    },
    {
        # Original: September 2019 (Amazon, n.d. -> Amazon announcement Sept 25, 2019)
        "name": "Alexa_Privacy_Hub_Launched_Sep2019",
        "date": "2019-09-25"
    },
    {
        # Original: June 2020 (Roland-C, 2020a) - Table says June 2020.
        # Using mid-month as placeholder; try to find a more precise rollout date if possible.
        "name": "Alexa_Reminders_Across_Devices_Jun2020",
        "date": "2020-06-15" # Placeholder: Verify exact date
    },
    {
        # Original: January 2021 (Campbell, 2021 -> The Verge, Jan 25, 2021)
        "name": "Alexa_Proactive_Hunches_Guard_Plus_Jan2021",
        "date": "2021-01-25"
    },
    {
        # Original: September 2023 (Chen, 2023 -> aboutamazon.com, Sept 20, 2023)
        "name": "Alexa_Smarter_Alexa_New_Echo_Show_Sep2023",
        "date": "2023-09-20"
    },
    {
        # Original: February 2025 (Panay, 2025 -> aboutamazon.com, Feb 26, 2025)
        # Note: Data collection ends March 2025. Post-period will be very short.
        "name": "Alexa_Plus_Launch_Feb2025",
        "date": "2025-02-26"
    }
    # Add any other relevant Alexa-specific events here
]

# Time series aggregation period
AGGREGATION_PERIOD = "W-MON"  # Weekly, starting Mondays

# Sentiment metric for aggregation
SENTIMENT_AGGREGATION_METRIC = 'mean_score' # Options: 'mean_score', 'prop_positive', 'prop_negative', 'net_sentiment'

# Use Google Assistant's sentiment as a covariate?
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


# --- Main Processing Logic ---
print(f"--- Starting BSTS Analysis for {ASSISTANT_NAME.upper()} ---")
print(f"Base directory: {BASE_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Alexa BSTS Output directory: {ALEXA_BSTS_OUTPUT_DIR}\n")

alexa_ts_data_full = load_and_prepare_data(ASSISTANT_NAME, RESULTS_DIR, AGGREGATION_PERIOD, SENTIMENT_AGGREGATION_METRIC)
google_ts_data_full = None

if USE_OTHER_ASSISTANT_AS_COVARIATE:
    google_ts_data_full = load_and_prepare_data("google", RESULTS_DIR, AGGREGATION_PERIOD, SENTIMENT_AGGREGATION_METRIC)

if alexa_ts_data_full is None or alexa_ts_data_full.empty:
    print(f"Critical Error: Alexa time series data could not be loaded or is empty. Aborting analysis.")
else:
    for event in ALEXA_INTERVENTION_EVENTS:
        event_name = event["name"]
        intervention_date_str = event["date"]

        print(f"\n--- Analyzing Event for Alexa: {event_name} (Date: {intervention_date_str}) ---")

        y = alexa_ts_data_full.copy()
        impact_data = y.to_frame()
        covariate_names = []

        if USE_OTHER_ASSISTANT_AS_COVARIATE and google_ts_data_full is not None and not google_ts_data_full.empty:
            # Align Google's data with Alexa's timeline using reindex and fill missing values
            # Ensure indices are compatible.
            if not y.index.equals(google_ts_data_full.index):
                 print("Aligning covariate index with outcome variable index...")
                 # Create a union of both indices to ensure full coverage
                 # Then reindex both series to this union index before further processing
                 common_index = y.index.union(google_ts_data_full.index)
                 y_aligned = y.reindex(common_index).ffill().bfill() # Reindex y first
                 impact_data = y_aligned.to_frame() # Use aligned y for impact_data
                 aligned_covariate = google_ts_data_full.reindex(common_index).ffill().bfill()
                 # Only keep covariate data for the range of y_aligned
                 aligned_covariate = aligned_covariate[y_aligned.index]
            else: # Indices are already the same
                 aligned_covariate = google_ts_data_full
                 impact_data = y.to_frame() # y is already correctly indexed

            impact_data['google_sentiment_covariate'] = aligned_covariate
            covariate_names.append('google_sentiment_covariate')
            print("Using Google sentiment as a covariate.")
        elif USE_OTHER_ASSISTANT_AS_COVARIATE:
             print("Google sentiment data not available or empty. Proceeding without covariate for this event.")


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
            # The impact_data DataFrame should contain the outcome variable as the first column,
            # and any covariates as subsequent columns.
            # If impact_data only has one column (y), it works fine.
            ci = CausalImpact(impact_data, pre_period_for_ci, post_period_for_ci)

            summary_filename = ALEXA_BSTS_OUTPUT_DIR / f"{event_name}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write(f"Causal Impact Analysis Summary for Event: {event_name}\n")
                f.write(f"Intervention Date: {intervention_date_str}\n")
                f.write(f"Assistant Analyzed: {ASSISTANT_NAME.upper()}\n")
                f.write(f"Aggregation Period: {AGGREGATION_PERIOD}\n")
                f.write(f"Sentiment Metric: {SENTIMENT_AGGREGATION_METRIC}\n")
                f.write(f"Covariates Used: {', '.join(covariate_names) if covariate_names else 'None'}\n")
                f.write("-" * 50 + "\n")
                f.write(ci.summary())
                f.write("\n" + "-" * 50 + "\n")
                f.write("Full Report:\n")
                f.write(ci.summary(output='report'))
            print(f"Summary saved to {summary_filename}")

            plot_filename = ALEXA_BSTS_OUTPUT_DIR / f"{event_name}_plot.png"
            fig = ci.plot(figsize=(15, 12))
            fig.suptitle(f"Causal Impact: {event_name} on {ASSISTANT_NAME.upper()} Sentiment ({SENTIMENT_AGGREGATION_METRIC})", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.close(fig)

            print(f"Successfully processed event: {event_name}")

        except Exception as e:
            print(f"Error during CausalImpact analysis for {event_name}: {e}")
            print("This could be due to various issues like insufficient data, model non-convergence, or data properties.")

print(f"\n--- Alexa BSTS analysis complete. Check the '{ALEXA_BSTS_OUTPUT_DIR}' folder. ---")
