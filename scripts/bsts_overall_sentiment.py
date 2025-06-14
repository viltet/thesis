import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- 1. Configuration ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_DIR = SCRIPT_DIR.parent
except NameError:
    PROJECT_ROOT_DIR = Path.cwd()
    if PROJECT_ROOT_DIR.name == "scripts":
        PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

RESULTS_DIR = PROJECT_ROOT_DIR / "results"
BSTS_OUTPUT_DIR = RESULTS_DIR / "bsts_outputs" / "overall_sentiment"
BSTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSISTANT_NAME_TO_ANALYZE = "alexa"

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
USE_OTHER_ASSISTANT_AS_COVARIATE = True
# --- End of Configuration ---

def load_and_prepare_data(assistant_name_func, data_dir_func, agg_period_func, sentiment_metric_func):
    """Loads sentiment data, converts to time series, and aggregates."""
    input_file = data_dir_func / f"{assistant_name_func}_sentiment.csv"
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return None
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['at'] = pd.to_datetime(df['at'])
    df = df.set_index('at').sort_index()

    print(f"Aggregating {assistant_name_func} sentiment to '{agg_period_func}'...")
    ts_data = df['sentiment_score'].resample(agg_period_func).mean().dropna()
    
    print(f"{assistant_name_func} time series prepared: {len(ts_data)} points, from {ts_data.index.min().date()} to {ts_data.index.max().date()}.")
    return ts_data.rename(f"{assistant_name_func}_sentiment")


plt.switch_backend('agg')

# --- Main Processing Logic ---
print(f"--- Starting BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} ---")

main_ts_data_full = load_and_prepare_data(ASSISTANT_NAME_TO_ANALYZE, RESULTS_DIR, AGGREGATION_PERIOD, SENTIMENT_AGGREGATION_METRIC)
other_assistant_name = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
other_ts_data_full = None

if USE_OTHER_ASSISTANT_AS_COVARIATE:
    other_ts_data_full = load_and_prepare_data(other_assistant_name, RESULTS_DIR, AGGREGATION_PERIOD, SENTIMENT_AGGREGATION_METRIC)

if main_ts_data_full is None or main_ts_data_full.empty:
    print(f"Critical Error: {ASSISTANT_NAME_TO_ANALYZE.title()} time series data is empty. Aborting.")
else:
    for event in INTERVENTION_EVENTS:
        event_name = event["name"]
        intervention_date_str = event["date"]
        print(f"\n--- Analyzing Event: {event_name} (Date: {intervention_date_str}) ---")

        impact_data = main_ts_data_full.to_frame()
        if USE_OTHER_ASSISTANT_AS_COVARIATE and other_ts_data_full is not None:
            aligned_df = pd.concat([main_ts_data_full, other_ts_data_full], axis=1, join='outer').ffill().bfill()
            impact_data = aligned_df.loc[main_ts_data_full.index].rename(columns={
                f"{ASSISTANT_NAME_TO_ANALYZE}_sentiment": "y",
                f"{other_assistant_name}_sentiment": "X1"
            })
            print(f"Using {other_assistant_name.title()} sentiment as a covariate.")
        else:
            impact_data = impact_data.rename(columns={f"{ASSISTANT_NAME_TO_ANALYZE}_sentiment": "y"})

        intervention_timestamp = pd.to_datetime(intervention_date_str)
        if not (impact_data.index.min() <= intervention_timestamp <= impact_data.index.max()):
            print("Intervention date out of range. Skipping.")
            continue

        pre_period_end_date = impact_data.index[impact_data.index < intervention_timestamp][-1]
        post_period_start_date = impact_data.index[impact_data.index >= intervention_timestamp][0]
        
        pre_period_for_ci = [str(impact_data.index.min().date()), str(pre_period_end_date.date())]
        post_period_for_ci = [str(post_period_start_date.date()), str(impact_data.index.max().date())]

        print(f"Pre-period: {pre_period_for_ci}")
        print(f"Post-period: {post_period_for_ci}")

        try:
            print(f"Running CausalImpact for {event_name}...")
            ci = CausalImpact(impact_data, pre_period_for_ci, post_period_for_ci)

            summary_filename = BSTS_OUTPUT_DIR / f"{ASSISTANT_NAME_TO_ANALYZE}_{event_name}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write(ci.summary())
                f.write("\n\n" + ci.summary(output='report'))
            print(f"Summary saved to {summary_filename}")

            # --- FINAL PLOTTING LOGIC ---
            plot_filename = BSTS_OUTPUT_DIR / f"{ASSISTANT_NAME_TO_ANALYZE}_{event_name}_plot.png"
            print(f"Generating plot and saving to {plot_filename}...")
            
            # Call the library's plot method, which creates the plot in the background
            ci.plot(figsize=(15, 12))
            
            # Get a reference to the figure that was just created
            fig = plt.gcf()
            
            # Add a custom title and adjust layout
            fig.suptitle(f"Causal Impact: {event_name}", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save and close the figure
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Plot saved to {plot_filename}")
            
        except Exception as e:
            print(f"An error occurred during CausalImpact analysis for {event_name}: {e}")

print(f"\n--- BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} Complete ---")