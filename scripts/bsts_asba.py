import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- 1. Configuration ---
# Set these variables to control which analysis to run.

# Choose the primary assistant to analyze
ASSISTANT_NAME_TO_ANALYZE = "alexa"  # Options: "google" or "alexa"

# To run for specific aspects, provide a list: e.g., ["Usability & Interface", "Privacy & Security"]
# To run for ALL aspects found in the data file, set to None
ASPECTS_TO_ANALYZE = ["Privacy & Security"]

# Set to True to use the competing assistant's same aspect as a control variable
USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE = True
# Set to True to use the main assistant's own overall sentiment as a control variable
USE_OVERALL_SENTIMENT_AS_COVARIATE = True

# --- Path and File Setup ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_DIR = SCRIPT_DIR.parent
except NameError:
    # Fallback for interactive environments (like Colab or Jupyter).
    # IMPORTANT: Adjust this path to your main thesis folder.
    PROJECT_ROOT_DIR = Path("/Users/viltetverijonaite/Desktop/MSC/THESIS/thesis/")

RESULTS_DIR = PROJECT_ROOT_DIR / "results"
BSTS_OUTPUT_DIR = RESULTS_DIR / "bsts_outputs" / "absa_sentiment"
BSTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- ABSA Data File Mapping ---
ABSA_FILE_MAPPING = {
    "alexa": "absa_full_results_colab/alexa_full_absa_sentiments_colab.csv",
    "google": "absa_full_results_colab/google_full_absa_sentiments_colab_COMBINED.csv"
}

# --- Intervention Events ---
ALEXA_EVENTS = [
    {"name": "Alexa_Hunches_Introduction_Sep2018", "date": "2018-09-20"},
    {"name": "Alexa_Wolfram_Alpha_Integration_Dec2018", "date": "2018-12-20"},
    {"name": "Alexa_Privacy_Hub_Launched_Sep2019", "date": "2019-09-25"},
    {"name": "Alexa_Reminders_Across_Devices_Jun2020", "date": "2020-06-15"},
    {"name": "Alexa_Proactive_Hunches_Guard_Plus_Jan2021", "date": "2021-01-25"},
    {"name": "Alexa_Smarter_Alexa_New_Echo_Show_Sep2023", "date": "2023-09-20"},
    {"name": "Alexa_Plus_Launch_Feb2025", "date": "2025-02-26"}
]
GOOGLE_EVENTS = [
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
# --- End of Configuration ---

def load_and_prepare_absa_data(assistant_name, data_dir, agg_period, aspect_list_filter=None):
    """Loads and processes the ABSA sentiment data from a CSV file."""
    input_filename = ABSA_FILE_MAPPING.get(assistant_name)
    if not input_filename:
        print(f"Error: No ABSA file mapping for '{assistant_name}'")
        return None
    input_file = data_dir / input_filename

    if not input_file.exists():
        print(f"Error: ABSA input file not found at {input_file}")
        return None

    print(f"Loading ABSA data from {input_file} for {assistant_name}...")
    df = pd.read_csv(input_file, low_memory=False)
    df = df.copy()

    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_score'] = df['aspect_sentiment'].astype(str).map(sentiment_mapping)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    aspects_to_process = aspect_list_filter if aspect_list_filter else df['identified_aspect'].unique()
    aspect_ts_dict = {}

    for aspect in aspects_to_process:
        df_filtered = df[df['identified_aspect'] == aspect]
        if not df_filtered.empty:
            ts_data = df_filtered['sentiment_score'].resample(agg_period).mean().dropna()
            if not ts_data.empty:
                aspect_ts_dict[aspect] = ts_data

    return aspect_ts_dict

def load_overall_sentiment_data(assistant_name, data_dir, agg_period):
    """Loads and processes the overall sentiment data for use as a covariate."""
    input_file = data_dir / f"{assistant_name}_sentiment.csv"
    if not input_file.exists():
        print(f"Warning: Overall sentiment file not found for {assistant_name} at {input_file}")
        return None
    df = pd.read_csv(input_file)
    df['at'] = pd.to_datetime(df['at'])
    df = df.set_index('at').sort_index()
    return df['sentiment_score'].resample(agg_period).mean().dropna()

def run_bsts_for_one_aspect(assistant_name, current_aspect_name, main_aspect_ts, events_list, other_assistant_aspect_ts=None, overall_sentiment_ts=None):
    """Performs the Causal Impact analysis and plotting for a single aspect and all its events."""
    aspect_dir_name = current_aspect_name.replace(' & ', '_and_').replace('/', '_').lower()
    aspect_specific_output_dir = BSTS_OUTPUT_DIR / aspect_dir_name
    aspect_specific_output_dir.mkdir(parents=True, exist_ok=True)

    for event in events_list:
        event_name = event["name"]
        print(f"\n--- Analyzing: {assistant_name.upper()} - {current_aspect_name} - {event_name} ---")

        # Prepare data for CausalImpact
        impact_data = {"y": main_aspect_ts}
        if USE_OTHER_ASSISTANT_ASPECT_AS_COVARIATE and other_assistant_aspect_ts is not None and not other_assistant_aspect_ts.empty:
            impact_data["X1_competing_aspect"] = other_assistant_aspect_ts
        if USE_OVERALL_SENTIMENT_AS_COVARIATE and overall_sentiment_ts is not None and not overall_sentiment_ts.empty:
            impact_data["X2_overall_sentiment"] = overall_sentiment_ts
        ci_input_df = pd.DataFrame(impact_data).ffill().bfill().dropna()

        if ci_input_df.empty or "y" not in ci_input_df.columns:
            print("Skipping event due to empty data after covariate alignment.")
            continue

        try:
            intervention_timestamp = pd.to_datetime(event["date"])
            if not (ci_input_df.index.min() <= intervention_timestamp <= ci_input_df.index.max()):
                print("Intervention date out of range. Skipping.")
                continue

            pre_period_end_date = ci_input_df.index[ci_input_df.index < intervention_timestamp][-1]
            post_period_start_date = ci_input_df.index[ci_input_df.index >= intervention_timestamp][0]
            pre_period = [str(ci_input_df.index.min().date()), str(pre_period_end_date.date())]
            post_period = [str(post_period_start_date.date()), str(ci_input_df.index.max().date())]

            print("Running CausalImpact...")
            ci = CausalImpact(ci_input_df, pre_period, post_period)

            output_prefix = f"{assistant_name}_{aspect_dir_name}_{event_name}"
            
            # Save summary
            summary_filename = aspect_specific_output_dir / f"{output_prefix}_summary.txt"
            with open(summary_filename, "w") as f:
                f.write(ci.summary() + "\n\n" + ci.summary(output='report'))
            print(f"Summary saved to {summary_filename}")

            # ### FULLY CORRECTED PLOTTING LOGIC ###
            plot_filename = aspect_specific_output_dir / f"{output_prefix}_plot.png"
            print(f"Generating plot and saving to {plot_filename}...")

            ci.plot(figsize=(15, 12))
            fig = plt.gcf() # Get current figure generated by ci.plot()
            fig.suptitle(f"Causal Impact: {event_name}\non {assistant_name.upper()} - {current_aspect_name} Sentiment", fontsize=14, y=0.99)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            plt.savefig(plot_filename)
            plt.close(fig) # Close the figure to free up memory
            print(f"Plot saved to {plot_filename}")
            
        except Exception as e:
            print(f"An error occurred during analysis for {event_name}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    plt.switch_backend('agg')
    print(f"--- Starting ABSA BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} ---")

    # Load all aspect data for the main assistant
    main_assistant_data_dict = load_and_prepare_absa_data(ASSISTANT_NAME_TO_ANALYZE, RESULTS_DIR, AGGREGATION_PERIOD, ASPECTS_TO_ANALYZE)
    if not main_assistant_data_dict:
        exit(f"Critical Error: No ABSA data loaded for {ASSISTANT_NAME_TO_ANALYZE}.")

    # Load covariate data
    other_assistant_name = "google" if ASSISTANT_NAME_TO_ANALYZE == "alexa" else "alexa"
    other_assistant_data_dict = load_and_prepare_absa_data(other_assistant_name, RESULTS_DIR, AGGREGATION_PERIOD)
    overall_sentiment_covariate = load_overall_sentiment_data(ASSISTANT_NAME_TO_ANALYZE, RESULTS_DIR, AGGREGATION_PERIOD)

    events = GOOGLE_EVENTS if ASSISTANT_NAME_TO_ANALYZE == "google" else ALEXA_EVENTS
    
    # Loop through each aspect and run the analysis
    for aspect, main_ts in main_assistant_data_dict.items():
        print(f"\n==================== Processing Aspect: {aspect} ====================")
        
        other_aspect_ts = other_assistant_data_dict.get(aspect) if other_assistant_data_dict else None
        
        run_bsts_for_one_aspect(
            ASSISTANT_NAME_TO_ANALYZE,
            aspect,
            main_ts,
            events,
            other_assistant_aspect_ts=other_aspect_ts,
            overall_sentiment_ts=overall_sentiment_covariate
        )

    print(f"\n--- ABSA BSTS Analysis for {ASSISTANT_NAME_TO_ANALYZE.upper()} Complete ---")