import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
from pathlib import Path # Added for robust path handling

# --- 1. Configuration and Path Setup ---
try:
    # Assumes the script might be in a 'scripts' subdirectory of the main 'thesis' folder
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_DIR = SCRIPT_DIR.parent
except NameError:

    PROJECT_ROOT_DIR = Path.cwd()
    if PROJECT_ROOT_DIR.name == "scripts": # If current working directory is 'scripts', go up
        PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent


RESULTS_DIR = PROJECT_ROOT_DIR / "results"
INPUT_DATA_DIR = RESULTS_DIR / "absa_full_results_colab"
OUTPUT_VISUALS_DIR = RESULTS_DIR / "absa_visuals_local_all_aspects" # New output folder name

# Ensure output directory exists
OUTPUT_VISUALS_DIR.mkdir(parents=True, exist_ok=True)

ALEXA_FILE_NAME = "alexa_full_absa_sentiments_colab.csv"
GOOGLE_FILE_NAME = "google_full_absa_sentiments_colab_COMBINED.csv"

ALEXA_FILE_PATH = INPUT_DATA_DIR / ALEXA_FILE_NAME
GOOGLE_FILE_PATH = INPUT_DATA_DIR / GOOGLE_FILE_NAME

# --- 2. Data Loading ---
try:
    alexa_df = pd.read_csv(ALEXA_FILE_PATH)
    google_df = pd.read_csv(GOOGLE_FILE_PATH)
    print(f"Successfully loaded '{ALEXA_FILE_PATH.name}' and '{GOOGLE_FILE_PATH.name}'.")
except FileNotFoundError as e:
    print(f"Error: File not found. {e}.")
    print(f"Please ensure files exist at: \n{ALEXA_FILE_PATH}\n{GOOGLE_FILE_PATH}")
    print(f"And that your project root is correctly identified as: {PROJECT_ROOT_DIR}")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()

# --- 3. Data Preprocessing Function ---
def preprocess_absa_data(df, date_col='timestamp', aspect_col='identified_aspect', sentiment_col='aspect_sentiment'):
    """Preprocesses the ABSA data."""
    processed_df = df.copy()

    try:
        processed_df[date_col] = pd.to_datetime(processed_df[date_col])
    except Exception as e:
        print(f"Error converting date column '{date_col}': {e}. Attempting with errors='coerce'")
        processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
        processed_df.dropna(subset=[date_col], inplace=True)

    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    if processed_df[sentiment_col].dtype == 'object':
        processed_df['sentiment_score'] = processed_df[sentiment_col].map(sentiment_mapping)
    elif pd.api.types.is_numeric_dtype(processed_df[sentiment_col]):
        expected_values = set(sentiment_mapping.values())
        if not set(processed_df[sentiment_col].unique()).issubset(expected_values):
            print(f"Warning: Sentiment column '{sentiment_col}' is numeric but contains values outside of {expected_values}. Please check.")
        processed_df['sentiment_score'] = processed_df[sentiment_col]
    else:
        print(f"Warning: Sentiment column '{sentiment_col}' is not string or recognized numeric. Skipping mapping.")
        processed_df['sentiment_score'] = np.nan

    processed_df.dropna(subset=['sentiment_score'], inplace=True)
    processed_df['Year_Quarter'] = processed_df[date_col].dt.to_period('Q')
    return processed_df

alexa_processed = preprocess_absa_data(alexa_df)
google_processed = preprocess_absa_data(google_df)

print("\nPreprocessing complete.")
print(f"Alexa data shape after preprocessing: {alexa_processed.shape}")
print(f"Google data shape after preprocessing: {google_processed.shape}")

# --- 4. Overall Aspect Sentiment Summary Table ---
def generate_overall_aspect_summary(df, assistant_name, aspect_col='identified_aspect'):
    summary = df.groupby(aspect_col)['sentiment_score'].agg(['mean', 'count']).reset_index()
    summary.columns = ['Aspect', f'{assistant_name}_Mean_Sentiment', f'{assistant_name}_Count']
    return summary.sort_values(by=f'{assistant_name}_Count', ascending=False)

alexa_summary = generate_overall_aspect_summary(alexa_processed, 'Alexa')
google_summary = generate_overall_aspect_summary(google_processed, 'Google')

overall_summary_table = pd.merge(alexa_summary, google_summary, on='Aspect', how='outer')
overall_summary_table = overall_summary_table.sort_values(by=['Alexa_Count', 'Google_Count'], ascending=[False, False])

print("\n--- Overall Aspect Sentiment Summary (Top 15) ---")
print(overall_summary_table.head(15)) # Still prints top 15 for brevity in console output

summary_table_file = OUTPUT_VISUALS_DIR / "overall_aspect_sentiment_summary.csv"
overall_summary_table.to_csv(summary_table_file, index=False)
print(f"\nOverall summary table saved to: {summary_table_file}")

# --- 5. Quarterly Aggregation for Time Series ---
def aggregate_sentiment_quarterly(df, aspect_col='identified_aspect'):
    quarterly_agg = df.groupby(['Year_Quarter', aspect_col])['sentiment_score'].agg(
        mean_sentiment='mean',
        count='count',
        sem='sem'
    ).reset_index()
    return quarterly_agg.sort_values(by=[aspect_col, 'Year_Quarter'])

alexa_quarterly = aggregate_sentiment_quarterly(alexa_processed)
google_quarterly = aggregate_sentiment_quarterly(google_processed)
print("\nQuarterly aggregation complete.")

# --- 6. Line Charts: Evolution of Aspect-Specific Sentiment Scores ---
# Select ALL common aspects for plotting, sorted by their combined total count
common_aspects_series = overall_summary_table.dropna(subset=['Alexa_Count', 'Google_Count'])['Aspect']
selected_aspects_for_plotting = []

if not common_aspects_series.empty:
    overall_summary_table_common = overall_summary_table.dropna(subset=['Alexa_Count', 'Google_Count']).copy()
    overall_summary_table_common['Total_Count'] = overall_summary_table_common['Alexa_Count'] + overall_summary_table_common['Google_Count']
    # MODIFIED LINE: Plot all common aspects, sorted by total count
    selected_aspects_for_plotting = overall_summary_table_common.sort_values('Total_Count', ascending=False)['Aspect'].tolist()
else:
    print("No common aspects found to plot based on current summary logic.")

if not selected_aspects_for_plotting:
    print("Warning: Dynamic selection of aspects for plotting yielded no results. Skipping plots.")

print(f"\nSelected aspects for plotting ({len(selected_aspects_for_plotting)} total): {selected_aspects_for_plotting}")

for aspect in selected_aspects_for_plotting:
    plt.figure(figsize=(14, 7))

    alexa_aspect_data = alexa_quarterly[alexa_quarterly['identified_aspect'] == aspect].sort_values('Year_Quarter')
    google_aspect_data = google_quarterly[google_quarterly['identified_aspect'] == aspect].sort_values('Year_Quarter')

    alexa_aspect_data['plot_date'] = alexa_aspect_data['Year_Quarter'].dt.to_timestamp()
    google_aspect_data['plot_date'] = google_aspect_data['Year_Quarter'].dt.to_timestamp()

    if not alexa_aspect_data.empty:
        plt.plot(alexa_aspect_data['plot_date'], alexa_aspect_data['mean_sentiment'], label=f'Alexa - {aspect}', marker='o', linestyle='-')
        plt.fill_between(alexa_aspect_data['plot_date'],
                         alexa_aspect_data['mean_sentiment'] - 1.96 * alexa_aspect_data['sem'],
                         alexa_aspect_data['mean_sentiment'] + 1.96 * alexa_aspect_data['sem'],
                         alpha=0.2)

    if not google_aspect_data.empty:
        plt.plot(google_aspect_data['plot_date'], google_aspect_data['mean_sentiment'], label=f'Google - {aspect}', marker='x', linestyle='--')
        plt.fill_between(google_aspect_data['plot_date'],
                         google_aspect_data['mean_sentiment'] - 1.96 * google_aspect_data['sem'],
                         google_aspect_data['mean_sentiment'] + 1.96 * google_aspect_data['sem'],
                         alpha=0.2)

    plt.title(f'Quarterly Sentiment Evolution for Aspect: {aspect}', fontsize=16)
    plt.xlabel('Quarter', fontsize=14)
    plt.ylabel('Mean Sentiment Score (-1 to 1)', fontsize=14)
    plt.ylim(-1, 1)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q%q'))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plot_filename = OUTPUT_VISUALS_DIR / f"sentiment_evolution_{aspect.replace(' ', '_').replace('&', 'and').replace('/', '_')}.png" # Ensure slashes are replaced
    plt.savefig(plot_filename)
    print(f"Plot saved to: {plot_filename}")
    plt.close()

print("\n--- Script Finished ---")
print(f"All generated files are in the '{OUTPUT_VISUALS_DIR}' directory.")