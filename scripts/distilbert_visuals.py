import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import matplotlib.dates as mdates

# --- Configuration ---
# Base directory for the script (e.g., thesis/scripts)
BASE_DIR = Path(__file__).resolve().parent

# Go up one directory from BASE_DIR (to thesis/) and then into 'results'
THESIS_ROOT = BASE_DIR.parent # This gets you to /Users/viltetverijonaite/Desktop/MSC/THESIS/thesis/

# Input files inside the 'results' folder, relative to the THESIS_ROOT
# These are the OUTPUT files from your FIRST sentiment analysis script
input_files = {
    "alexa": THESIS_ROOT / "results" / "alexa_sentiment.csv",
    "google": THESIS_ROOT / "results" / "google_sentiment.csv"
}

# Output directory for saving the visualizations, also inside the 'results' folder
output_visuals_dir = THESIS_ROOT / "results" / "sentiment_visuals"
output_visuals_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist


# --- Visualization Parameters ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a pleasant style
sns.set_palette("viridis") # Set a color palette

# --- Processing and Visualization ---

all_platform_data = []

# Correcting the print statement to reflect the actual path being looked in
# It should look in THESIS_ROOT / "results", not BASE_DIR / "results"
print(f"Looking for sentiment results in: {THESIS_ROOT / 'results'}")
print(f"Saving visualizations to: {output_visuals_dir}")


for name, path in input_files.items():
    print(f"\n📈 Processing results for {name} from {path}...")

    # Load data
    try:
        df = pd.read_csv(path)

        # --- Data Validation ---
        # Check for the required columns using 'at' for the date/timestamp
        required_cols = ['at', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns {missing} in {path}. Skipping {name}.")
            continue

        # Convert 'at' column to datetime objects
        # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
        df['at'] = pd.to_datetime(df['at'], errors='coerce')

        # Drop rows where 'at' is NaT (couldn't be parsed)
        df.dropna(subset=['at'], inplace=True)

        if df.empty:
            print(f"Warning: No valid date entries found after parsing in {path}. Skipping {name}.")
            continue

        print(f"Loaded {len(df)} reviews with valid dates.")

        # --- Quarterly Aggregation ---
        # Group by quarter using the 'at' column and sentiment, then count
        quarterly_sentiment = df.groupby([df['at'].dt.to_period('Q'), 'sentiment']).size().unstack(fill_value=0)

        # Calculate proportions
        quarterly_sentiment_prop = quarterly_sentiment.div(quarterly_sentiment.sum(axis=1), axis=0)

        # Convert PeriodIndex (derived from 'at') to Timestamp for plotting
        quarterly_sentiment_prop.index = quarterly_sentiment_prop.index.to_timestamp()

        # Add platform name
        quarterly_sentiment_prop['platform'] = name

        # Reshape for seaborn plotting (long format)
        # Use 'at' as the identifier variable
        quarterly_sentiment_long = quarterly_sentiment_prop.reset_index().melt(
            id_vars=['at', 'platform'], # Use 'at' here
            value_vars=['negative', 'neutral', 'positive'],
            var_name='sentiment_category',
            value_name='proportion'
        )
        all_platform_data.append(quarterly_sentiment_long)


        # --- Plotting for current platform ---
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=quarterly_sentiment_long, x='at', y='proportion', hue='sentiment_category') # Use 'at' here

        plt.title(f'Quarterly Sentiment Distribution for {name.capitalize()} Reviews') # Removed date range as it might vary
        plt.xlabel('Date (Quarter)') # Keep label descriptive
        plt.ylabel('Proportion of Reviews')
        plt.legend(title='Sentiment', loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Improve date formatting on x-axis (works on datetime objects regardless of column name)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator()) # Show years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3)) # Show quarters
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # Save plot
        plot_filename = output_visuals_dir / f"{name}_quarterly_sentiment_distribution.png"
        plt.savefig(plot_filename)
        print(f"Saved sentiment distribution plot to {plot_filename}")
        plt.close() # Close the plot to free memory

    except FileNotFoundError:
        print(f"Error: Input file not found at {path}. Skipping {name}.")
    except Exception as e:
        print(f"Error processing or plotting {path}: {e}. Skipping {name}.")
        import traceback
        traceback.print_exc() # Print detailed error info

print("\n--- Visualization of Sentiment Distribution Complete ---")

# --- Optional: Plotting Average Sentiment Score Comparison ---
# Combine data from both platforms if both were processed successfully
if len(all_platform_data) > 0 and len(all_platform_data) == len(input_files): # Ensure at least one file was processed successfully
    # Need to re-calculate average sentiment score from the original df
    # Rerun quarterly aggregation specifically for the score using 'at'
    all_avg_sentiment_data = []
    for name, path in input_files.items():
        try:
            df = pd.read_csv(path)
            # Ensure 'at' and 'sentiment_score' columns exist
            if 'at' not in df.columns or 'sentiment_score' not in df.columns:
                 print(f"Warning: Required columns ('at' or 'sentiment_score') not found in {path}. Cannot plot average score comparison for {name}.")
                 continue

            df['at'] = pd.to_datetime(df['at'], errors='coerce')
            df.dropna(subset=['at'], inplace=True)

            if df.empty:
                 print(f"Warning: No valid date entries found after parsing 'at' in {path}. Skipping average score plot for {name}.")
                 continue

            quarterly_avg_score = df.groupby(df['at'].dt.to_period('Q'))['sentiment_score'].mean() # Use 'at' here
            quarterly_avg_score.index = quarterly_avg_score.index.to_timestamp() # Index derived from 'at'

            avg_score_df = quarterly_avg_score.reset_index(name='average_score')
            # The index column from reset_index will be named 'at' if the original index was named 'at'
            # Ensure the date column is correctly named 'at' for plotting
            avg_score_df = avg_score_df.rename(columns={'index': 'at'}) # Rename default index col name if needed

            avg_score_df['platform'] = name
            all_avg_sentiment_data.append(avg_score_df)

        except FileNotFoundError:
            pass # Already reported above
        except Exception as e:
            print(f"Error calculating average score for {name}: {e}")
            import traceback
            traceback.print_exc()


    if len(all_avg_sentiment_data) == len(input_files): # Only plot if data for both platforms is available
        combined_avg_df = pd.concat(all_avg_sentiment_data)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=combined_avg_df, x='at', y='average_score', hue='platform') # Use 'at' here

        plt.title('Quarterly Average Sentiment Score Comparison (Google Assistant vs. Alexa)')
        plt.xlabel('Date (Quarter)') # Keep label descriptive
        plt.ylabel('Average Sentiment Score (-1 to 1)')
        plt.legend(title='Platform')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(-1, 1) # Set y-axis limits to the possible range

        # Improve date formatting on x-axis (works on datetime objects regardless of column name)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator()) # Show years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3)) # Show quarters
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save plot
        plot_filename = output_visuals_dir / "overall_average_sentiment_comparison.png"
        plt.savefig(plot_filename)
        print(f"\nSaved average sentiment comparison plot to {plot_filename}")
        plt.close()
    else:
        print("\nCould not generate average sentiment comparison plot (data for both platforms missing or errors).")

print("\n--- Visualization Script Finished ---")