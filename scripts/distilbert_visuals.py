import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import matplotlib.dates as mdates

# --- Configuration ---
# Base directory for the project (assuming this script is in the same parent dir as 'data' and 'results')
# Adjust BASE_DIR if your script is located elsewhere relative to 'results'
BASE_DIR = Path(__file__).resolve().parent

# Input directories and files (output from your sentiment analysis script)
input_files = {
    "alexa": BASE_DIR / "data" / "alexa_sentiment.csv",
    "google": BASE_DIR / "data" / "google_sentiment.csv"
}

# Output directory for saving the visualizations
output_visuals_dir = BASE_DIR / "results" / "sentiment_visuals"
output_visuals_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

# --- Visualization Parameters ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a pleasant style
sns.set_palette("viridis") # Set a color palette

# --- Processing and Visualization ---

all_platform_data = []

print(f"Looking for sentiment results in: {BASE_DIR / 'results'}")
print(f"Saving visualizations to: {output_visuals_dir}")

for name, path in input_files.items():
    print(f"\n📈 Processing results for {name} from {path}...")

    # Load data
    try:
        df = pd.read_csv(path)

        # --- Data Validation ---
        required_cols = ['date', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns {missing} in {path}. Skipping {name}.")
            continue

        # Convert 'date' column to datetime objects
        # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows where date is NaT (couldn't be parsed)
        df.dropna(subset=['date'], inplace=True)

        if df.empty:
            print(f"Warning: No valid date entries found after parsing in {path}. Skipping {name}.")
            continue

        print(f"Loaded {len(df)} reviews with valid dates.")

        # --- Quarterly Aggregation ---
        # Group by quarter and sentiment, then count
        quarterly_sentiment = df.groupby([df['date'].dt.to_period('Q'), 'sentiment']).size().unstack(fill_value=0)

        # Calculate proportions
        quarterly_sentiment_prop = quarterly_sentiment.div(quarterly_sentiment.sum(axis=1), axis=0)

        # Convert PeriodIndex to Timestamp for plotting
        quarterly_sentiment_prop.index = quarterly_sentiment_prop.index.to_timestamp()

        # Add platform name for combined plotting later if needed, or just keep separate
        quarterly_sentiment_prop['platform'] = name

        # Reshape for seaborn plotting (long format)
        quarterly_sentiment_long = quarterly_sentiment_prop.reset_index().melt(
            id_vars=['date', 'platform'],
            value_vars=['negative', 'neutral', 'positive'],
            var_name='sentiment_category',
            value_name='proportion'
        )
        all_platform_data.append(quarterly_sentiment_long)


        # --- Plotting for current platform ---
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=quarterly_sentiment_long, x='date', y='proportion', hue='sentiment_category')

        plt.title(f'Quarterly Sentiment Distribution for {name.capitalize()} Reviews (2017-2025)')
        plt.xlabel('Date (Quarter)')
        plt.ylabel('Proportion of Reviews')
        plt.legend(title='Sentiment', loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Improve date formatting on x-axis
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
if len(all_platform_data) == len(input_files):
    combined_df = pd.concat(all_platform_data)

    # Need to re-calculate average sentiment score from the original df *before* melting
    # Let's reload or re-process if needed, or modify the first loop to store aggregate score
    # Simpler approach: Rerun quarterly aggregation specifically for the score
    all_avg_sentiment_data = []
    for name, path in input_files.items():
        try:
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            if 'sentiment_score' not in df.columns:
                 print(f"Warning: 'sentiment_score' column not found in {path}. Cannot plot average score comparison.")
                 continue

            quarterly_avg_score = df.groupby(df['date'].dt.to_period('Q'))['sentiment_score'].mean()
            quarterly_avg_score.index = quarterly_avg_score.index.to_timestamp()
            avg_score_df = quarterly_avg_score.reset_index(name='average_score')
            avg_score_df['platform'] = name
            all_avg_sentiment_data.append(avg_score_df)

        except FileNotFoundError:
            pass # Already reported above
        except Exception as e:
            print(f"Error calculating average score for {name}: {e}")
            import traceback
            traceback.print_exc()


    if len(all_avg_sentiment_data) > 0:
        combined_avg_df = pd.concat(all_avg_sentiment_data)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=combined_avg_df, x='date', y='average_score', hue='platform')

        plt.title('Quarterly Average Sentiment Score Comparison (Google Assistant vs. Alexa)')
        plt.xlabel('Date (Quarter)')
        plt.ylabel('Average Sentiment Score (-1 to 1)')
        plt.legend(title='Platform')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(-1, 1) # Set y-axis limits to the possible range

        # Improve date formatting on x-axis
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
        print("\nCould not generate average sentiment comparison plot (data missing or errors).")

print("\n--- Visualization Script Finished ---")