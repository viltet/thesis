import pandas as pd
import os

# === Configuration ===
INPUT_FILES = {
    "alexa": "/Users/viltetverijonaite/Desktop/MSC/THESIS/alexa_processed.csv",
    "google": "/Users/viltetverijonaite/Desktop/MSC/THESIS/google_processed.csv"
}
OUTPUT_DIR = "/Users/viltetverijonaite/Desktop/MSC/THESIS/model_comparison"
PILOT_SAMPLE_FILENAME = "pilot_subset.csv"
TOTAL_SAMPLE_SIZE = 200  # total reviews to sample across all years and sources

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_combine(input_files):
    """
    Load each CSV, ensure 'at' timestamp parsed, and combine into one DataFrame.
    """
    frames = []
    for source, path in input_files.items():
        print(f"Loading {source} from {path}")
        df = pd.read_csv(path)
        # Parse 'at' timestamp
        if 'at' in df.columns:
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
        else:
            raise KeyError(f"Expected 'at' column in {path}")
        df['source'] = source
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def create_pilot_subset(df, total_size):
    """
    Create a stratified sample by year across the combined DataFrame.
    Returns a DataFrame of sampled rows.
    """
    # Extract year from 'at'
    df = df.copy()
    df['year'] = df['at'].dt.year
    years = sorted(df['year'].dropna().unique())
    per_year = max(1, total_size // len(years))

    samples = []
    for year in years:
        year_df = df[df['year'] == year]
        if year_df.empty:
            continue
        n = min(per_year, len(year_df))
        samples.append(year_df.sample(n, random_state=100))

    pilot_df = pd.concat(samples).reset_index(drop=True)

    # Convert score column to numeric if needed
    pilot_df['score'] = pd.to_numeric(pilot_df['score'], errors='coerce')

    # Map 1-5 star scores to sentiment classes (optional baseline)
    pilot_df['sentiment_class'] = pilot_df['score'].apply(
        lambda x: 0 if x <= 2 else (1 if x == 3 else 2)
    )

    return pilot_df


def main():
    # Load & combine data
    combined_df = load_and_combine(INPUT_FILES)
    print(f"Total reviews loaded: {len(combined_df)}")

    # Create pilot subset
    pilot_df = create_pilot_subset(combined_df, TOTAL_SAMPLE_SIZE)
    print(f"Pilot subset size: {len(pilot_df)}")

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, PILOT_SAMPLE_FILENAME)
    pilot_df.to_csv(output_path, index=False)
    print(f"Pilot subset saved to: {output_path}")


if __name__ == "__main__":
    main()

