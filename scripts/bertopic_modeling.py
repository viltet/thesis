import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import warnings
from pathlib import Path
import torch # Import torch for cuda check

warnings.filterwarnings("ignore", category=FutureWarning) # Suppress specific future warnings

# --- Configuration ---
# Base directory is the folder where the script is located (e.g., thesis/scripts)
BASE_DIR = Path(__file__).resolve().parent

# Go up one directory from BASE_DIR (to thesis/)
THESIS_ROOT = BASE_DIR.parent

# Input files in the results folder 
# These files should contain 'at', 'clean_content', 'sentiment', 'sentiment_score'
input_files = {
    "alexa": THESIS_ROOT / "results" / "alexa_sentiment.csv", 
    "google": THESIS_ROOT / "results" / "google_sentiment.csv" 
}

# Output directory (results folder inside thesis)
output_dir = THESIS_ROOT / "results" 
output_dir.mkdir(parents=True, exist_ok=True)

# Topic model directory inside results
topic_model_dir = output_dir / "topic_models"
topic_model_dir.mkdir(parents=True, exist_ok=True)

# Dynamic GPU check
use_gpu = torch.cuda.is_available() # Dynamic check
device = "cuda" if use_gpu else "cpu" # Define device

if use_gpu:
    print(f"Using GPU: {torch.cuda.get_device_name(0)} for embeddings") # Print device name
else:
    print("Using CPU for embeddings")

# Load the sentence transformer model ONCE on the correct device
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device) # Pass device


# Configure BERTopic parameters
vectorizer_model = CountVectorizer(stop_words="english",
                                  min_df=10,
                                  ngram_range=(1, 2))

umap_model = UMAP(n_neighbors=15,
                 n_components=5,
                 min_dist=0.0,
                 metric='cosine',
                 random_state=42)

hdbscan_model = HDBSCAN(min_cluster_size=30,
                       min_samples=10,
                       metric='euclidean',
                       gen_min_span_tree=True,
                       prediction_data=True)


# --- Processing each dataset ---
print(f"Looking for sentiment results in: {THESIS_ROOT / 'results'}") # Corrected print
print(f"Saving outputs to: {output_dir}") # Corrected print
print(f"Saving topic models and visuals to: {topic_model_dir}") # Added print


for name, path in input_files.items():
    print(f"\n🔍 Processing {name} reviews for topic modeling...")

    # Load data
    try:
        # Specify 'at' as parse_dates column directly in read_csv
        df = pd.read_csv(path, parse_dates=['at'])

        # --- Data Validation and Cleaning ---
        # Check for the required columns ('at' for date, 'clean_content' for text, sentiment for Topic-Sentiment)
        required_cols = ['at', 'clean_content', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns {missing} in {path}. Skipping {name}.")
            continue

        # Drop rows where 'at' is NaT (couldn't be parsed) or clean_content is missing/empty
        df.dropna(subset=['at', 'clean_content'], inplace=True)
        df = df[df['clean_content'].astype(str).str.strip() != ''] # Remove empty strings after conversion to str

        if df.empty:
            print(f"Warning: No valid reviews with content and dates found after parsing in {path}. Skipping {name}.")
            continue

        print(f"Loaded and cleaned {len(df)} valid reviews.")

        # Sort by date - CRITICAL for temporal analysis later
        df_sorted = df.sort_values('at').reset_index(drop=True) # Use a new name for clarity

        # Get documents and timestamps from the *sorted* dataframe for temporal analysis
        docs_sorted = df_sorted['clean_content'].astype(str).tolist()
        timestamps_sorted = df_sorted['at'].tolist() # Use timestamps from the sorted df

        # Get documents from the *original* dataframe for fitting (sorting isn't strictly required by fit_transform itself)
        # However, using the sorted docs is fine and often simpler to keep consistent
        docs_for_fitting = docs_sorted # Use the sorted list for fitting


        print(f"Creating and fitting BERTopic model for {name} on {len(docs_for_fitting)} documents...")

        # Create and fit BERTopic model (embedding_model device already set)
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True, # Keep probabilities if needed later
            verbose=True
        )

        # Fit the model - DO NOT pass timestamps here to avoid the TypeError
        topics, probs = topic_model.fit_transform(docs_for_fitting)


        # Add topics back to the *sorted* dataframe
        df_sorted['topic'] = topics
        # Optional: store topic probabilities if needed
        # df_sorted['topic_probability'] = [p[t] if t != -1 else 0 for t, p in zip(topics, probs)] # Example for assigned topic probability


        # Save the number of documents per topic
        topic_counts = topic_model.get_topic_info()
        topic_counts.to_csv(os.path.join(topic_model_dir, f"{name}_topic_counts.csv"), index=False)


        # Generate topic representations (keywords) and save
        print("Generating topic keywords...")
        # get_topic_info() already contains keywords, map from there
        topic_keywords_map = topic_counts.set_index('Topic')['Name'].to_dict()
        df_sorted['topic_keywords'] = df_sorted['topic'].map(lambda x: topic_keywords_map.get(x, "Outlier"))

        # Save the sorted dataframe with topic assignments
        output_path = os.path.join(output_dir, f"{name}_with_topics.csv")
        df_sorted.to_csv(output_path, index=False) # Save the sorted df
        print(f"Saved dataframe with topics to {output_path}")

        # Save the BERTopic model
        model_save_path = os.path.join(topic_model_dir, f"{name}_bertopic_model")
        topic_model.save(model_save_path)
        print(f"Saved BERTopic model to {model_save_path}")


        # --- Create Visualizations ---
        print("Creating topic visualizations...")

        # Get the top topics for visualization (excluding outlier topic -1)
        top_n_viz = 15 # Number of top topics to visualize
        # Filter topic_counts for non-outlier topics before getting head
        viz_topics = topic_counts[topic_counts['Topic'] != -1].sort_values('Count', ascending=False)['Topic'].head(top_n_viz).tolist()


        # Topic word scores (shows top words per topic)
        try:
            fig = topic_model.visualize_barchart(top_n_topics=top_n_viz)
            barchart_path = os.path.join(topic_model_dir, f"{name}_topic_barchart.html")
            fig.write_html(barchart_path)
            print(f"Saved topic barchart to {barchart_path}")
        except Exception as e:
            print(f"Error visualizing barchart: {e}")

        # Topic similarity map (UMAP/t-SNE projection)
        try:
            fig = topic_model.visualize_topics(topics=viz_topics) # Visualize only top topics
            map_path = os.path.join(topic_model_dir, f"{name}_topic_map.html")
            fig.write_html(map_path)
            print(f"Saved topic map to {map_path}")
        except Exception as e:
             print(f"Error visualizing topic map: {e}")


        # Hierarchical clustering of topics
        try:
            fig = topic_model.visualize_hierarchy(topics=viz_topics) # Visualize only top topics
            hierarchy_path = os.path.join(topic_model_dir, f"{name}_topic_hierarchy.html")
            fig.write_html(hierarchy_path)
            print(f"Saved topic hierarchy to {hierarchy_path}")
        except Exception as e:
             print(f"Error visualizing hierarchy: {e}")


        # --- Temporal topic analysis ---
        # Perform temporal topic modeling as a separate step
        print("Performing temporal topic analysis...")
        try:
            # IMPORTANT: Pass the *sorted* documents and timestamps
            topics_over_time_df = topic_model.topics_over_time(docs_sorted, timestamps_sorted, freq='Q') # Call topics_over_time separately with sorted data

            oot_csv_path = os.path.join(topic_model_dir, f"{name}_topics_over_time.csv")
            topics_over_time_df.to_csv(oot_csv_path, index=False)
            print(f"Saved topics over time data to {oot_csv_path}")

            # Visualize evolution of topics over time
            # Filter for topics that exist in the topics_over_time output and are not outliers
            topics_for_oot_viz = topics_over_time_df['Topic'].unique().tolist()
            if -1 in topics_for_oot_viz:
                topics_for_oot_viz.remove(-1)
            # Select top N topics based on overall count (ensuring they appeared over time)
            available_oot_topics = topic_counts.loc[topic_counts['Topic'].isin(topics_for_oot_viz)]
            topics_for_oot_viz = available_oot_topics.sort_values('Count', ascending=False)['Topic'].head(10).tolist()

            # Pass the dataframe returned by topics_over_time and the selected topics
            fig = topic_model.visualize_topics_over_time(topics_over_time_df, topics=topics_for_oot_viz)
            oot_html_path = os.path.join(topic_model_dir, f"{name}_topic_evolution.html")
            fig.write_html(oot_html_path)
            print(f"Saved topic evolution plot to {oot_html_path}")
        except Exception as e:
            print(f"Error in temporal analysis: {e}")
            import traceback
            traceback.print_exc()


        # --- Preparing Data for Topic-Sentiment Analysis & ABSA Input ---
        print("Preparing data for Topic-Sentiment Analysis and potential ABSA input...")

        # Calculate sentiment distribution per topic (Topic-Sentiment Analysis)
        # Use the sorted dataframe which contains topics and original sentiment
        if 'sentiment' in df_sorted.columns: # Ensure sentiment column exists
            sentiment_by_topic = pd.crosstab(df_sorted['topic'], df_sorted['sentiment'], normalize='index') * 100
            sentiment_by_topic = sentiment_by_topic.reset_index()
            sentiment_by_topic.columns.name = None # Remove index name
            sentiment_by_topic['topic_keywords'] = sentiment_by_topic['topic'].map(lambda x: topic_keywords_map.get(x, "Outlier"))

            topic_sentiment_path = os.path.join(output_dir, f"{name}_topic_sentiment.csv")
            sentiment_by_topic.to_csv(topic_sentiment_path, index=False)
            print(f"Saved topic sentiment distribution to {topic_sentiment_path}")
        else:
             print("Warning: 'sentiment' column not found in input data. Skipping Topic-Sentiment analysis.")


        # Extract topic-term-relevance for potential aspect extraction later
        # These are the terms that define each topic
        topic_terms = {}
        for topic_id in set(topics):
            if topic_id != -1:
                terms = topic_model.get_topic(topic_id)
                topic_terms[topic_id] = {term: score for term, score in terms}

        topic_terms_path = os.path.join(topic_model_dir, f"{name}_topic_terms.pkl")
        with open(topic_terms_path, "wb") as f:
            pickle.dump(topic_terms, f)
        print(f"Saved topic terms to {topic_terms_path}")

        # Create a subset of the original dataframe for reviews assigned to the *most frequent* topics
        # This subset can be used as input for a dedicated Aspect-Based Sentiment Analysis step
        # Note: This dataframe includes the overall review sentiment, not aspect-specific sentiment
        # Use topic_counts from the fitted model
        top_topics_ids = topic_counts.loc[topic_counts['Topic'] != -1].sort_values('Count', ascending=False)['Topic'].head(50).tolist() # Get top 50 topics

        # Filter the sorted dataframe (df_sorted) for these top topics
        absa_df = df_sorted[df_sorted['topic'].isin(top_topics_ids)].copy() # Filter sorted df
        absa_output_path = os.path.join(output_dir, f"{name}_reviews_top_topics_for_absa.csv")
        absa_df.to_csv(absa_output_path, index=False)
        print(f"Saved reviews from top topics for potential ABSA input to {absa_output_path}")


    except FileNotFoundError:
        print(f"Error: Input file not found at {path}. Skipping {name}.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {name}: {e}")
        import traceback
        traceback.print_exc() # Print detailed error info

print("\n🎉 BERTopic modeling complete for all datasets! ")