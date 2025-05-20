import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import os
import pickle
import warnings
from pathlib import Path
import torch # Import torch for cuda check
import traceback # For detailed error printing

warnings.filterwarnings("ignore", category=FutureWarning) # Suppress specific future warnings

# --- Configuration ---
# Base directory is the folder where the script is located (e.g., thesis/scripts)
BASE_DIR = Path(__file__).resolve().parent

# Go up one directory from BASE_DIR (to thesis/)
THESIS_ROOT = BASE_DIR.parent

# Input files in the results folder
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
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"

if use_gpu:
    print(f"Using GPU: {torch.cuda.get_device_name(0)} for embeddings")
else:
    print("Using CPU for embeddings")

# Load the sentence transformer model ONCE on the correct device
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


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
print(f"Input data will be read from: {THESIS_ROOT / 'results'}")
print(f"General outputs will be saved to: {output_dir}")
print(f"Topic models and topic-specific visuals will be saved to: {topic_model_dir}")


for name, path in input_files.items():
    print(f"\n🔍 Processing {name} reviews for topic modeling...")

    try:
        df = pd.read_csv(path, parse_dates=['at'])

        required_cols = ['at', 'clean_content', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns {missing} in {path}. Skipping {name}.")
            continue

        df.dropna(subset=['at', 'clean_content'], inplace=True)
        df = df[df['clean_content'].astype(str).str.strip() != '']

        if df.empty:
            print(f"Warning: No valid reviews with content and dates found after parsing in {path}. Skipping {name}.")
            continue

        print(f"Loaded and cleaned {len(df)} valid reviews for {name}.")

        df_sorted = df.sort_values('at').reset_index(drop=True)
        docs_for_fitting = df_sorted['clean_content'].astype(str).tolist()

        print(f"Creating and fitting BERTopic model for {name} on {len(docs_for_fitting)} documents...")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(docs_for_fitting)
        print(f"BERTopic model fitting complete for {name}. Found {len(set(topics))} unique topic IDs (including -1 for outliers).")

        df_sorted['topic'] = topics
        topic_counts = topic_model.get_topic_info()
        topic_counts_path = topic_model_dir / f"{name}_topic_counts.csv"
        topic_counts.to_csv(topic_counts_path, index=False)
        print(f"Saved topic counts to {topic_counts_path}")

        print("Generating topic keywords for mapping...")
        topic_keywords_map = topic_counts.set_index('Topic')['Name'].to_dict()
        df_sorted['topic_keywords'] = df_sorted['topic'].map(lambda x: topic_keywords_map.get(x, "Outlier_Topic_-1"))

        output_path_with_topics = output_dir / f"{name}_with_topics.csv"
        df_sorted.to_csv(output_path_with_topics, index=False)
        print(f"Saved dataframe with documents, timestamps, and topics to {output_path_with_topics}")

        model_save_path = topic_model_dir / f"{name}_bertopic_model"
        topic_model.save(str(model_save_path))
        print(f"Saved BERTopic model to {model_save_path}")

        print(f"Creating standard topic visualizations (HTML files) for {name}...")
        top_n_viz = 15
        valid_model_topics = topic_counts[topic_counts['Topic'] != -1]['Topic'].tolist()
        viz_topics = []
        if valid_model_topics:
            viz_topics = topic_counts[topic_counts['Topic'].isin(valid_model_topics)].sort_values('Count', ascending=False)['Topic'].head(top_n_viz).tolist()
            if not viz_topics:
                 viz_topics = valid_model_topics[:min(top_n_viz, len(valid_model_topics))]

        if viz_topics:
            print(f"Selected top {len(viz_topics)} topics for visualization: {viz_topics}")
            try:
                fig = topic_model.visualize_barchart(top_n_topics=len(viz_topics), topics=viz_topics)
                barchart_path = topic_model_dir / f"{name}_topic_barchart.html"
                fig.write_html(str(barchart_path))
                print(f"Saved topic barchart to {barchart_path}")
            except Exception as e:
                print(f"Error visualizing barchart for {name}: {e}")

            try:
                fig = topic_model.visualize_topics(topics=viz_topics)
                map_path = topic_model_dir / f"{name}_topic_map.html"
                fig.write_html(str(map_path))
                print(f"Saved topic map to {map_path}")
            except Exception as e:
                 print(f"Error visualizing topic map for {name}: {e}")

            try:
                fig = topic_model.visualize_hierarchy(topics=viz_topics)
                hierarchy_path = topic_model_dir / f"{name}_topic_hierarchy.html"
                fig.write_html(str(hierarchy_path))
                print(f"Saved topic hierarchy to {hierarchy_path}")
            except Exception as e:
                 print(f"Error visualizing hierarchy for {name}: {e}")
        else:
            print(f"Skipping standard topic visualizations for {name} as no valid topics were available or selected for visualization.")

        # --- Temporal Topic Analysis Section REMOVED ---
        print(f"Temporal topic analysis (generation of _topics_over_time.csv and _topic_evolution.html) should be run using a separate script.")

        print(f"Preparing data for Topic-Sentiment Analysis for {name}...")
        if 'sentiment' in df_sorted.columns:
            df_topics_for_sentiment = df_sorted[df_sorted['topic'] != -1]
            if not df_topics_for_sentiment.empty:
                sentiment_by_topic = pd.crosstab(df_topics_for_sentiment['topic'], df_topics_for_sentiment['sentiment'], normalize='index') * 100
                sentiment_by_topic = sentiment_by_topic.reset_index()
                sentiment_by_topic.columns.name = None
                sentiment_by_topic['topic_keywords'] = sentiment_by_topic['topic'].astype(int).map(topic_keywords_map)
                
                topic_sentiment_path = output_dir / f"{name}_topic_sentiment.csv"
                sentiment_by_topic.to_csv(topic_sentiment_path, index=False)
                print(f"Saved topic sentiment distribution to {topic_sentiment_path}")
            else:
                print(f"No non-outlier topics found for {name}. Skipping Topic-Sentiment analysis.")
        else:
             print(f"Warning: 'sentiment' column not found in input data for {name}. Skipping Topic-Sentiment analysis.")

        print(f"Extracting topic terms for {name}...")
        topic_terms = {}
        for topic_id in topic_counts[topic_counts['Topic'] != -1]['Topic']:
            terms = topic_model.get_topic(topic_id)
            if terms is not None:
                topic_terms[topic_id] = {term: score for term, score in terms}

        if topic_terms:
            topic_terms_path = topic_model_dir / f"{name}_topic_terms.pkl"
            with open(topic_terms_path, "wb") as f:
                pickle.dump(topic_terms, f)
            print(f"Saved topic terms to {topic_terms_path}")
        else:
            print(f"No topic terms extracted for {name} (or only outlier topic found).")
            
    except FileNotFoundError:
        print(f"FATAL Error: Input file not found at {path}. Skipping {name}.")
    except Exception as e:
        print(f"An unexpected FATAL error occurred while processing {name}: {e}")
        traceback.print_exc()

print("\n🎉 BERTopic modeling (core outputs) complete for all datasets!")