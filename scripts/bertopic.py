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
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# File paths 
input_files = {
    "alexa": "/Users/viltetverijonaite/Desktop/MSC/THESIS/alexa_sentiment.csv",
    "google": "/Users/viltetverijonaite/Desktop/MSC/THESIS/google_sentiment.csv"
}
output_dir = "/Users/viltetverijonaite/Desktop/MSC/THESIS/"

# Create topic modeling directory if it doesn't exist
topic_model_dir = os.path.join(output_dir, "topic_models")
os.makedirs(topic_model_dir, exist_ok=True)

# Use GPU if available
use_gpu = True
if use_gpu:
    print("Using GPU for embeddings")
else:
    print("Using CPU for embeddings")

# Function to convert string dates to datetime
def parse_dates(df):
    df['at'] = pd.to_datetime(df['at'])
    return df

# Function to extract year-quarter from date
def extract_year_quarter(date):
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}-Q{quarter}"

# Function to preprocess data for temporal topic analysis
def preprocess_for_temporal(df):
    # Extract year-quarter
    df['period'] = df['at'].apply(extract_year_quarter)
    # Sort by date
    df = df.sort_values('at')
    return df

# Load the sentence transformer model
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Process each dataset
for name, path in input_files.items():
    print(f"\n🔍 Processing {name} reviews for topic modeling...")
    
    # Load data
    df = pd.read_csv(path)
    df = parse_dates(df)
    df_temporal = preprocess_for_temporal(df)
    
    # Only use reviews with valid content
    docs = df['clean_content'].astype(str).tolist()
    timestamps = df['at'].tolist()
    
    print(f"Creating BERTopic model for {name}...")
    
    # Create and fit BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    # Fit the model and get topics
    topics, probs = topic_model.fit_transform(docs)
    
    # Save the number of documents per topic
    topic_counts = topic_model.get_topic_info()
    topic_counts.to_csv(os.path.join(topic_model_dir, f"{name}_topic_counts.csv"), index=False)
    
    # Get the top topics
    top_topics = topic_counts.loc[topic_counts['Topic'] != -1].sort_values('Count', ascending=False)['Topic'].head(20).tolist()
    
    # Store topics and probabilities in the original dataframe
    df['topic'] = topics
    
    # Generate topic representations and save
    topic_keywords = {}
    print("Generating topic keywords and descriptions...")
    for topic_id in tqdm(set(topics)):
        if topic_id != -1:  # Skip outlier topic
            words = [word for word, _ in topic_model.get_topic(topic_id)]
            topic_keywords[topic_id] = ", ".join(words[:10])
    
    # Map topic IDs to keywords
    df['topic_keywords'] = df['topic'].map(lambda x: topic_keywords.get(x, "Outlier"))
    
    # Save results
    output_path = os.path.join(output_dir, f"{name}_with_topics.csv")
    df.to_csv(output_path, index=False)
    
    # Save the model
    topic_model.save(os.path.join(topic_model_dir, f"{name}_bertopic_model"))
    
    # Create visualizations
    print("Creating topic visualizations...")
    
    # Topic word scores
    fig = topic_model.visualize_barchart(top_n_topics=15)
    fig.write_html(os.path.join(topic_model_dir, f"{name}_topic_barchart.html"))
    
    # Topic similarity map
    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(topic_model_dir, f"{name}_topic_map.html"))
    
    # Hierarchical clustering of topics
    fig = topic_model.visualize_hierarchy()
    fig.write_html(os.path.join(topic_model_dir, f"{name}_topic_hierarchy.html"))
    
    # Temporal topic analysis
    print("Performing temporal topic analysis...")
    
    # Group by period for temporal analysis
    periods = sorted(df_temporal['period'].unique())
    
    # Create timestamps for temporal analysis
    df_temporal['timestamp'] = pd.to_datetime(df_temporal['period'].apply(lambda x: f"{x.split('-')[0]}-{int(x.split('Q')[1])*3-2}-01"))
    
    # Prepare data for temporal topic analysis
    timestamps = df_temporal['timestamp'].tolist()
    documents = df_temporal['clean_content'].astype(str).tolist()
    
    # Perform temporal topic modeling
    try:
        topics_over_time = topic_model.topics_over_time(documents, timestamps, nr_bins=len(periods))
        topics_over_time.to_csv(os.path.join(topic_model_dir, f"{name}_topics_over_time.csv"), index=False)
        
        # Visualize evolution of topics over time
        fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
        fig.write_html(os.path.join(topic_model_dir, f"{name}_topic_evolution.html"))
    except Exception as e:
        print(f"Error in temporal analysis: {e}")
    
    # Create aspect-sentiment dataframe for ABSA
    print("Preparing data for Aspect-Based Sentiment Analysis...")
    
    # Create a pivot table of sentiment by topic
    sentiment_by_topic = pd.crosstab(df['topic'], df['sentiment'], normalize='index') * 100
    sentiment_by_topic.reset_index(inplace=True)
    sentiment_by_topic.columns.name = None
    
    # Map topic IDs to keywords for better readability
    sentiment_by_topic['topic_keywords'] = sentiment_by_topic['topic'].map(lambda x: topic_keywords.get(x, "Outlier"))
    
    # Save for ABSA
    sentiment_by_topic.to_csv(os.path.join(output_dir, f"{name}_topic_sentiment.csv"), index=False)
    
    # Extract topic-term-relevance for aspect extraction
    topic_terms = {}
    for topic_id in set(topics):
        if topic_id != -1:
            terms = topic_model.get_topic(topic_id)
            topic_terms[topic_id] = {term: score for term, score in terms}
    
    # Save topic terms for ABSA
    with open(os.path.join(topic_model_dir, f"{name}_topic_terms.pkl"), "wb") as f:
        pickle.dump(topic_terms, f)
    
    # Create dataset with top topics for ABSA
    if -1 in topics:  # Remove outlier topic if present
        top_topics = [t for t in top_topics if t != -1]
    
    # Filter data for top topics for ABSA
    absa_df = df[df['topic'].isin(top_topics)].copy()
    absa_df.to_csv(os.path.join(output_dir, f"{name}_for_absa.csv"), index=False)
    
    print(f"✅ Topic modeling for {name} completed. Results saved to {output_dir}")

print("\n🎉 BERTopic modeling complete for all datasets! ")