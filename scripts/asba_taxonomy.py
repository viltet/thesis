import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from collections import defaultdict
import spacy
from itertools import combinations
from pathlib import Path

# Load spaCy for sentence segmentation and dependency parsing
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Base directory is the folder where the script is located (e.g., thesis/scripts)
BASE_DIR = Path(__file__).resolve().parent

# Go up one directory from BASE_DIR (to thesis/)
THESIS_ROOT = BASE_DIR.parent # This gets you to /Users/viltetverijonaite/Desktop/MSC/THESIS/thesis/

# Define output directory within results
output_dir = THESIS_ROOT / "results" / "absa_results" # Relative path
output_dir.mkdir(parents=True, exist_ok=True)

# Input files relative to THESIS_ROOT
input_files = {
    "alexa": {
        "with_topics": THESIS_ROOT / "results" / "alexa_with_topics.csv", # Relative path
        "topic_terms": THESIS_ROOT / "results" / "topic_models" / "alexa_topic_terms.pkl", # Relative path
        "topics_over_time": THESIS_ROOT / "results" / "topic_models" / "alexa_topics_over_time.csv" # Relative path
    },
    "google": {
        "with_topics": THESIS_ROOT / "results" / "google_with_topics.csv", # Relative path
        "topic_terms": THESIS_ROOT / "results" / "topic_models" / "google_topic_terms.pkl", # Relative path
        "topics_over_time": THESIS_ROOT / "results" / "topic_models" / "google_topics_over_time.csv" # Relative path
    }
}
# Define theory-aligned taxonomy with extended keywords for more precise matching
taxonomy = {
    "Functionality & Performance": [
        "command", "task", "function", "request", "execute", "perform", "play", "control",
        "music", "timer", "alarm", "respond", "slow", "fast", "quick", "accurate", "ability",
        "capability", "feature", "work", "operation", "answer", "weather", "news", "skill",
        "search", "query", "song", "playlist", "speed", "performance", "reliable", "inconsistent",
        "consistent", "accomplish", "smart", "intelligence", "stupid", "dumb", "basic"
    ],
    
    "Voice Recognition": [
        "hear", "listen", "recognize", "understanding", "mic", "voice", "accent", "speech",
        "microphone", "wake", "alexa", "hey google", "ok google", "command", "activation",
        "trigger", "phrase", "call", "name", "hear me", "misheard", "mishear", "understand",
        "detection", "sensitivity", "accent", "pronunciation", "dialect", "language", "recognition"
    ],
    
    "Knowledge Base": [
        "answer", "knowledge", "info", "response", "fact", "question", "data", "correct",
        "wrong", "information", "knowing", "research", "source", "accurate", "inaccurate",
        "encyclopedia", "intelligence", "smart", "learn", "education", "informed", "wisdom",
        "trivia", "facts", "content", "query", "request", "answer", "respond"
    ],
    
    "Integration & Ecosystem": [
        "integrate", "connect", "compatible", "device", "home", "nest", "smart home", "ecosystem",
        "philips", "hue", "lights", "thermostat", "tv", "television", "speaker", "app", "phone",
        "smartphone", "skill", "third-party", "partner", "service", "platform", "sync",
        "connection", "pair", "bluetooth", "wifi", "wireless", "smart", "bulb", "plug", "switch",
        "camera", "doorbell", "lock", "appliance", "interoperability", "echo", "home mini"
    ],
    
    "Usability & Interface": [
        "setup", "interface", "easy", "use", "design", "confusing", "intuitive", "simple",
        "complicated", "difficult", "user-friendly", "accessibility", "accessible", "learn",
        "instructions", "guide", "tutorial", "help", "clear", "straightforward", "configuration",
        "settings", "customize", "personalize", "navigate", "interaction", "command structure"
    ],
    
    "Privacy & Security": [
        "privacy", "data", "listening", "security", "surveillance", "record", "spy", "collect",
        "tracking", "concern", "worry", "safe", "unsafe", "breach", "leak", "consent", "permission",
        "trust", "trustworthy", "creepy", "scary", "suspicious", "watching", "monitoring", "gdpr",
        "policy", "terms", "agreement", "encryption", "protected", "vulnerable", "hack", "risk",
        "danger", "paranoid", "microphone", "camera", "recording", "personal", "information", "location"
    ],
    
    "Updates & Evolution": [
        "update", "version", "bug", "feature", "release", "patch", "upgrade", "improve",
        "improvement", "fix", "issue", "problem", "solved", "downgrade", "regression", "change",
        "changed", "new", "added", "removed", "missing", "development", "roadmap", "progress",
        "evolve", "evolution", "grow", "maturity", "mature", "immature", "beta", "alpha", "stable"
    ],
    
    "Support & Service": [
        "support", "help", "service", "issue", "resolution", "customer", "contact", "call",
        "phone", "email", "chat", "representative", "agent", "ticket", "case", "response",
        "warranty", "replacement", "refund", "return", "satisfaction", "dissatisfaction",
        "frustrated", "complaint", "feedback", "solve", "solution", "troubleshoot", "repair"
    ],
    
    "Social & Emotional Aspects": [
        "personality", "character", "funny", "humor", "joke", "laugh", "fun", "entertaining",
        "companion", "friend", "relationship", "emotion", "emotional", "human-like", "humanlike",
        "personal", "personable", "warm", "cold", "robotic", "mechanical", "natural", "unnatural",
        "conversation", "conversational", "chat", "talk", "dialogue", "interaction", "interactive",
        "respond", "response", "reply", "engaging", "engage", "connection", "connect", "relate"
    ],
    
    "Personalization & Intelligence": [
        "personalize", "customize", "preference", "learn", "adapt", "suggest", "recommendation",
        "profile", "account", "user", "individual", "specific", "tailored", "custom", "habit",
        "routine", "pattern", "predict", "predictive", "anticipate", "remember", "memory",
        "context", "contextual", "awareness", "recognize", "familiar", "personal", "special",
        "unique", "adjust", "adaptation", "history", "previous", "past", "experience"
    ]
}

# Load Sentence Transformer model for semantic matching
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create embeddings for taxonomy categories (same as before)
def create_taxonomy_embeddings(taxonomy):
    category_descriptions = {}
    category_embeddings = {}
    
    for category, keywords in taxonomy.items():
        # Create a descriptive text for each category
        description = f"{category}: " + ", ".join(keywords)
        category_descriptions[category] = description
        
        # Generate embeddings for the description
        category_embeddings[category] = embedding_model.encode(description)
    
    return category_descriptions, category_embeddings

# Enhanced function to map topics to multiple taxonomy categories
def map_topics_to_taxonomy_multi(topic_terms, taxonomy_embeddings, threshold=0.35):
    topic_to_categories = {}
    topic_to_category_scores = {}
    
    print("Mapping topics to taxonomy categories...")
    for topic_id, terms_dict in tqdm(topic_terms.items()):
        # Create a descriptive text for the topic using its top terms
        top_terms = sorted(terms_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        topic_description = ", ".join([term for term, _ in top_terms])
        
        # Generate embedding for the topic description
        topic_embedding = embedding_model.encode(topic_description)
        
        # Calculate similarity with each category
        similarities = {}
        for category, category_embedding in taxonomy_embeddings.items():
            similarity = cosine_similarity([topic_embedding], [category_embedding])[0][0]
            similarities[category] = similarity
        
        # Get categories that exceed the similarity threshold
        selected_categories = {category: score for category, score in similarities.items() 
                              if score >= threshold}
        
        # If no category exceeds threshold, use the best one
        if not selected_categories:
            best_category = max(similarities, key=similarities.get)
            selected_categories = {best_category: similarities[best_category]}
        
        # Store the mappings and scores
        topic_to_categories[topic_id] = selected_categories
        topic_to_category_scores[topic_id] = similarities
    
    return topic_to_categories, topic_to_category_scores
# Load the spaCy language model
print("Loading spaCy language model...")
nlp = spacy.load("en_core_web_sm")

# Function to split reviews into sentences and identify aspect-sentiment pairs
def extract_aspect_sentiment_pairs(df, topic_to_categories, taxonomy):
    print("Extracting aspect-sentiment pairs from reviews...")
    aspect_sentiment_pairs = []
    
    # Process a sample of reviews for efficiency (adjust as needed)
    sample_size = min(50000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Flatten the taxonomy keywords for easier matching
    aspect_keywords = {}
    for aspect, keywords in taxonomy.items():
        for keyword in keywords:
            aspect_keywords[keyword.lower()] = aspect
    
    # Process each review
    for idx, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0]):
        review_text = row['clean_content']
        review_sentiment = row['sentiment']
        review_sentiment_score = row['sentiment_score']
        
        # Get topic-based aspect categories
        topic_id = row['topic']
        if topic_id in topic_to_categories:
            topic_aspects = topic_to_categories[topic_id]
        else:
            continue
        
        # Process review to identify sentences and potential aspect terms
        doc = nlp(review_text)
        
        # For each sentence, try to identify aspects and associate with sentiment
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) < 5:  # Skip very short sentences
                continue
                
            # Calculate sentence sentiment - simple approach: use review sentiment
            # For more accuracy, you could use a sentence-level sentiment model
            sentence_sentiment = review_sentiment
            sentence_score = review_sentiment_score
            
            # Method 1: Use topic-assigned aspects for the entire review
            for aspect, similarity in topic_aspects.items():
                aspect_sentiment_pairs.append({
                    'review_id': idx,
                    'sentence': sentence_text,
                    'aspect': aspect,
                    'detection_method': 'topic',
                    'confidence': similarity,
                    'sentiment': sentence_sentiment,
                    'sentiment_score': sentence_score
                })
            
            # Method 2: Look for explicit aspect keywords in the sentence
            found_aspects = set()
            sentence_lower = sentence_text.lower()
            for keyword, aspect in aspect_keywords.items():
                if keyword in sentence_lower:
                    found_aspects.add(aspect)
            
            for aspect in found_aspects:
                aspect_sentiment_pairs.append({
                    'review_id': idx,
                    'sentence': sentence_text,
                    'aspect': aspect,
                    'detection_method': 'keyword',
                    'confidence': 1.0,  # Direct keyword match
                    'sentiment': sentence_sentiment,
                    'sentiment_score': sentence_score
                })
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(aspect_sentiment_pairs)
    return pairs_df

# Function to analyze multi-aspect sentiment over time
def analyze_multi_aspect_sentiment_over_time(reviews_df, pairs_df):
    # Convert date to period (quarter)
    reviews_df['at'] = pd.to_datetime(reviews_df['at'])
    reviews_df['period'] = reviews_df['at'].dt.to_period('Q')
    
    # Create a mapping from review_id to period
    review_to_period = pd.Series(reviews_df['period'].values, index=reviews_df.index)
    
    # Add period information to the pairs dataframe
    pairs_df['period'] = pairs_df['review_id'].map(review_to_period)
    
    # Convert sentiment to numerical (-1 for negative, 0 for neutral, 1 for positive)
    sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    pairs_df['sentiment_value'] = pairs_df['sentiment'].map(sentiment_map)
    
    # Group by period and aspect to calculate average sentiment
    aspect_sentiment = pairs_df.groupby(['period', 'aspect'])['sentiment_value'].mean().reset_index()
    
    # Count mentions of each aspect per period
    aspect_counts = pairs_df.groupby(['period', 'aspect']).size().reset_index(name='mentions')
    
    # Merge sentiment and counts
    aspect_analysis = aspect_sentiment.merge(aspect_counts, on=['period', 'aspect'])
    
    # Pivot for easier analysis
    sentiment_pivot = aspect_analysis.pivot(index='period', columns='aspect', values='sentiment_value')
    mentions_pivot = aspect_analysis.pivot(index='period', columns='aspect', values='mentions')
    
    return aspect_analysis, sentiment_pivot, mentions_pivot

# Function to analyze aspect co-occurrence
def analyze_aspect_co_occurrence(pairs_df):
    # Group by review_id to get all aspects mentioned in each review
    review_aspects = pairs_df.groupby('review_id')['aspect'].unique()
    
    # Count co-occurrences
    co_occurrence = defaultdict(int)
    
    for aspects in review_aspects:
        if len(aspects) > 1:
            for aspect1, aspect2 in combinations(sorted(aspects), 2):
                co_occurrence[(aspect1, aspect2)] += 1
    
    # Convert to DataFrame
    co_occur_rows = []
    for (aspect1, aspect2), count in co_occurrence.items():
        co_occur_rows.append({
            'aspect1': aspect1,
            'aspect2': aspect2,
            'co_occurrences': count
        })
    
    co_occur_df = pd.DataFrame(co_occur_rows)
    return co_occur_df

# Function to plot co-occurrence network
def plot_aspect_co_occurrence(co_occur_df, min_occurrences=50, assistant_name=""):
    import networkx as nx
    
    # Filter by minimum occurrences
    filtered_df = co_occur_df[co_occur_df['co_occurrences'] >= min_occurrences]
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes for all unique aspects
    aspects = set(filtered_df['aspect1'].unique()) | set(filtered_df['aspect2'].unique())
    for aspect in aspects:
        G.add_node(aspect)
    
    # Add weighted edges based on co-occurrences
    for _, row in filtered_df.iterrows():
        G.add_edge(row['aspect1'], row['aspect2'], weight=row['co_occurrences'])
    
    # Calculate node size based on degree
    degrees = dict(nx.degree(G))
    node_size = [degrees[node] * 100 for node in G.nodes()]
    
    # Calculate edge width based on weight
    edge_width = [G[u][v]['weight'] / 10 for u, v in G.edges()]
    
    # Create plot
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using force-directed layout
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title(f'Aspect Co-occurrence Network for {assistant_name.capitalize()}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{assistant_name}_aspect_cooccurrence_network.png"), dpi=300)
    plt.close()

# Main processing loop
for assistant_name, files in input_files.items():
    print(f"\n🔍 Processing {assistant_name} for Enhanced Aspect-Based Sentiment Analysis...")
    
    # Load data
    df = pd.read_csv(files["with_topics"])
    
    # Load topic terms
    with open(files["topic_terms"], "rb") as f:
        topic_terms = pickle.load(f)
    
    # Create taxonomy embeddings
    print("Creating taxonomy embeddings...")
    category_descriptions, category_embeddings = create_taxonomy_embeddings(taxonomy)
    
    # Map topics to multiple taxonomy categories
    topic_to_categories, topic_to_category_scores = map_topics_to_taxonomy_multi(topic_terms, category_embeddings)
    
    # Extract aspect-sentiment pairs
    pairs_df = extract_aspect_sentiment_pairs(df, topic_to_categories, taxonomy)
    pairs_df.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment_pairs.csv"), index=False)
    
    # Analyze multi-aspect sentiment over time
    aspect_analysis, sentiment_pivot, mentions_pivot = analyze_multi_aspect_sentiment_over_time(df, pairs_df)
    
    # Save aspect analysis data
    aspect_analysis.to_csv(os.path.join(output_dir, f"{assistant_name}_multi_aspect_analysis.csv"), index=False)
    sentiment_pivot.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment_pivot_enhanced.csv"))
    mentions_pivot.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_mentions_pivot.csv"))
    
    # Plot enhanced aspect sentiment trends
    # Function to plot aspect sentiment trends
    def plot_aspect_sentiment_trends(aspect_sentiment_pivot, assistant_name):
        plt.figure(figsize=(14, 8))
        
        # Plot each aspect as a line
        for aspect in aspect_sentiment_pivot.columns:
            plt.plot(aspect_sentiment_pivot.index.astype(str), aspect_sentiment_pivot[aspect], 
                     marker='o', linewidth=2, label=aspect)
        
        plt.title(f'Aspect-Level Sentiment Trends for {assistant_name.capitalize()}', fontsize=16)
        plt.xlabel('Time Period (Quarterly)', fontsize=14)
        plt.ylabel('Average Sentiment (-1 to 1)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment_trends.png"), dpi=300)
        plt.close()
    
        plot_aspect_sentiment_trends(sentiment_pivot, assistant_name + "_enhanced")
    
    # Analyze and plot aspect co-occurrence
    co_occur_df = analyze_aspect_co_occurrence(pairs_df)
    co_occur_df.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_cooccurrence.csv"), index=False)
    plot_aspect_co_occurrence(co_occur_df, assistant_name=assistant_name)
    
# Function to plot aspect sentiment trends
def plot_aspect_sentiment_trends(aspect_sentiment_pivot, assistant_name):
    plt.figure(figsize=(14, 8))
    
    # Plot each aspect as a line
    for aspect in aspect_sentiment_pivot.columns:
        plt.plot(aspect_sentiment_pivot.index.astype(str), aspect_sentiment_pivot[aspect], 
                 marker='o', linewidth=2, label=aspect)
    
    plt.title(f'Aspect-Level Sentiment Trends for {assistant_name.capitalize()}', fontsize=16)
    plt.xlabel('Time Period (Quarterly)', fontsize=14)
    plt.ylabel('Average Sentiment (-1 to 1)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment_trends.png"), dpi=300)
    plt.close()

# Function to analyze aspect importance over time
def analyze_aspect_importance(df, topic_to_category):
    # Map topics to aspects
    df['aspect'] = df['topic'].map(topic_to_category)
    
    # Convert date to period (quarter)
    df['at'] = pd.to_datetime(df['at'])
    df['period'] = df['at'].dt.to_period('Q')
    
    # Count reviews per aspect per period
    aspect_counts = df.groupby(['period', 'aspect']).size().reset_index(name='count')
    
    # Calculate total reviews per period
    period_totals = aspect_counts.groupby('period')['count'].sum().reset_index(name='total')
    
    # Merge to calculate percentages
    aspect_importance = aspect_counts.merge(period_totals, on='period')
    aspect_importance['percentage'] = (aspect_importance['count'] / aspect_importance['total']) * 100
    
    # Pivot for visualization
    aspect_importance_pivot = aspect_importance.pivot(index='period', 
                                                     columns='aspect', 
                                                     values='percentage')
    
    return aspect_importance, aspect_importance_pivot

# Function to plot aspect importance heatmap
def plot_aspect_importance_heatmap(aspect_importance_pivot, assistant_name):
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(aspect_importance_pivot, cmap="YlGnBu", annot=True, fmt='.1f', 
                linewidths=.5, cbar_kws={'label': 'Percentage of Reviews'})
    
    plt.title(f'Aspect Importance Over Time for {assistant_name.capitalize()}', fontsize=16)
    plt.ylabel('Time Period (Quarterly)', fontsize=14)
    plt.xlabel('Aspect', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{assistant_name}_aspect_importance_heatmap.png"), dpi=300)
    plt.close()

# Process each assistant's data
for assistant_name, files in input_files.items():
    print(f"\n🔍 Processing {assistant_name} for Aspect-Based Sentiment Analysis...")
    
    # Load data
    df = pd.read_csv(files["with_topics"])
    
    # Load topic terms
    with open(files["topic_terms"], "rb") as f:
        topic_terms = pickle.load(f)
    
    # Create taxonomy embeddings
    print("Creating taxonomy embeddings...")
    category_descriptions, category_embeddings = create_taxonomy_embeddings(taxonomy)
    
    # Map topics to taxonomy categories
    topic_to_category, topic_to_category_scores = map_topics_to_taxonomy_multi(topic_terms, category_embeddings)
    
    # Add category information to the dataframe
    df['category'] = df['topic'].map(topic_to_category)
    
    # Save the enhanced dataframe
    output_path = os.path.join(output_dir, f"{assistant_name}_with_categories.csv")
    df.to_csv(output_path, index=False)
    
    # Create report of topic-to-category mapping
    topic_category_df = pd.DataFrame({
        'topic_id': list(topic_to_category.keys()),
        'category': list(topic_to_category.values())
    })
    
    # Add topic keywords for reference
    topic_keywords = {}
    for topic_id in topic_terms:
        top_words = sorted(topic_terms[topic_id].items(), key=lambda x: x[1], reverse=True)[:10]
        topic_keywords[topic_id] = ", ".join([word for word, _ in top_words])
    
    topic_category_df['keywords'] = topic_category_df['topic_id'].map(topic_keywords)
    
    # Save topic-category mapping
    topic_category_df.to_csv(os.path.join(output_dir, f"{assistant_name}_topic_category_mapping.csv"), index=False)
    
    # Analyze category distribution
    category_counts = df['category'].value_counts()
    category_percent = df['category'].value_counts(normalize=True) * 100
    
    category_distribution = pd.DataFrame({
        'count': category_counts,
        'percentage': category_percent
    }).reset_index().rename(columns={'index': 'category'})
    
    category_distribution.to_csv(os.path.join(output_dir, f"{assistant_name}_category_distribution.csv"), index=False)
    
    # Plot category distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(x='category', y='percentage', data=category_distribution.sort_values('percentage', ascending=False))
    plt.title(f'Category Distribution for {assistant_name.capitalize()}', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Percentage of Reviews', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{assistant_name}_category_distribution.png"), dpi=300)
    plt.close()
    
    # Analyze aspect-level sentiment
    print("Analyzing aspect-level sentiment over time...")
    aspect_analysis, aspect_sentiment_pivot, mentions_pivot = analyze_multi_aspect_sentiment_over_time(df, pairs_df)
    
    # Save aspect sentiment data
    aspect_analysis.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment.csv"), index=False)
    aspect_sentiment_pivot.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment_pivot.csv"))
    
    # Plot aspect sentiment trends
    plot_aspect_sentiment_trends(aspect_sentiment_pivot, assistant_name)
    
    # Analyze aspect importance over time
    print("Analyzing aspect importance over time...")
    aspect_importance, aspect_importance_pivot = analyze_aspect_importance(df, topic_to_category)
    
    # Save aspect importance data
    aspect_importance.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_importance.csv"), index=False)
    aspect_importance_pivot.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_importance_pivot.csv"))
    
    # Plot aspect importance heatmap
    plot_aspect_importance_heatmap(aspect_importance_pivot, assistant_name)
    
    # Create detailed aspect-based sentiment report
    print("Creating detailed aspect-based sentiment report...")
    
    # Identify top positive and negative keywords per aspect
    aspect_sentiment_keywords = defaultdict(lambda: {'positive': [], 'negative': []})
    
    # For each aspect, find reviews with highest/lowest sentiment
    for aspect in df['category'].unique():
        aspect_df = df[df['category'] == aspect]
        
        # Get positive examples
        positive_examples = aspect_df[aspect_df['sentiment'] == 'positive'].sort_values('sentiment_score', ascending=False).head(50)
        if not positive_examples.empty:
            # Extract common terms from positive reviews
            positive_text = " ".join(positive_examples['clean_content'])
            # Simple term extraction (could be improved with TF-IDF or other methods)
            positive_terms = re.findall(r'\b[a-zA-Z]{3,15}\b', positive_text.lower())
            term_counts = pd.Series(positive_terms).value_counts()
            # Filter out stopwords and common terms
            positive_keywords = [term for term in term_counts.index[:20] if term not in ['the', 'and', 'for', 'this', 'that', 'with', 'have', 'you', 'not', 'are', 'but']]
            aspect_sentiment_keywords[aspect]['positive'] = positive_keywords[:10]
        
        # Get negative examples
        negative_examples = aspect_df[aspect_df['sentiment'] == 'negative'].sort_values('sentiment_score').head(50)
        if not negative_examples.empty:
            # Extract common terms from negative reviews
            negative_text = " ".join(negative_examples['clean_content'])
            negative_terms = re.findall(r'\b[a-zA-Z]{3,15}\b', negative_text.lower())
            term_counts = pd.Series(negative_terms).value_counts()
            # Filter out stopwords and common terms
            negative_keywords = [term for term in term_counts.index[:20] if term not in ['the', 'and', 'for', 'this', 'that', 'with', 'have', 'you', 'not', 'are', 'but']]
            aspect_sentiment_keywords[aspect]['negative'] = negative_keywords[:10]
    
    # Create a DataFrame with aspect sentiment keywords
    aspect_keyword_rows = []
    for aspect, sentiment_dict in aspect_sentiment_keywords.items():
        aspect_keyword_rows.append({
            'aspect': aspect,
            'positive_keywords': ', '.join(sentiment_dict['positive']),
            'negative_keywords': ', '.join(sentiment_dict['negative'])
        })
    
    aspect_keyword_df = pd.DataFrame(aspect_keyword_rows)
    aspect_keyword_df.to_csv(os.path.join(output_dir, f"{assistant_name}_aspect_sentiment_keywords.csv"), index=False)
    
    print(f"✅ ABSA processing for {assistant_name} completed. Results saved to {output_dir}")

print("\n🎉 Aspect-Based Sentiment Analysis complete for all assistants!")