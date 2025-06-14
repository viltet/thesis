import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tqdm import tqdm
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='kneed') # Suppress kneed warning if no knee found

# For topic modeling
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from umap import UMAP
from hdbscan import HDBSCAN

# For LDA
import gensim
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# For elbow detection
try:
    from kneed import KneeLocator
except ImportError:
    print("Installing kneed for elbow detection...")
    import subprocess
    subprocess.check_call(["pip", "install", "kneed"])
    from kneed import KneeLocator

def find_elbow_point(x, y, curve="concave", direction="increasing"):
    """Find elbow point using the kneedle algorithm"""
    if not x or not y or len(x) != len(y) or len(x) < 2 : # kneed needs at least 2 points
        print("Warning: Not enough data points for kneedle. Falling back to max value.")
        return x[np.argmax(y)] if x else None
    try:
        kl = KneeLocator(x, y, curve=curve, direction=direction, S=1.0) # S sensitivity parameter
        # If kl.elbow is None, it means no knee was found. Fallback to max.
        return kl.elbow if kl.elbow else x[np.argmax(y)]
    except Exception as e:
        print(f"Kneedle failed: {e}. Falling back to max value.")
        # Fallback to simple max if kneedle fails for any reason
        return x[np.argmax(y)]

def comprehensive_lda_evaluation(corpus, dictionary, texts, min_topics=2, max_topics=25):
    """
    Systematic LDA topic number determination using multiple metrics and elbow method
    """
    topic_range = list(range(min_topics, max_topics + 1, 2))
    if not topic_range: # Ensure topic_range is not empty
        return {
            'topic_range': [], 'coherence_scores': [], 'perplexity_scores': [],
            'optimal_topics': None, 'max_coherence_topics': None
        }

    coherence_scores = []
    perplexity_scores = []
    
    print(f"Evaluating LDA with {len(topic_range)} different topic numbers...")
    
    for num_topics in tqdm(topic_range, desc="LDA Topic Evaluation"):
        try:
            lda_model = gensim.models.LdaModel(
                corpus=corpus, id2word=dictionary, num_topics=num_topics,
                random_state=42, passes=10, alpha='auto', eta='auto',
                chunksize=100, # Added for potentially better memory management / speed
                iterations=50 # Explicitly set iterations
            )
            coherence_model = CoherenceModel(
                model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v'
            )
            coherence_scores.append(coherence_model.get_coherence())
            perplexity_scores.append(lda_model.log_perplexity(corpus))
        except Exception as e:
            print(f"LDA failed for {num_topics} topics: {e}")
            coherence_scores.append(np.nan) # Use NaN for failed runs
            perplexity_scores.append(np.nan)

    # Filter out NaNs before finding elbow or max
    valid_indices = [i for i, score in enumerate(coherence_scores) if not np.isnan(score)]
    valid_topic_range = [topic_range[i] for i in valid_indices]
    valid_coherence_scores = [coherence_scores[i] for i in valid_indices]

    optimal_topics_val = None
    max_coherence_topics_val = None

    if valid_topic_range and valid_coherence_scores:
        optimal_topics_val = find_elbow_point(valid_topic_range, valid_coherence_scores)
        max_coherence_topics_val = valid_topic_range[np.argmax(valid_coherence_scores)]
    
    return {
        'topic_range': topic_range,
        'coherence_scores': coherence_scores, # Original scores with NaNs
        'perplexity_scores': perplexity_scores, # Original scores with NaNs
        'optimal_topics': optimal_topics_val,
        'max_coherence_topics': max_coherence_topics_val
    }

def optimize_bertopic_parameters(docs, embedding_model, min_cluster_range=[15, 20, 25, 30, 40]):
    """
    Systematic BERTopic parameter optimization using clustering quality metrics
    """
    results = []
    best_score = -float('inf') # Initialize best_score to negative infinity
    best_params = None
    best_model = None # Note: Storing many BERTopic models can be memory intensive
    
    print(f"Optimizing BERTopic parameters across {len(min_cluster_range)} cluster sizes...")
    
    print("Computing document embeddings...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    
    for min_cluster_size in tqdm(min_cluster_range, desc="BERTopic Optimization"):
        for min_samples_val in [5, 10]:
            try:
                umap_model = UMAP(
                    n_neighbors=15, n_components=5, min_dist=0.0, 
                    metric='cosine', random_state=42, low_memory=True # Added low_memory
                )
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_cluster_size, min_samples=min_samples_val,
                    metric='euclidean', gen_min_span_tree=True, prediction_data=True
                )
                # Adjust CountVectorizer min_df based on min_cluster_size to avoid errors
                # If min_cluster_size is very small, min_df=5 might be too large.
                current_min_df = max(2, min(5, int(min_cluster_size * 0.1))) # Heuristic: min_df is at most 10% of min_cluster_size, but at least 2
                
                vectorizer_model = CountVectorizer(
                    stop_words="english", min_df=current_min_df, ngram_range=(1, 2),
                    max_df=0.95 # Added max_df to prevent overly common (but not stopword) terms dominating
                )
                
                topic_model = BERTopic(
                    embedding_model=None, umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model, calculate_probabilities=False, # False for speed if probs not needed
                    verbose=False
                )
                topics, _ = topic_model.fit_transform(docs, embeddings)
                
                # Filter out outliers for silhouette calculation
                non_outlier_mask = np.array(topics) != -1
                # Need at least 2 clusters and more samples than clusters for silhouette
                if non_outlier_mask.sum() > 1 and len(set(np.array(topics)[non_outlier_mask])) > 1 and \
                   non_outlier_mask.sum() > len(set(np.array(topics)[non_outlier_mask])):

                    # Simpler: use full embeddings subset if UMAP already ran internally
                    silhouette_avg = silhouette_score(embeddings[non_outlier_mask], np.array(topics)[non_outlier_mask])
                    calinski_score = calinski_harabasz_score(embeddings[non_outlier_mask], np.array(topics)[non_outlier_mask])

                else:
                    silhouette_avg = -1 # Indicates poor clustering or not enough data
                    calinski_score = 0
                
                num_topics = len(set(topics)) - (1 if -1 in topics else 0)
                outlier_ratio = (np.array(topics) == -1).sum() / len(topics) if len(topics) > 0 else 1.0
                
                composite_score = silhouette_avg * (1 - outlier_ratio) if silhouette_avg != -1 else -1
                
                result = {
                    'min_cluster_size': min_cluster_size, 'min_samples': min_samples_val,
                    'num_topics': num_topics, 'silhouette_score': silhouette_avg,
                    'calinski_score': calinski_score, 'outlier_ratio': outlier_ratio,
                    'composite_score': composite_score
                }
                results.append(result)
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_params = (min_cluster_size, min_samples_val)
                    # best_model = topic_model # Storing all models can be very memory intensive.
                                            # Only store if you absolutely need the model object later.
                                            # Otherwise, re-train with best_params at the end.
                    
            except Exception as e:
                print(f"BERTopic Failed with min_cluster_size={min_cluster_size}, min_samples={min_samples_val}: {e}")
                # Append a result indicating failure for these parameters
                results.append({
                    'min_cluster_size': min_cluster_size, 'min_samples': min_samples_val,
                    'num_topics': 0, 'silhouette_score': -1, 'calinski_score': 0,
                    'outlier_ratio': 1.0, 'composite_score': -1, 'error': str(e)
                })
    
    return pd.DataFrame(results), best_params # Removed best_model from return if not storing all

def create_comprehensive_visualizations(lda_results, bertopic_results_df, dataset_name, output_dir):
    """Create comprehensive diagnostic visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12)) # Increased figure size slightly
    fig.suptitle(f'Topic Modeling Analysis - {dataset_name.capitalize()}', fontsize=18)
    
    # 1. LDA Coherence Analysis
    if lda_results and lda_results.get('topic_range') and lda_results.get('coherence_scores'):
        # Filter out NaNs for plotting LDA coherence
        valid_lda_indices = [i for i, score in enumerate(lda_results['coherence_scores']) if not np.isnan(score)]
        plot_lda_topic_range = [lda_results['topic_range'][i] for i in valid_lda_indices]
        plot_lda_coherence_scores = [lda_results['coherence_scores'][i] for i in valid_lda_indices]

        if plot_lda_topic_range and plot_lda_coherence_scores:
            axes[0,0].plot(plot_lda_topic_range, plot_lda_coherence_scores, 'o-', color='blue', label='Coherence (C_v)')
            if lda_results.get('optimal_topics') is not None:
                 axes[0,0].axvline(x=lda_results['optimal_topics'], color='red', linestyle='--', 
                                   label=f'Elbow: {lda_results["optimal_topics"]} topics')
            if lda_results.get('max_coherence_topics') is not None:
                 axes[0,0].axvline(x=lda_results['max_coherence_topics'], color='orange', linestyle=':', 
                                   label=f'Max: {lda_results["max_coherence_topics"]} topics')
            axes[0,0].set_xlabel('Number of Topics')
            axes[0,0].set_ylabel('Coherence Score (C_v)')
            axes[0,0].set_title('LDA Coherence Analysis')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        else:
            axes[0,0].text(0.5, 0.5, "LDA Coherence data unavailable", ha='center', va='center')
            axes[0,0].set_title('LDA Coherence Analysis')
    else:
        axes[0,0].text(0.5, 0.5, "LDA Coherence data unavailable", ha='center', va='center')
        axes[0,0].set_title('LDA Coherence Analysis')

    # 2. LDA Perplexity Analysis
    if lda_results and lda_results.get('topic_range') and lda_results.get('perplexity_scores'):
        valid_lda_perp_indices = [i for i, score in enumerate(lda_results['perplexity_scores']) if not np.isnan(score)]
        plot_lda_perp_topic_range = [lda_results['topic_range'][i] for i in valid_lda_perp_indices]
        plot_lda_perplexity_scores = [lda_results['perplexity_scores'][i] for i in valid_lda_perp_indices]

        if plot_lda_perp_topic_range and plot_lda_perplexity_scores:
            axes[0,1].plot(plot_lda_perp_topic_range, plot_lda_perplexity_scores, 's-', color='green')
            axes[0,1].set_xlabel('Number of Topics')
            axes[0,1].set_ylabel('Log Perplexity (lower is better)')
            axes[0,1].set_title('LDA Perplexity Analysis')
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, "LDA Perplexity data unavailable", ha='center', va='center')
            axes[0,1].set_title('LDA Perplexity Analysis')
    else:
        axes[0,1].text(0.5, 0.5, "LDA Perplexity data unavailable", ha='center', va='center')
        axes[0,1].set_title('LDA Perplexity Analysis')
    
    # 3. BERTopic Parameter Optimization Heatmap - Silhouette
    if not bertopic_results_df.empty and 'min_cluster_size' in bertopic_results_df and \
       'min_samples' in bertopic_results_df and 'silhouette_score' in bertopic_results_df:
        try:
            plot_df_silhouette = bertopic_results_df.copy()
            if 'error' in plot_df_silhouette.columns:
                plot_df_silhouette = plot_df_silhouette[plot_df_silhouette['error'].isna()]
            
            if not plot_df_silhouette.empty and not plot_df_silhouette.dropna(subset=['silhouette_score']).empty : # Ensure there's data to pivot
                # Ensure min_cluster_size and min_samples have enough unique values to form a 2D pivot
                if plot_df_silhouette['min_cluster_size'].nunique() > 0 and plot_df_silhouette['min_samples'].nunique() > 0:
                    pivot_silhouette = plot_df_silhouette.pivot(index='min_cluster_size', columns='min_samples', values='silhouette_score')
                    sns.heatmap(pivot_silhouette, annot=True, fmt='.2f', cmap='viridis', ax=axes[0,2], cbar_kws={'label': 'Silhouette Score'})
                else:
                    axes[0,2].text(0.5, 0.5, "Not enough unique params for BERTopic Silhouette heatmap", ha='center', va='center')
            else:
                 axes[0,2].text(0.5, 0.5, "No valid data for BERTopic Silhouette heatmap", ha='center', va='center')
        except Exception as e:
            print(f"Could not generate BERTopic silhouette heatmap: {e}")
            axes[0,2].text(0.5, 0.5, "BERTopic silhouette data error", ha='center', va='center')
        axes[0,2].set_title('BERTopic Silhouette Scores')
    else:
        axes[0,2].text(0.5, 0.5, "BERTopic silhouette data unavailable", ha='center', va='center')
        axes[0,2].set_title('BERTopic Silhouette Scores')

    # 4. BERTopic Parameter Optimization Heatmap - Outlier Ratio
    if not bertopic_results_df.empty and 'min_cluster_size' in bertopic_results_df and \
       'min_samples' in bertopic_results_df and 'outlier_ratio' in bertopic_results_df:
        try:
            plot_df_outlier = bertopic_results_df.copy()
            if 'error' in plot_df_outlier.columns:
                plot_df_outlier = plot_df_outlier[plot_df_outlier['error'].isna()]

            if not plot_df_outlier.empty and not plot_df_outlier.dropna(subset=['outlier_ratio']).empty:
                if plot_df_outlier['min_cluster_size'].nunique() > 0 and plot_df_outlier['min_samples'].nunique() > 0:
                    pivot_outlier = plot_df_outlier.pivot(index='min_cluster_size', columns='min_samples', values='outlier_ratio')
                    sns.heatmap(pivot_outlier, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,0], cbar_kws={'label': 'Outlier Ratio'})
                else:
                    axes[1,0].text(0.5, 0.5, "Not enough unique params for BERTopic Outlier heatmap", ha='center', va='center')
            else:
                axes[1,0].text(0.5, 0.5, "No valid data for BERTopic Outlier heatmap", ha='center', va='center')
        except Exception as e:
            print(f"Could not generate BERTopic outlier heatmap: {e}")
            axes[1,0].text(0.5, 0.5, "BERTopic outlier data error", ha='center', va='center')
        axes[1,0].set_title('BERTopic Outlier Ratios')
    else:
        axes[1,0].text(0.5, 0.5, "BERTopic outlier data unavailable", ha='center', va='center')
        axes[1,0].set_title('BERTopic Outlier Ratios')
    
    # 5. Topic Count Comparison
    comparison_data = []
    lda_optimal = lda_results.get('optimal_topics') if lda_results else None
    lda_max_coh = lda_results.get('max_coherence_topics') if lda_results else None

    if lda_optimal is not None:
        comparison_data.append({'Method': 'LDA (Elbow)', 'Topics': lda_optimal})
    if lda_max_coh is not None:
        comparison_data.append({'Method': 'LDA (Max Coherence)', 'Topics': lda_max_coh})

    best_bertopic_num_topics = None
    if not bertopic_results_df.empty and 'composite_score' in bertopic_results_df and 'num_topics' in bertopic_results_df:
        valid_composite_scores = bertopic_results_df.dropna(subset=['composite_score'])
        if not valid_composite_scores.empty:
            # Ensure composite_score is numeric before idxmax
            valid_composite_scores['composite_score'] = pd.to_numeric(valid_composite_scores['composite_score'], errors='coerce')
            valid_composite_scores = valid_composite_scores.dropna(subset=['composite_score']) # Drop if coercion failed
            if not valid_composite_scores.empty:
                best_bertopic_idx = valid_composite_scores['composite_score'].idxmax()
                best_bertopic_num_topics = valid_composite_scores.loc[best_bertopic_idx, 'num_topics']
                comparison_data.append({'Method': 'BERTopic (Optimized)', 'Topics': best_bertopic_num_topics})
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        if not comp_df.empty:
            sns.barplot(data=comp_df, x='Method', y='Topics', ax=axes[1,1], palette="muted")
            axes[1,1].set_title('Optimal Topic Counts by Method')
            axes[1,1].tick_params(axis='x', rotation=30) # REMOVED ha='right'
            for p in axes[1,1].patches:
                 axes[1,1].annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        else:
            axes[1,1].text(0.5, 0.5, "Topic count comparison data empty after processing", ha='center', va='center')    
    else:
        axes[1,1].text(0.5, 0.5, "Topic count comparison data unavailable", ha='center', va='center')
    axes[1,1].set_title('Optimal Topic Counts by Method')
    
    # 6. BERTopic Quality Metrics Scatter Plot
    if not bertopic_results_df.empty and 'silhouette_score' in bertopic_results_df and \
       'num_topics' in bertopic_results_df and 'outlier_ratio' in bertopic_results_df:
        
        plot_df_scatter = bertopic_results_df.copy()
        if 'error' in plot_df_scatter.columns: 
            plot_df_scatter = plot_df_scatter[plot_df_scatter['error'].isna()]
        
        plot_df_scatter = plot_df_scatter[plot_df_scatter['silhouette_score'] != -1]

        if not plot_df_scatter.empty:
            scatter_plot = axes[1,2].scatter(
                plot_df_scatter['silhouette_score'], plot_df_scatter['num_topics'], 
                c=plot_df_scatter['outlier_ratio'], cmap='coolwarm', alpha=0.7, s=50 
            )
            axes[1,2].set_xlabel('Silhouette Score')
            axes[1,2].set_ylabel('Number of Topics')
            axes[1,2].set_title('BERTopic: Silhouette vs. Topic Count (Color: Outlier Ratio)')
            fig.colorbar(scatter_plot, ax=axes[1,2], label='Outlier Ratio') 
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, "BERTopic quality data unavailable for scatter plot", ha='center', va='center')
            axes[1,2].set_title('BERTopic Quality vs Topic Count')
    else:
        axes[1,2].text(0.5, 0.5, "BERTopic quality data unavailable", ha='center', va='center')
        axes[1,2].set_title('BERTopic Quality vs Topic Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_comprehensive_analysis.png"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def main():
    # Configuration
    # Path to the directory containing this script
    script_dir = Path(__file__).resolve().parent

    # Path to the main 'thesis' root folder (one level up from 'scripts')
    thesis_root_dir = script_dir.parent

    # Define the 'results' directory under the 'thesis' root
    results_dir = thesis_root_dir / "results"

    input_files = {
        "alexa": results_dir / "alexa_sentiment.csv",
        "google": results_dir / "google_sentiment.csv"
    }

    # Output directory is the same 'results' directory
    output_dir = results_dir
    output_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    
    np.random.seed(42)
    SAMPLE_SIZE = 2000 # Keep sample size small for testing, increase for final run
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_results_summary = []
    
    for name, path in input_files.items():
        print(f"\n{'='*60}")
        print(f"🧪 COMPREHENSIVE TOPIC MODELING ANALYSIS: {name.upper()}")
        print(f"{'='*60}")
        
        if not os.path.exists(path):
            print(f"SKIPPING: Input file not found for {name}: {path}")
            continue
            
        df = pd.read_csv(path, on_bad_lines='skip') # Handle potential bad lines in CSV
        if 'clean_content' not in df.columns:
            print(f"SKIPPING: 'clean_content' column not found in {name}.csv")
            continue
        
        # Ensure 'clean_content' is string and drop NA before sampling
        df['clean_content'] = df['clean_content'].astype(str)
        df.dropna(subset=['clean_content'], inplace=True)
        
        df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42) if len(df) > 0 else pd.DataFrame()
        if df_sample.empty:
            print(f"SKIPPING: No data after sampling for {name}.")
            continue

        docs = df_sample['clean_content'].tolist()
        docs = [doc for doc in docs if len(doc.strip()) > 10] # Filter very short docs
        if not docs:
            print(f"SKIPPING: No valid documents after filtering for {name}.")
            continue
            
        print(f"Processing {len(docs)} documents for {name}...")
        
        # --- LDA COMPREHENSIVE EVALUATION ---
        print(f"\n🔍 PHASE 1: LDA Systematic Evaluation for {name.upper()}")
        start_time_lda = time.time()
        
        tokenized_texts = [[token for token in doc.lower().split() if len(token) > 2 and token.isalpha()] for doc in docs] # Basic preprocessing
        dictionary = Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=max(2, int(len(docs)*0.005)), no_above=0.6) # Dynamic no_below
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        if not corpus or not dictionary:
            print(f"SKIPPING LDA for {name}: Empty corpus or dictionary after preprocessing.")
            lda_results_data = {}
            lda_time_val = time.time() - start_time_lda
        else:
            lda_results_data = comprehensive_lda_evaluation(corpus, dictionary, tokenized_texts)
            lda_time_val = time.time() - start_time_lda
            if lda_results_data.get('optimal_topics'):
                print(f"Training final LDA model for {name} with {lda_results_data['optimal_topics']} topics...")
                try:
                    _ = gensim.models.LdaModel( # _ to indicate not used later
                        corpus=corpus, id2word=dictionary, num_topics=lda_results_data['optimal_topics'],
                        random_state=42, passes=15, alpha='auto', eta='auto', chunksize=100, iterations=50
                    )
                except Exception as e:
                    print(f"Final LDA training failed for {name}: {e}")
            else:
                print(f"Skipping final LDA model training for {name} as optimal topics not found.")

        # --- BERTOPIC OPTIMIZATION ---
        print(f"\n🔍 PHASE 2: BERTopic Parameter Optimization for {name.upper()}")
        start_time_bertopic = time.time()
        bertopic_results_df, best_bertopic_params = optimize_bertopic_parameters(docs, embedding_model)
        bertopic_time_val = time.time() - start_time_bertopic

        # --- RESULTS COMPILATION ---
        print(f"\n📊 PHASE 3: Results Analysis for {name.upper()}")
        create_comprehensive_visualizations(lda_results_data, bertopic_results_df, name, output_dir)
        
        current_run_summary = {'dataset': name}
        if lda_results_data:
            current_run_summary.update({
                'lda_optimal_topics': lda_results_data.get('optimal_topics'),
                'lda_max_coherence_topics': lda_results_data.get('max_coherence_topics'),
                'lda_max_coherence': max(lda_results_data.get('coherence_scores', [np.nan]), default=np.nan), # Handle empty or NaN list
                'lda_execution_time': lda_time_val,
            })
        else: # Defaults if LDA was skipped
             current_run_summary.update({
                'lda_optimal_topics': None, 'lda_max_coherence_topics': None,
                'lda_max_coherence': np.nan, 'lda_execution_time': lda_time_val,
            })


        if not bertopic_results_df.empty and 'composite_score' in bertopic_results_df:
            # Ensure 'composite_score' is numeric and handle potential NaNs from failed runs
            bertopic_results_df['composite_score'] = pd.to_numeric(bertopic_results_df['composite_score'], errors='coerce')
            valid_bertopic_runs = bertopic_results_df.dropna(subset=['composite_score'])

            if not valid_bertopic_runs.empty:
                best_bert_result_row = valid_bertopic_runs.loc[valid_bertopic_runs['composite_score'].idxmax()]
                current_run_summary.update({
                    'bertopic_optimal_topics': best_bert_result_row['num_topics'],
                    'bertopic_silhouette_score': best_bert_result_row['silhouette_score'],
                    'bertopic_outlier_ratio': best_bert_result_row['outlier_ratio'],
                    'bertopic_execution_time': bertopic_time_val,
                    'bertopic_best_params': str(best_bertopic_params),
                    'topic_count_ratio': (best_bert_result_row['num_topics'] / lda_results_data.get('optimal_topics')) 
                                         if lda_results_data.get('optimal_topics') and lda_results_data.get('optimal_topics') > 0 else np.nan
                })
            else: # If all BERTopic runs failed or resulted in NaN composite_score
                 current_run_summary.update({
                    'bertopic_optimal_topics': 0, 'bertopic_silhouette_score': -1,
                    'bertopic_outlier_ratio': 1.0, 'bertopic_execution_time': bertopic_time_val,
                    'bertopic_best_params': None, 'topic_count_ratio': np.nan, 'error': 'BERTopic optimization failed or yielded no valid results'
                })
        else: # If bertopic_results_df is empty
            current_run_summary.update({
                'bertopic_optimal_topics': 0, 'bertopic_silhouette_score': -1,
                'bertopic_outlier_ratio': 1.0, 'bertopic_execution_time': bertopic_time_val,
                'bertopic_best_params': None, 'topic_count_ratio': np.nan, 'error': 'BERTopic optimization produced no results dataframe'
            })
        
        all_results_summary.append(current_run_summary)
        
        pd.DataFrame([current_run_summary]).to_csv(
            os.path.join(output_dir, f"{name}_detailed_summary.csv"), index=False # Renamed for clarity
        )
        if not bertopic_results_df.empty:
            bertopic_results_df.to_csv(
                os.path.join(output_dir, f"{name}_bertopic_parameter_sweep.csv"), index=False
            )
        
        print(f"\n✅ ANALYSIS COMPLETE FOR {name.upper()}")
        if lda_results_data.get('optimal_topics') is not None:
            print(f"LDA Optimal Topics (Elbow Method): {lda_results_data['optimal_topics']}")
        
        if 'bertopic_optimal_topics' in current_run_summary and current_run_summary['bertopic_optimal_topics'] > 0 :
            print(f"BERTopic Optimal Topics: {current_run_summary['bertopic_optimal_topics']}")
            if current_run_summary.get('topic_count_ratio') is not np.nan:
                 print(f"Topic Count Ratio (BERTopic/LDA): {current_run_summary['topic_count_ratio']:.1f}x")
        elif 'error' in current_run_summary and 'BERTopic' in current_run_summary['error']:
            print(f"BERTopic: {current_run_summary['error']}")
        
    summary_df = pd.DataFrame(all_results_summary)
    summary_df.to_csv(os.path.join(output_dir, "final_comparison_summary.csv"), index=False)
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    if not summary_df.empty:
        print("\nFINAL SUMMARY:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()