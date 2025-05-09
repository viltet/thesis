def main():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    import os
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # For topic modeling
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    # For LDA
    import gensim
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    import pyLDAvis
    import pyLDAvis.gensim_models

    # File paths 
    input_files = {
        "alexa": "/Users/viltetverijonaite/Desktop/MSC/THESIS/alexa_sentiment.csv",
        "google": "/Users/viltetverijonaite/Desktop/MSC/THESIS/google_sentiment.csv"
    }
    output_dir = "/Users/viltetverijonaite/Desktop/MSC/THESIS/"

    # Create topic modeling comparison directory
    comparison_dir = os.path.join(output_dir, "topic_model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Sample size for comparison
    SAMPLE_SIZE = 2000

    # Function to prepare text for LDA
    def prepare_for_lda(texts):
        """Convert already cleaned texts to tokenized format for LDA"""
        processed_texts = []
        for text in tqdm(texts, desc="Preparing texts for LDA"):
            tokens = text.split()
            tokens = [token for token in tokens if len(token) > 2]
            processed_texts.append(tokens)
        return processed_texts

    # Function to evaluate topic models
    def evaluate_topic_model(model_name, model, topics, docs, dictionary=None, corpus=None, tokenized_texts=None):
        """Calculate evaluation metrics for topic models"""
        results = {"model": model_name}
        
        # Calculate topic diversity
        if model_name == "LDA":
            top_words = []
            for topic_id in range(model.num_topics):
                topic_words = [word for word, _ in model.show_topic(topic_id, 10)]
                top_words.extend(topic_words)
            
            results["topic_diversity"] = len(set(top_words)) / len(top_words) if top_words else 0
            
            coherence_model = CoherenceModel(
                model=model,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            results["coherence_score"] = coherence_model.get_coherence()
        
        elif model_name == "BERTopic":
            top_words = []
            topic_words_list = []
            
            try:
                topic_info = model.get_topic_info()
                valid_topics = topic_info[topic_info.Topic != -1]['Topic'].tolist()
            except AttributeError:
                valid_topics = list(set(topics))
                valid_topics = [t for t in valid_topics if t != -1]
            
            for topic_id in valid_topics:
                words = model.get_topic(topic_id)
                if words is not None:
                    top_words.extend([word[0] for word in words[:10]])
                    topic_words_list.append([word[0] for word in words])
            
            results["topic_diversity"] = len(set(top_words)) / len(top_words) if top_words else 0
            
            # Calculate coherence using Gensim's CoherenceModel
            if tokenized_texts is not None and dictionary is not None and len(topic_words_list) > 0:
                coherence_model = CoherenceModel(
                    topics=topic_words_list,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                results["coherence_score"] = coherence_model.get_coherence()
            else:
                results["coherence_score"] = np.nan
        
        return results

    # Process each dataset
    for name, path in input_files.items():
        print(f"\n🧪 Running topic model comparison on {name} dataset...")
        
        # Load data
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} reviews")
        
        # Take a random sample
        df_sample = df.sample(SAMPLE_SIZE, random_state=42) if len(df) > SAMPLE_SIZE else df
        print(f"Using {len(df_sample)} reviews for comparison")
        
        # Extract preprocessed data
        docs = df_sample['clean_content'].astype(str).tolist()
        docs = [doc for doc in docs if len(doc.strip()) > 0]
        
        results = []
        
        # 1. LDA Topic Modeling
        print("\n🔍 Evaluating LDA Topic Model...")
        start_time = time.time()
        
        tokenized_texts = prepare_for_lda(docs)
        dictionary = Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Find optimal number of topics
        coherence_scores = []
        topic_range = range(5, 31, 5)
        
        print("Finding optimal number of topics for LDA...")
        for num_topics in tqdm(topic_range):
            lda_model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                eta='auto'
            )
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_scores.append(coherence_model.get_coherence())
        
        optimal_num_topics = topic_range[np.argmax(coherence_scores)]
        print(f"Optimal number of topics for LDA: {optimal_num_topics}")
        
        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=optimal_num_topics,
            random_state=42,
            passes=15,
            alpha='auto',
            eta='auto'
        )
        
        lda_doc_topics = []
        for doc in corpus:
            topic_probs = lda_model.get_document_topics(doc)
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0] if topic_probs else -1
            lda_doc_topics.append(dominant_topic)
        
        lda_time = time.time() - start_time
        
        lda_results = evaluate_topic_model("LDA", lda_model, lda_doc_topics, docs, dictionary, corpus, tokenized_texts)
        lda_results["execution_time"] = lda_time
        lda_results["num_topics"] = optimal_num_topics
        results.append(lda_results)
        
        try:
            lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(lda_vis, os.path.join(comparison_dir, f"{name}_lda_visualization.html"))
        except Exception as e:
            print(f"Could not generate LDA visualization: {e}")
        
        # 2. BERTopic Modeling
        print("\n🔍 Evaluating BERTopic Model...")
        start_time = time.time()
        
        try:
            bertopic_model = BERTopic(
                embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
                umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
                hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', 
                                     gen_min_span_tree=True, prediction_data=True),
                vectorizer_model=CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)),
                calculate_probabilities=True,
                verbose=True
            )
            bertopic_topics, probs = bertopic_model.fit_transform(docs)
        except Exception as e:
            print(f"Error fitting BERTopic: {e}")
            try:
                bertopic_model = BERTopic(calculate_probabilities=True)
                bertopic_topics, probs = bertopic_model.fit_transform(docs)
            except Exception as e2:
                print(f"BERTopic failed: {e2}")
                bertopic_results = {
                    "model": "BERTopic",
                    "coherence_score": np.nan,
                    "topic_diversity": np.nan,
                    "execution_time": np.nan,
                    "num_topics": 0
                }
                results.append(bertopic_results)
                continue
        
        bertopic_time = time.time() - start_time
        
        bertopic_results = evaluate_topic_model(
            "BERTopic", bertopic_model, bertopic_topics, docs, 
            dictionary, corpus, tokenized_texts
        )
        bertopic_results["execution_time"] = bertopic_time
        bertopic_results["num_topics"] = len(set(bertopic_topics)) - (1 if -1 in bertopic_topics else 0)
        results.append(bertopic_results)
        # Save BERTopic visualizations with error handling
        try:
            # Topic map
            fig = bertopic_model.visualize_topics()
            fig.write_html(os.path.join(comparison_dir, f"{name}_bertopic_topic_map.html"))
            
            # Topic hierarchy
            try:
                fig = bertopic_model.visualize_hierarchy()
                fig.write_html(os.path.join(comparison_dir, f"{name}_bertopic_hierarchy.html"))
            except Exception as e:
                print(f"Could not generate topic hierarchy: {e}")
            
            # Topic barchart
            fig = bertopic_model.visualize_barchart(top_n_topics=15)
            fig.write_html(os.path.join(comparison_dir, f"{name}_bertopic_barchart.html"))
        except Exception as e:
            print(f"Could not generate BERTopic visualization: {e}")
        
        # Compile results into a DataFrame
        results_df = pd.DataFrame(results)
        
        # Extract top topics for each model to include in the results
        try:
            # For LDA
            lda_top_topics = []
            for i in range(min(5, optimal_num_topics)):
                topic_terms = lda_model.show_topic(i, 10)
                lda_top_topics.append(f"Topic {i}: " + ", ".join([term for term, _ in topic_terms]))
            
            # For BERTopic
            bertopic_top_topics = []
            # Get topic info safely
            if hasattr(bertopic_model, 'get_topic_info'):
                topic_info = bertopic_model.get_topic_info()
                if len(topic_info) > 5:
                    for topic_id in topic_info.iloc[1:6]['Topic'].tolist():
                        topic_terms = bertopic_model.get_topic(topic_id)
                        bertopic_top_topics.append(f"Topic {topic_id}: " + ", ".join([term for term, _ in topic_terms[:10]]))
                else:
                    # Handle case with fewer topics
                    for topic_id in topic_info['Topic'].tolist():
                        if topic_id != -1:  # Skip outlier topic
                            topic_terms = bertopic_model.get_topic(topic_id)
                            bertopic_top_topics.append(f"Topic {topic_id}: " + ", ".join([term for term, _ in topic_terms[:10]]))
            
            # Create a DataFrame for top topics
            topics_df = pd.DataFrame({
                'LDA': lda_top_topics + [''] * (5 - len(lda_top_topics)),
                'BERTopic': bertopic_top_topics + [''] * (5 - len(bertopic_top_topics))
            })
            
            # Save top topics
            topics_df.to_csv(os.path.join(comparison_dir, f"{name}_top_topics_comparison.csv"), index=False)
        except Exception as e:
            print(f"Could not extract top topics: {e}")
        
        # Save quantitative results
        results_df.to_csv(os.path.join(comparison_dir, f"{name}_topic_model_comparison.csv"), index=False)
        
        # Create visualization of comparison with error handling
        try:
            plt.figure(figsize=(12, 10))
            
            # Bar chart for coherence scores
            plt.subplot(2, 2, 1)
            sns.barplot(x='model', y='coherence_score', data=results_df)
            plt.title('Topic Coherence Score (higher is better)')
            if results_df['coherence_score'].dropna().size > 0:
                plt.ylim(0, max(results_df['coherence_score'].dropna()) * 1.2)
            
            # Bar chart for topic diversity
            plt.subplot(2, 2, 2)
            sns.barplot(x='model', y='topic_diversity', data=results_df)
            plt.title('Topic Diversity (higher is better)')
            plt.ylim(0, 1)
            
            # Bar chart for execution time
            plt.subplot(2, 2, 3)
            sns.barplot(x='model', y='execution_time', data=results_df)
            plt.title('Execution Time in Seconds (lower is better)')
            
            # Bar chart for number of topics
            plt.subplot(2, 2, 4)
            sns.barplot(x='model', y='num_topics', data=results_df)
            plt.title('Number of Topics')
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"{name}_topic_model_comparison.png"))
            
            # Produce a plot showing the coherence scores for different numbers of topics in LDA
            plt.figure(figsize=(10, 6))
            plt.plot(topic_range, coherence_scores, 'o-')
            plt.xlabel('Number of Topics')
            plt.ylabel('Coherence Score')
            plt.title(f'LDA Topic Coherence Scores by Number of Topics - {name.capitalize()}')
            plt.xticks(topic_range)
            plt.grid(True)
            plt.savefig(os.path.join(comparison_dir, f"{name}_lda_coherence_by_topics.png"))
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        print(f"✅ Topic model comparison for {name} completed.")
        print(f"Results saved to {comparison_dir}")
        print("\nComparison Results:")
        print(results_df.to_string())

    print("\n🎉 Topic model comparison complete for all datasets!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()