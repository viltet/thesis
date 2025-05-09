import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from tqdm import tqdm
from pathlib import Path


# === Configuration ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_COMPARISON_DIR = BASE_DIR / "model_comparison"

INPUT_DIR = DATA_DIR.as_posix()
LABELED_DATA_FILE = (DATA_DIR / "pilot_subset.csv").as_posix()
OUTPUT_DIR = (MODEL_COMPARISON_DIR / "results").as_posix()
MODELS_DIR = (MODEL_COMPARISON_DIR / "models").as_posix()

RANDOM_SEED = 100

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "model_comparison.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# === Helper Functions ===

def load_labeled_data(file_path):
    """
    Load the dataset with sentiment_class labels.
    """
    df = pd.read_csv(file_path)
    
    # Verify sentiment_class exists
    if 'sentiment_class' not in df.columns:
        raise ValueError("'sentiment_class' column not found in dataset.")
    
    # Convert sentiment_class (0,1,2) to sentiment (-1,0,1)
    # 0 = negative (≤ 2 stars), 1 = neutral (3 stars), 2 = positive (≥ 4 stars)
    mapping = {0: -1, 1: 0, 2: 1}
    df['sentiment'] = df['sentiment_class'].map(mapping)
    
    logger.info(f"Sentiment class distribution:")
    logger.info(df['sentiment_class'].value_counts().to_string())
    logger.info(f"Mapped sentiment distribution:")
    logger.info(df['sentiment'].value_counts().to_string())
    
    return df


def plot_confusion_matrix(cm, labels, model_name, output_dir):
    """
    Plot and save confusion matrix visualization.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate and return evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate detailed classification report
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
    logger.info(f"\nClassification Report for {model_name}:\n{report}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def plot_class_distribution(predictions_dict, true_labels, output_dir):
    """
    Plot a comparison of class distributions between models and true labels.
    """
    # Create a dataframe with predictions from all models and true labels
    df = pd.DataFrame({'True Labels': true_labels})
    
    # Add model predictions
    for model_name, preds in predictions_dict.items():
        df[model_name] = preds
    
    # Convert to long format for easier plotting
    df_long = pd.melt(df, var_name='Model', value_name='Sentiment')
    
    # Map sentiment values to readable labels
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    df_long['Sentiment'] = df_long['Sentiment'].map(sentiment_map)
    
    # Count plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Sentiment', hue='Model', data=df_long, palette='viridis')
    plt.title('Sentiment Distribution Across Models')
    plt.ylabel('Count')
    plt.legend(title='Model')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "sentiment_distribution_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved sentiment distribution comparison to {output_path}")


def calibrate_thresholds(texts, true_labels, model, tokenizer, device, validation_size=0.3):
    """
    Calibrate thresholds for binary sentiment model to optimize for three-class performance.
    Returns optimal upper and lower thresholds.
    """
    logger.info("Calibrating thresholds for optimal performance...")
    
    # Split data for threshold calibration
    if validation_size > 0:
        cal_texts, val_texts, cal_labels, val_labels = train_test_split(
            texts, true_labels, test_size=validation_size, random_state=RANDOM_SEED,
            stratify=true_labels
        )
    else:
        cal_texts, cal_labels = texts, true_labels
    
    # Get raw probabilities
    batch_size = 16
    all_probs = []
    
    for i in tqdm(range(0, len(cal_texts), batch_size), desc="Extracting probabilities"):
        batch_texts = cal_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    # For binary models (DistilBERT), the probability of positive class
    if all_probs.shape[1] == 2:
        pos_probs = all_probs[:, 1]  # Probability of positive sentiment
        
        # Try different threshold combinations
        best_f1 = 0
        best_thresholds = (0.25, 0.75)  # Default thresholds
        
        # Grid search for thresholds
        lower_thresholds = np.arange(0.05, 0.5, 0.05)
        upper_thresholds = np.arange(0.5, 0.96, 0.05)
        
        results = []
        
        for lower in lower_thresholds:
            for upper in upper_thresholds:
                if upper > lower:  # Ensure upper threshold is higher than lower
                    # Apply thresholds
                    preds = []
                    for prob in pos_probs:
                        if prob >= upper:
                            preds.append(1)      # Positive
                        elif prob <= lower:
                            preds.append(-1)     # Negative
                        else:
                            preds.append(0)      # Neutral
                    
                    # Calculate F1 score
                    _, _, f1, _ = precision_recall_fscore_support(cal_labels, preds, average='weighted')
                    results.append((lower, upper, f1))
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresholds = (lower, upper)
        
        # Plot threshold optimization results
        results_df = pd.DataFrame(results, columns=['Lower Threshold', 'Upper Threshold', 'F1 Score'])
        
        plt.figure(figsize=(10, 6))
        
        # Create a pivot table for the heatmap
        pivot_data = results_df.pivot(index='Lower Threshold', columns='Upper Threshold', values='F1 Score')
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f')
        plt.title('F1 Score by Threshold Combinations')
        plt.tight_layout()
        
        threshold_path = os.path.join(OUTPUT_DIR, "threshold_optimization.png")
        plt.savefig(threshold_path, dpi=300)
        plt.close()
        
        # Plot the distribution of probabilities by true class
        plt.figure(figsize=(12, 6))
        for label in np.unique(cal_labels):
            mask = np.array(cal_labels) == label
            sns.kdeplot(pos_probs[mask], label=f'Class {label}')
        
        plt.axvline(x=best_thresholds[0], color='r', linestyle='--', alpha=0.7, label='Lower Threshold')
        plt.axvline(x=best_thresholds[1], color='g', linestyle='--', alpha=0.7, label='Upper Threshold')
        
        plt.title('Distribution of Positive Probabilities by True Class')
        plt.xlabel('Probability of Positive Sentiment')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        prob_dist_path = os.path.join(OUTPUT_DIR, "probability_distribution.png")
        plt.savefig(prob_dist_path, dpi=300)
        plt.close()
        
        logger.info(f"Best thresholds: Lower={best_thresholds[0]:.2f}, Upper={best_thresholds[1]:.2f} (F1={best_f1:.4f})")
        
        # Calculate performance metrics with best thresholds if validation set was created
        if validation_size > 0:
            # Get raw probabilities for validation set
            val_probs = []
            for i in range(0, len(val_texts), batch_size):
                batch_texts = val_texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                val_probs.extend(probs.cpu().numpy())
            
            val_pos_probs = np.array(val_probs)[:, 1]
            
            # Apply best thresholds
            val_preds = []
            for prob in val_pos_probs:
                if prob >= best_thresholds[1]:
                    val_preds.append(1)      # Positive
                elif prob <= best_thresholds[0]:
                    val_preds.append(-1)     # Negative
                else:
                    val_preds.append(0)      # Neutral
            
            # Calculate metrics on validation set
            val_acc = accuracy_score(val_labels, val_preds)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
            
            logger.info(f"Validation metrics with calibrated thresholds: Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        return best_thresholds
    else:
        # For three-class models (like RoBERTa), no calibration needed
        logger.info("No threshold calibration needed for this model (already 3-class)")
        return None


# === Model Implementations ===

def run_vader_sentiment(texts):
    """
    Run VADER sentiment analysis on texts.
    Returns sentiment scores mapped to -1, 0, 1.
    """
    analyzer = SentimentIntensityAnalyzer()
    start_time = time.time()
    
    logger.info("Running VADER sentiment analysis...")
    predictions = []
    scores_list = []  # Store raw scores for analysis
    
    # Use tqdm for progress tracking
    for text in tqdm(texts, desc="VADER Processing"):
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        scores_list.append(scores)
        
        if compound <= -0.05:
            predictions.append(-1)  # Negative
        elif compound >= 0.05:
            predictions.append(1)   # Positive
        else:
            predictions.append(0)   # Neutral
    
    execution_time = time.time() - start_time
    logger.info(f"VADER completed in {execution_time:.2f} seconds")
    
    # Analyze score distributions
    scores_df = pd.DataFrame(scores_list)
    
    # Plot distribution of compound scores
    plt.figure(figsize=(10, 6))
    sns.histplot(scores_df['compound'], bins=50, kde=True)
    plt.axvline(x=-0.05, color='r', linestyle='--', alpha=0.7, label='Negative Threshold (-0.05)')
    plt.axvline(x=0.05, color='g', linestyle='--', alpha=0.7, label='Positive Threshold (0.05)')
    plt.title('Distribution of VADER Compound Scores')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    vader_dist_path = os.path.join(OUTPUT_DIR, "vader_score_distribution.png")
    plt.savefig(vader_dist_path, dpi=300)
    plt.close()
    
    # Print class distribution
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    logger.info(f"VADER class distribution: {pred_counts.to_dict()}")
    
    return predictions, execution_time


def run_distilbert_sentiment(texts, true_labels, device, calibrate=True):
    """
    Run DistilBERT sentiment analysis on texts.
    Returns sentiment scores mapped to -1, 0, 1.
    """
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    logger.info(f"Loading DistilBERT model: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
    
    # Calibrate thresholds (if requested)
    if calibrate and true_labels is not None:
        logger.info("Calibrating DistilBERT thresholds...")
        thresholds = calibrate_thresholds(texts, true_labels, model, tokenizer, device, validation_size=0.2)
        lower_threshold, upper_threshold = thresholds
    else:
        # Default thresholds
        lower_threshold, upper_threshold = 0.25, 0.75
        logger.info(f"Using default thresholds: Lower={lower_threshold}, Upper={upper_threshold}")
    
    start_time = time.time()
    
    # Process in batches to avoid memory issues
    batch_size = 16
    predictions = []
    all_probs = []  # For visualization
    
    logger.info(f"Running DistilBERT with batch size {batch_size}")
    
    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT Processing"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # DistilBERT fine-tuned on SST-2 gives binary sentiment (0=negative, 1=positive)
        # We map to our three-class scheme using calibrated thresholds
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        
        for prob in probs:
            if prob[1] > upper_threshold:  # Strong positive
                predictions.append(1)
            elif prob[1] < lower_threshold:  # Strong negative
                predictions.append(-1)
            else:  # Neutral/mixed
                predictions.append(0)
    
    execution_time = time.time() - start_time
    logger.info(f"DistilBERT completed in {execution_time:.2f} seconds")
    
    # Plot distribution of positive probabilities
    if len(all_probs) > 0:
        plt.figure(figsize=(10, 6))
        pos_probs = [p[1] for p in all_probs]
        sns.histplot(pos_probs, bins=50, kde=True)
        plt.axvline(x=lower_threshold, color='r', linestyle='--', alpha=0.7, 
                   label=f'Negative Threshold ({lower_threshold:.2f})')
        plt.axvline(x=upper_threshold, color='g', linestyle='--', alpha=0.7,
                   label=f'Positive Threshold ({upper_threshold:.2f})')
        plt.title('Distribution of DistilBERT Positive Class Probabilities')
        plt.xlabel('Probability of Positive Class')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        distilbert_dist_path = os.path.join(OUTPUT_DIR, "distilbert_prob_distribution.png")
        plt.savefig(distilbert_dist_path, dpi=300)
        plt.close()
    
    # Print class distribution
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    logger.info(f"DistilBERT class distribution: {pred_counts.to_dict()}")
    
    return predictions, execution_time


def run_bert_sentiment(texts, device):
    """
    Run BERT sentiment analysis on texts.
    Returns sentiment scores mapped to -1, 0, 1.
    """
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    logger.info(f"Loading BERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    start_time = time.time()
    
    # Process in batches to avoid memory issues
    batch_size = 16
    predictions = []
    all_preds = []  # Store raw predictions for analysis
    
    logger.info(f"Running BERT with batch size {batch_size}")
    
    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Processing"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # This BERT model is trained for 5-class star rating (1-5 stars)
        # Map to our scheme: 1-2 stars → negative (-1), 3 stars → neutral (0), 4-5 stars → positive (1)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        
        for pred in preds:
            star_rating = pred + 1  # Model outputs 0-4, add 1 to get 1-5 stars
            if star_rating <= 2:
                predictions.append(-1)  # Negative (1-2 stars)
            elif star_rating == 3:
                predictions.append(0)   # Neutral (3 stars)
            else:
                predictions.append(1)   # Positive (4-5 stars)
    
    execution_time = time.time() - start_time
    logger.info(f"BERT completed in {execution_time:.2f} seconds")
    
    # Plot distribution of star ratings
    plt.figure(figsize=(10, 6))
    star_ratings = [pred + 1 for pred in all_preds]  # Convert to 1-5 stars
    sns.countplot(x=star_ratings, palette='viridis')
    plt.title('Distribution of BERT Star Rating Predictions')
    plt.xlabel('Star Rating')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
    plt.tight_layout()
    
    bert_dist_path = os.path.join(OUTPUT_DIR, "bert_rating_distribution.png")
    plt.savefig(bert_dist_path, dpi=300)
    plt.close()
    
    # Print class distribution
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    logger.info(f"BERT class distribution: {pred_counts.to_dict()}")
    
    return predictions, execution_time


def run_roberta_sentiment(texts, device):
    """
    Run RoBERTa sentiment analysis on texts.
    Returns sentiment scores mapped to -1, 0, 1.
    """
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    logger.info(f"Loading RoBERTa model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    start_time = time.time()
    
    # Process in batches to avoid memory issues
    batch_size = 16
    predictions = []
    all_outputs = []  # For visualization
    
    logger.info(f"Running RoBERTa with batch size {batch_size}")
    
    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa Processing"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # RoBERTa model is trained for 3-class sentiment: 0=negative, 1=neutral, 2=positive
        # Map to our scheme: -1=negative, 0=neutral, 1=positive
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        all_outputs.extend(probs)
        
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        mapped_preds = [pred - 1 for pred in preds]  # Convert 0,1,2 to -1,0,1
        predictions.extend(mapped_preds)
    
    execution_time = time.time() - start_time
    logger.info(f"RoBERTa completed in {execution_time:.2f} seconds")
    
    # Plot distribution of class probabilities
    if len(all_outputs) > 0:
        all_outputs = np.array(all_outputs)
        
        plt.figure(figsize=(12, 6))
        
        # Create a DataFrame for easier plotting
        probs_df = pd.DataFrame({
            'Negative Prob': all_outputs[:, 0],
            'Neutral Prob': all_outputs[:, 1],
            'Positive Prob': all_outputs[:, 2]
        })
        
        # Plot distributions
        sns.kdeplot(data=probs_df, palette='viridis')
        plt.title('Distribution of RoBERTa Class Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend(title='Class')
        plt.tight_layout()
        
        roberta_dist_path = os.path.join(OUTPUT_DIR, "roberta_prob_distribution.png")
        plt.savefig(roberta_dist_path, dpi=300)
        plt.close()
    
    # Print class distribution
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    logger.info(f"RoBERTa class distribution: {pred_counts.to_dict()}")
    
    return predictions, execution_time


# === Main Script ===

def main():
    try:
        # Determine device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load labeled data
        input_path = os.path.join(INPUT_DIR, LABELED_DATA_FILE)
        logger.info(f"Loading labeled data from: {input_path}")
        df = load_labeled_data(input_path)
        logger.info(f"Loaded {len(df)} labeled examples")
        
        # Use clean_content column if available, otherwise use content
        content_column = 'clean_content' if 'clean_content' in df.columns else 'content'
        if content_column not in df.columns:
            raise ValueError(f"Neither 'clean_content' nor 'content' columns found in dataset.")
        
        # Check for empty or null values
        empty_rows = df[df[content_column].isna() | (df[content_column] == '')].index
        if len(empty_rows) > 0:
            logger.warning(f"Found {len(empty_rows)} rows with empty content. Removing these rows.")
            df = df.dropna(subset=[content_column]).reset_index(drop=True)
        
        # Convert to list of texts and labels
        texts = df[content_column].tolist()
        true_labels = df['sentiment'].tolist()  # Using our mapped sentiment values (-1,0,1)
        
        # Run models
        vader_preds, vader_time = run_vader_sentiment(texts)
        distilbert_preds, distilbert_time = run_distilbert_sentiment(texts, true_labels, device, calibrate=True)
        bert_preds, bert_time = run_bert_sentiment(texts, device)
        roberta_preds, roberta_time = run_roberta_sentiment(texts, device)
        
        # Evaluate models
        logger.info("\nEvaluating models...")
        vader_metrics = evaluate_model(true_labels, vader_preds, "VADER")
        distilbert_metrics = evaluate_model(true_labels, distilbert_preds, "DistilBERT")
        bert_metrics = evaluate_model(true_labels, bert_preds, "BERT")
        roberta_metrics = evaluate_model(true_labels, roberta_preds, "RoBERTa")
        
        # Add execution times
        vader_metrics['execution_time'] = vader_time
        distilbert_metrics['execution_time'] = distilbert_time
        bert_metrics['execution_time'] = bert_time
        roberta_metrics['execution_time'] = roberta_time
        
        # Combine results
        all_metrics = [vader_metrics, distilbert_metrics, bert_metrics, roberta_metrics]
        
        # Create summary dataframe
        metrics_df = pd.DataFrame([
            {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            for metrics in all_metrics
        ])
        
        # Plot confusion matrices
        labels = ['Negative', 'Neutral', 'Positive']
        plot_confusion_matrix(vader_metrics['confusion_matrix'], labels, "VADER", OUTPUT_DIR)
        plot_confusion_matrix(distilbert_metrics['confusion_matrix'], labels, "DistilBERT", OUTPUT_DIR)
        plot_confusion_matrix(bert_metrics['confusion_matrix'], labels, "BERT", OUTPUT_DIR)
        plot_confusion_matrix(roberta_metrics['confusion_matrix'], labels, "RoBERTa", OUTPUT_DIR)
        
        # Plot class distribution comparison
        plot_class_distribution({
            'VADER': vader_preds,
            'DistilBERT': distilbert_preds,
            'BERT': bert_preds,
            'RoBERTa': roberta_preds
        }, true_labels, OUTPUT_DIR)
        
        # Save detailed predictions for error analysis
        results_df = df.copy()
        results_df['vader_pred'] = vader_preds
        results_df['distilbert_pred'] = distilbert_preds
        results_df['bert_pred'] = bert_preds
        results_df['roberta_pred'] = roberta_preds
        
        # Mark correct/incorrect predictions
        results_df['vader_correct'] = results_df['vader_pred'] == results_df['sentiment']
        results_df['distilbert_correct'] = results_df['distilbert_pred'] == results_df['sentiment']
        results_df['bert_correct'] = results_df['bert_pred'] == results_df['sentiment']
        results_df['roberta_correct'] = results_df['roberta_pred'] == results_df['sentiment']
        
        # Add agreement column
        results_df['model_agreement'] = ((results_df['vader_pred'] == results_df['distilbert_pred']) &
                                         (results_df['distilbert_pred'] == results_df['bert_pred']) &
                                         (results_df['bert_pred'] == results_df['roberta_pred']))
        
        # Identify examples where all models failed
        results_df['all_failed'] = (~results_df['vader_correct'] &
                                   ~results_df['distilbert_correct'] &
                                   ~results_df['bert_correct'] &
                                   ~results_df['roberta_correct'])
        
        # Save results
        metrics_path = os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv")
        results_path = os.path.join(OUTPUT_DIR, "model_predictions.csv")
        
        metrics_df.to_csv(metrics_path, index=False)
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
        logger.info(f"Saved prediction results to {results_path}")
        
        # Print summary
        logger.info("\n===== Model Comparison Summary =====")
        logger.info(metrics_df.to_string(index=False))
        
        # Generate visualization of metrics
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        metrics_df_plot = metrics_df[['model'] + metrics_to_plot].set_index('model')
        ax = metrics_df_plot.plot(kind='bar', rot=0)
        plt.title('Sentiment Analysis Model Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.tight_layout()
        metrics_viz_path = os.path.join(OUTPUT_DIR, "model_comparison_metrics.png")
        plt.savefig(metrics_viz_path, dpi=300)
        logger.info(f"Saved metrics visualization to {metrics_viz_path}")
        
        # Generate execution time comparison
        plt.figure(figsize=(8, 5))
        execution_times = metrics_df[['model', 'execution_time']].set_index('model')
        ax = execution_times.plot(kind='bar', rot=0, color='green')
        plt.title('Model Execution Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        time_viz_path = os.path.join(OUTPUT_DIR, "model_execution_times.png")
        plt.savefig(time_viz_path, dpi=300)
        logger.info(f"Saved execution time visualization to {time_viz_path}")
        
        # Calculate error analysis statistics
        error_analysis = {
            'Total examples': len(results_df),
            'VADER correct': results_df['vader_correct'].sum(),
            'DistilBERT correct': results_df['distilbert_correct'].sum(),
            'BERT correct': results_df['bert_correct'].sum(),
            'RoBERTa correct': results_df['roberta_correct'].sum(),
            'All models agree': results_df['model_agreement'].sum(),
            'All models wrong': results_df['all_failed'].sum()
        }
        
        logger.info("\n===== Error Analysis =====")
        for key, value in error_analysis.items():
            logger.info(f"{key}: {value} ({value/len(results_df)*100:.2f}%)")
        
        # Create per-class error analysis
        logger.info("\n===== Per-Class Error Analysis =====")
        for sentiment_value, sentiment_name in {-1: "Negative", 0: "Neutral", 1: "Positive"}.items():
            class_df = results_df[results_df['sentiment'] == sentiment_value]
            if len(class_df) > 0:
                logger.info(f"\n{sentiment_name} class ({len(class_df)} examples):")
                logger.info(f"  VADER correct: {class_df['vader_correct'].sum()} ({class_df['vader_correct'].mean()*100:.2f}%)")
                logger.info(f"  DistilBERT correct: {class_df['distilbert_correct'].sum()} ({class_df['distilbert_correct'].mean()*100:.2f}%)")
                logger.info(f"  BERT correct: {class_df['bert_correct'].sum()} ({class_df['bert_correct'].mean()*100:.2f}%)")
                logger.info(f"  RoBERTa correct: {class_df['roberta_correct'].sum()} ({class_df['roberta_correct'].mean()*100:.2f}%)")
        
        # Export challenging examples (where all models failed)
        challenging_examples = results_df[results_df['all_failed']].copy()
        if len(challenging_examples) > 0:
            challenging_path = os.path.join(OUTPUT_DIR, "challenging_examples.csv")
            challenging_examples.to_csv(challenging_path, index=False)
            logger.info(f"\nSaved {len(challenging_examples)} challenging examples to {challenging_path}")
            
            # Sample a few challenging examples for inspection
            sample_size = min(5, len(challenging_examples))
            sample_examples = challenging_examples.sample(n=sample_size, random_state=RANDOM_SEED)
            logger.info(f"\nSample of challenging examples (all models failed):")
            for i, (_, row) in enumerate(sample_examples.iterrows(), 1):
                logger.info(f"Example {i}:")
                logger.info(f"Text: {row[content_column][:200]}{'...' if len(row[content_column]) > 200 else ''}")
                logger.info(f"True sentiment: {row['sentiment']} | Predictions: VADER={row['vader_pred']}, DistilBERT={row['distilbert_pred']}, BERT={row['bert_pred']}, RoBERTa={row['roberta_pred']}\n")
        
        # Determine best model
        best_model_idx = metrics_df['f1'].idxmax()
        best_model = metrics_df.loc[best_model_idx]['model']
        logger.info(f"\nBest performing model based on F1 score: {best_model}")
        logger.info(f"F1 Score: {metrics_df.loc[best_model_idx]['f1']:.4f}")
        
        # Print execution environment information
        logger.info("\n===== Execution Environment =====")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        logger.info("\nModel comparison completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during model comparison: {str(e)}")
        raise


if __name__ == "__main__":
    main()