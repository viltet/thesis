import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from tqdm.auto import tqdm # Use tqdm.auto for notebook/script compatibility
import os
import time 

# --- Configuration ---
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained DistilBERT model and tokenizer
# This model is fine-tuned on SST-2, which is binary (negative/positive)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
print(f"Loading model: {model_name}")
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval() # Set model to evaluation mode

from pathlib import Path

# Base directory for the project 
BASE_DIR = Path(__file__).resolve().parent  # Dynamically gets the current script's folder

# Input files inside the 'data' folder within the 'thesis' directory
input_files = {
    "alexa": BASE_DIR / "data" / "alexa_processed.csv",
    "google": BASE_DIR / "data" / "google_processed.csv"
}

# Output directory inside the 'results' folder within the 'thesis' directory
output_dir = BASE_DIR / "results"
# Parameters
batch_size = 32 # Adjust based on your available RAM/VRAM
# Thresholds for mapping binary output to 3 classes
# These thresholds were used in the pilot comparison code
POSITIVE_THRESHOLD = 0.75
NEGATIVE_THRESHOLD = 0.30

# Map predicted numerical sentiment (-1, 0, 1) to labels
# -1: Negative, 0: Neutral, 1: Positive
sentiment_label_map = {-1: "negative", 0: "neutral", 1: "positive"}

# --- Function for batched sentiment inference ---
def batch_sentiment_analysis(texts, batch_size=batch_size, device=device):
    """
    Runs DistilBERT sentiment analysis on texts in batches and maps to 3 classes.
    """
    predictions = []
    # Ensure model is in eval mode
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch
        # Ensure padding and truncation are handled
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512, # Or a suitable max length
            return_tensors="pt"
        )

        # Move tensors to the specified device (GPU or CPU)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # The model outputs logits for two classes (0: negative, 1: positive)
        # Convert logits to probabilities
        probs = softmax(outputs.logits, dim=1) # probs[:, 0] is neg prob, probs[:, 1] is pos prob

        # Map probabilities to our 3 sentiment classes based on thresholds
        # Iterate through the batch probabilities
        for prob in probs:
            # Probability of the positive class (label 1 in SST-2)
            pos_prob = prob[1].item()

            if pos_prob >= POSITIVE_THRESHOLD:
                predictions.append(1)   # Positive
            elif pos_prob <= NEGATIVE_THRESHOLD:
                predictions.append(-1)  # Negative
            else:
                predictions.append(0)   # Neutral

    return predictions

# --- Main Processing Loop ---
for name, path in input_files.items():
    print(f"\n🔍 Processing {name} reviews from {path}...")

    # Load data
    try:
        df = pd.read_csv(path)
        # Strictly require 'clean_content' column (no underscore)
        if 'clean_content' not in df.columns:
             raise ValueError(f"'clean_content' column not found in {path}. Please ensure data is preprocessed correctly.")

        # Convert clean_content to string and handle potential NaNs by filling with empty string
        # Using 'clean_content' (no underscore) here
        texts = df['clean_content'].astype(str).fillna("").tolist()

        print(f"Loaded {len(df)} reviews.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {path}. Skipping {name}.")
        continue
    except ValueError as ve: # Catch the specific ValueError if clean_content is missing
         print(f"Error processing {path}: {ve}. Skipping {name}.")
         continue
    except Exception as e:
        print(f"Error loading or processing {path}: {e}. Skipping {name}.")
        continue


    # Run sentiment analysis
    start_time = time.time()
    sentiment_preds_numerical = batch_sentiment_analysis(texts, batch_size, device)
    execution_time = time.time() - start_time
    print(f"Sentiment analysis for {name} completed in {execution_time:.2f} seconds.")

    # Map numerical predictions to string labels (-1, 0, 1 -> "negative", "neutral", "positive")
    df["sentiment"] = [sentiment_label_map[p] for p in sentiment_preds_numerical]
    df["sentiment_score"] = sentiment_preds_numerical # Optional: save numerical score too

    # Save results
    output_path = os.path.join(output_dir, f"{name}_sentiment.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Sentiment results for {name} saved to {output_path}")
    print("-" * 30)

print("\nSentiment analysis for all files complete.")