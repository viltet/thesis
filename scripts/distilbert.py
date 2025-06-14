import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from tqdm.auto import tqdm # Use tqdm.auto for notebook/script compatibility
import time
from pathlib import Path 

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
print(f"Loading model: {model_name}")
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval() # Set model to evaluation mode

# --- Path Setup ---
# Dynamically gets the current script's folder (e.g., 'thesis/scripts/')
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up one level to get the main project root (e.g., 'thesis/')
PROJECT_ROOT_DIR = SCRIPT_DIR.parent

# Input files are inside 'data' folder within the PROJECT_ROOT_DIR
input_data_dir = PROJECT_ROOT_DIR / "data"
input_files = {
    "alexa": input_data_dir / "alexa_processed.csv",
    "google": input_data_dir / "google_processed.csv"
}

# Output files go into 'results' folder within the PROJECT_ROOT_DIR
output_results_dir = PROJECT_ROOT_DIR / "results"
output_results_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists
# --- End of Corrected Path Setup ---

# Parameters
batch_size = 32 # Adjust based on your available RAM/VRAM
POSITIVE_THRESHOLD = 0.75
NEGATIVE_THRESHOLD = 0.30
sentiment_label_map = {-1: "negative", 0: "neutral", 1: "positive"}

# --- Function for batched sentiment inference ---
def batch_sentiment_analysis(texts, batch_size=batch_size, device=device):
    predictions = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        probs = softmax(outputs.logits, dim=1)
        for prob in probs:
            pos_prob = prob[1].item()
            if pos_prob >= POSITIVE_THRESHOLD:
                predictions.append(1)
            elif pos_prob <= NEGATIVE_THRESHOLD:
                predictions.append(-1)
            else:
                predictions.append(0)
    return predictions

# --- Main Processing Loop ---
print(f"\nProject Root Directory: {PROJECT_ROOT_DIR}")
print(f"Input Data Directory: {input_data_dir}")
print(f"Output Results Directory: {output_results_dir}\n")

for name, path_to_input_file in input_files.items(): # Renamed 'path' to 'path_to_input_file' for clarity
    print(f"🔍 Processing {name} reviews from {path_to_input_file}...")

    try:
        df = pd.read_csv(path_to_input_file)
        if 'clean_content' not in df.columns:
             raise ValueError(f"'clean_content' column not found in {path_to_input_file}. Please ensure data is preprocessed.")
        texts = df['clean_content'].astype(str).fillna("").tolist()
        print(f"Loaded {len(df)} reviews.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {path_to_input_file}. Skipping {name}.")
        continue
    except ValueError as ve:
         print(f"Error processing {path_to_input_file}: {ve}. Skipping {name}.")
         continue
    except Exception as e:
        print(f"Error loading or processing {path_to_input_file}: {e}. Skipping {name}.")
        continue

    start_time = time.time()
    sentiment_preds_numerical = batch_sentiment_analysis(texts, batch_size, device)
    execution_time = time.time() - start_time
    print(f"Sentiment analysis for {name} completed in {execution_time:.2f} seconds.")

    df["sentiment"] = [sentiment_label_map[p] for p in sentiment_preds_numerical]
    df["sentiment_score"] = sentiment_preds_numerical

    # Save results using the corrected output_results_dir
    output_file_name = f"{name}_sentiment.csv"
    # This now correctly uses the 'output_results_dir' defined at the top
    output_path_final = output_results_dir / output_file_name
    
    df.to_csv(output_path_final, index=False)

    print(f"✅ Sentiment results for {name} saved to {output_path_final}")
    print("-" * 30)

print("\nSentiment analysis for all files complete.")