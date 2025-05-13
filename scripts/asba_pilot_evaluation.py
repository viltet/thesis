import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm # Progress bar
import torch # For checking GPU availability and model device placement
import re # For parsing InstructABSA output
import warnings # To suppress specific warnings if needed
import gc # Garbage collector

# --- Filter specific Hugging Face warnings if they become noisy ---
# warnings.filterwarnings("ignore", message=".*Your system requires specified dependencies*")
# warnings.filterwarnings("ignore", message=".*Xformers is not available*")
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated") # Suppress specific torch warning if needed

# --- Model Imports ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Use AutoModelForSequenceClassification for standard text classification tasks
# Use AutoModelForSeq2SeqLM and AutoTokenizer for text-to-text generation models
# pipeline is a high-level wrapper
from transformers import pipeline as transformers_pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# --- Evaluation Imports ---
from sklearn.metrics import classification_report, accuracy_score, f1_score

# --- Configuration ---
# Path to your ANNOTATED pilot data CSV
# !!! ENSURE THIS FILE EXISTS AND THE 'manual_sentiment' COLUMN IS FILLED !!!
ANNOTATED_PILOT_FILE = Path("./results/absa_pilot/alexa_absa_pilot_annotation.csv") # <-- ADJUST IF NEEDED

# Output file for results with predictions
RESULTS_OUTPUT_FILE = Path("./results/absa_pilot/alexa_absa_pilot_evaluation_results.csv")

# --- Model Checkpoints ---
# Baseline Sentence Sentiment Model
MODEL_DISTILBERT_SST2 = "distilbert-base-uncased-finetuned-sst-2-english"
# Dedicated ABSA Sentiment Classification Model (expects sentence, aspect pair)
MODEL_DEBERTA_ABSA = 'yangheng/deberta-v3-base-absa-v1.1'
# Zero-Shot Classification Model (using NLI) - Can specify model or use default
MODEL_ZERO_SHOT = "facebook/bart-large-mnli" # A common default

# Ensure output directory exists
RESULTS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# --- Device Setup ---
# Use GPU if available, otherwise CPU
if torch.cuda.is_available():
    DEVICE_ID = 0
    DEVICE = torch.device(f"cuda:{DEVICE_ID}")
    DEVICE_NAME = f"cuda:{DEVICE_ID}"
    # Set default device for tensors (optional, pipelines handle device placement)
    # torch.set_default_device(DEVICE)
else:
    DEVICE_ID = -1 # transformers pipeline convention for CPU
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "cpu"
print(f"Using device: {DEVICE_NAME}")

# --- Helper Functions ---

def map_vader_to_sentiment(compound_score):
    """Maps VADER compound score to Positive/Negative/Neutral."""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def map_hf_sentiment_label(label):
    """Maps common Hugging Face sentiment labels (str/int) to standard Positive/Negative/Neutral."""
    label_str = str(label).upper()
    if 'POSITIVE' in label_str or label_str == '1':
        return 'Positive'
    elif 'NEGATIVE' in label_str or label_str == '0':
        return 'Negative'
    elif 'NEUTRAL' in label_str or label_str == '2':
         return 'Neutral'
    else:
        print(f"Warning: Unexpected HF sentiment label '{label}'. Mapping to Neutral.")
        return 'Neutral'

# --- Model Prediction Functions ---

# Utility to release model memory
def release_memory(model=None, tokenizer=None, pipeline=None):
    """Releases memory occupied by models, tokenizers, and pipelines."""
    if pipeline is not None:
         del pipeline
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory released.")

def predict_vader(df):
    """Generates sentiment predictions using VADER."""
    print("\nPredicting with VADER...")
    analyzer = SentimentIntensityAnalyzer()
    predictions = []
    for sentence in tqdm(df['sentence_text'], desc="VADER"):
        vs = analyzer.polarity_scores(sentence)
        predictions.append(map_vader_to_sentiment(vs['compound']))
    return predictions

def predict_distilbert_sst2(df):
    """Generates sentiment predictions using DistilBERT-SST2."""
    print(f"\nPredicting with DistilBERT-SST2 ({MODEL_DISTILBERT_SST2})...")
    sentiment_pipeline = None
    predictions = []
    try:
        sentiment_pipeline = transformers_pipeline(
            "sentiment-analysis",
            model=MODEL_DISTILBERT_SST2,
            device=DEVICE_ID # Use pipeline device convention (-1 for CPU, 0 for cuda:0)
        )
        print("DistilBERT pipeline loaded successfully.")

        # Process in batches for efficiency
        batch_size = 32 if DEVICE_NAME != "cpu" else 8
        for i in tqdm(range(0, len(df), batch_size), desc="DistilBERT"):
             batch_sentences = df['sentence_text'][i:i+batch_size].tolist()
             try:
                 results = sentiment_pipeline(batch_sentences, truncation=True, max_length=512)
                 predictions.extend([map_hf_sentiment_label(res['label']) for res in results])
             except Exception as e:
                 print(f"Warning: Error processing batch {i//batch_size} with DistilBERT: {e}. Assigning Neutral.")
                 predictions.extend(['Neutral'] * len(batch_sentences)) # Fallback for the batch

    except Exception as e:
        print(f"Error loading or running DistilBERT pipeline: {e}")
        predictions = ['Error'] * len(df) # Mark all as Error if loading fails
    finally:
        release_memory(pipeline=sentiment_pipeline) # Clean up memory

    # Ensure correct length in case of errors
    if len(predictions) != len(df):
         print(f"Error: Length mismatch in DistilBERT predictions ({len(predictions)} vs {len(df)}). Padding with Error.")
         predictions.extend(['Error'] * (len(df) - len(predictions)))
         predictions = predictions[:len(df)] # Ensure exact length

    return predictions


def predict_deberta_absa(df):
    """
    Generates ABSA predictions using the specified DeBERTa model.
    Performs manual tokenization and inference as pipeline fails for pairs with this model.
    """
    print(f"\nPredicting with DeBERTa ABSA ({MODEL_DEBERTA_ABSA}) [Manual Tokenization]...")
    tokenizer = None
    model = None
    predictions = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DEBERTA_ABSA)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DEBERTA_ABSA).to(DEVICE)
        model.eval() # Set model to evaluation mode
        print(f"DeBERTa ABSA model and tokenizer ({MODEL_DEBERTA_ABSA}) loaded successfully.")

        # Prepare inputs as text pairs
        sentences = df['sentence_text'].tolist()
        aspects = df['identified_aspect'].tolist()

        print(f"Running DeBERTa ABSA predictions (single inference with manual tokenization)...")

        with torch.no_grad(): # Disable gradient calculation for inference
            for sentence, aspect in tqdm(zip(sentences, aspects), total=len(df), desc="DeBERTa ABSA (Manual)"):
                try:
                    # Tokenize the sentence and aspect pair together
                    inputs = tokenizer(
                        sentence,
                        aspect,
                        truncation=True,        # Truncate if combined length exceeds max_length
                        padding='max_length',   # Pad to max_length
                        max_length=512,         # Standard max length, adjust if needed based on model
                        return_tensors='pt'     # Return PyTorch tensors
                    ).to(DEVICE) # Move tokenized inputs to the selected device

                    # Perform inference
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Get the predicted class ID
                    predicted_class_id = torch.argmax(logits, dim=-1).item()

                    # Map the ID back to the sentiment label
                    pred_sentiment = model.config.id2label[predicted_class_id].capitalize()

                    # Validate the predicted sentiment label
                    if pred_sentiment not in ['Positive', 'Negative', 'Neutral']:
                        print(f"Warning: Unexpected sentiment label ID {predicted_class_id} -> '{pred_sentiment}' from DeBERTa. Mapping to Neutral.")
                        predictions.append('Neutral')
                    else:
                        predictions.append(pred_sentiment)

                except Exception as e:
                    print(f"Error during manual DeBERTa inference for aspect '{aspect}' in sentence '{sentence[:50]}...': {e}")
                    predictions.append('Error')

    except Exception as e:
        print(f"Error loading DeBERTa model/tokenizer or during prediction setup: {e}")
        predictions = ['Error'] * len(df)
    finally:
        # Release memory
        release_memory(model=model, tokenizer=tokenizer)

    # Ensure correct length
    if len(predictions) != len(df):
         print(f"Error: Length mismatch in DeBERTa ABSA predictions ({len(predictions)} vs {len(df)}). Padding with Error.")
         predictions.extend(['Error'] * (len(df) - len(predictions)))
         predictions = predictions[:len(df)] # Ensure exact length

    return predictions


def predict_zero_shot_nli(df):
    """
    Generates ABSA predictions using the Zero-Shot Classification pipeline.
    Frames the task using a hypothesis template incorporating the aspect.
    """
    print(f"\nPredicting with Zero-Shot NLI ({MODEL_ZERO_SHOT})...")
    zero_shot_pipeline = None
    predictions = []
    candidate_labels = ['Positive', 'Negative', 'Neutral'] # Fixed order for consistency

    try:
        zero_shot_pipeline = transformers_pipeline(
            "zero-shot-classification",
            model=MODEL_ZERO_SHOT,
            device=DEVICE_ID
        )
        print(f"Zero-Shot pipeline loaded successfully with model {MODEL_ZERO_SHOT}.")

        # Process sequences one by one (batching can be complex with varying hypotheses)
        # A hypothesis template that includes the aspect
        hypothesis_template = "The sentiment about {} in this text is {}." # aspect, label placeholder

        print(f"Running Zero-Shot predictions (template: '{hypothesis_template}')...")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Zero-Shot NLI"):
            sentence = row['sentence_text']
            aspect = row['identified_aspect']

            # Format the hypothesis for this specific aspect
            current_hypothesis = hypothesis_template.format(aspect, "{}") # Pipeline fills placeholder

            try:
                # Perform zero-shot classification
                result = zero_shot_pipeline(
                    sentence,
                    candidate_labels,
                    hypothesis_template=current_hypothesis,
                    multi_label=False # Ensure only the top label is considered strongly
                )

                # The result dictionary contains 'labels' and 'scores' sorted highest first
                top_label = result['labels'][0]
                if top_label not in candidate_labels:
                     print(f"Warning: Unexpected top label '{top_label}' from Zero-Shot. Mapping to Neutral.")
                     predictions.append('Neutral')
                else:
                     predictions.append(top_label)

            except Exception as e:
                 print(f"Error during Zero-Shot inference for aspect '{aspect}' in sentence '{sentence[:50]}...': {e}")
                 predictions.append('Error')

    except Exception as e:
        print(f"Error loading Zero-Shot pipeline or during prediction setup: {e}")
        predictions = ['Error'] * len(df)
    finally:
        release_memory(pipeline=zero_shot_pipeline) # Clean up memory

    # Ensure correct length
    if len(predictions) != len(df):
         print(f"Error: Length mismatch in Zero-Shot predictions ({len(predictions)} vs {len(df)}). Padding with Error.")
         predictions.extend(['Error'] * (len(df) - len(predictions)))
         predictions = predictions[:len(df)] # Ensure exact length

    return predictions

# --- Evaluation Function ---

def evaluate_model(y_true, y_pred, model_name, target_labels=['Positive', 'Negative', 'Neutral']):
    """Calculates and prints evaluation metrics, handles errors and 'Not Found'."""
    print(f"\n--- Evaluation Report for: {model_name} ---")

    # --- Data Alignment ---
    valid_indices = [
        i for i, (true, pred) in enumerate(zip(y_true, y_pred))
        if pred not in ['Error', 'Not Found'] and true in target_labels
    ]
    num_total = len(y_true)
    num_evaluated = len(valid_indices)
    num_errors = sum(1 for p in y_pred if p == 'Error')
    num_not_found = sum(1 for p in y_pred if p == 'Not Found') # Specific to InstructABSA likely
    num_invalid_true = num_total - num_evaluated - num_errors - num_not_found

    if num_evaluated < num_total:
        print(f"Evaluating on {num_evaluated}/{num_total} samples.")
        if num_errors > 0: print(f"  ({num_errors} prediction errors excluded)")
        if num_not_found > 0: print(f"  ({num_not_found} 'Not Found' predictions excluded - model didn't identify target aspect)")
        if num_invalid_true > 0: print(f"  ({num_invalid_true} invalid true labels excluded)")

    if num_evaluated == 0:
        print("No valid predictions to evaluate.")
        return None, None, None, num_evaluated, num_total

    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]

    # Determine labels present in the filtered data for the report
    present_labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
    # Ensure target_labels are included in the report if they exist in true/pred, avoids errors if a class has 0 samples
    report_labels = sorted(list(set(present_labels) & set(target_labels)))


    try:
        report = classification_report(
            y_true_filtered,
            y_pred_filtered,
            labels=report_labels, # Report only on labels present in filtered data
            digits=3,
            zero_division=0
        )
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        # Calculate weighted F1-score as a summary metric over reported labels
        weighted_f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', labels=report_labels, zero_division=0)

        print(f"Accuracy (on {num_evaluated} samples): {accuracy:.3f}")
        print(f"Weighted F1-Score (on {num_evaluated} samples): {weighted_f1:.3f}")
        print("Classification Report:")
        print(report)
        return report, accuracy, weighted_f1, num_evaluated, num_total
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print(f"True labels unique (filtered): {set(y_true_filtered)}")
        print(f"Pred labels unique (filtered): {set(y_pred_filtered)}")
        return None, None, None, num_evaluated, num_total

# --- Main Execution ---

if __name__ == "__main__":
    print(f"Starting ABSA pilot evaluation at {pd.Timestamp.now()}")
    print(f"Loading annotated pilot data from: {ANNOTATED_PILOT_FILE}")

    if not ANNOTATED_PILOT_FILE.exists():
        print(f"Error: Annotated pilot file not found at '{ANNOTATED_PILOT_FILE}'.")
        print("Please ensure the file exists, is correctly named, and the 'manual_sentiment' column is filled.")
        exit()

    try:
        pilot_df = pd.read_csv(ANNOTATED_PILOT_FILE)
        print(f"Loaded {len(pilot_df)} annotated samples.")
        # Ensure specific columns are treated as strings
        pilot_df['sentence_text'] = pilot_df['sentence_text'].astype(str)
        pilot_df['identified_aspect'] = pilot_df['identified_aspect'].astype(str)
        pilot_df['manual_sentiment'] = pilot_df['manual_sentiment'].astype(str)

    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")
        exit()

    # --- Data Validation ---
    required_cols = ['sentence_text', 'identified_aspect', 'manual_sentiment']
    if not all(col in pilot_df.columns for col in required_cols):
        print(f"Error: CSV file must contain columns: {required_cols}")
        exit()

    # Check for missing annotations BEFORE capitalization
    if pilot_df['manual_sentiment'].isnull().any() or (pilot_df['manual_sentiment'] == '').any():
        print("Error: 'manual_sentiment' column contains missing values or empty strings. Please ensure all samples are annotated.")
        num_missing = pilot_df['manual_sentiment'].isnull().sum() + (pilot_df['manual_sentiment'] == '').sum()
        print(f"Found {num_missing} missing/empty annotations.")
        exit()

    pilot_df['manual_sentiment'] = pilot_df['manual_sentiment'].str.strip().str.capitalize()
    valid_sentiments = {'Positive', 'Negative', 'Neutral'}
    actual_sentiments = set(pilot_df['manual_sentiment'].unique())

    if not actual_sentiments.issubset(valid_sentiments):
        invalid_sentiments = actual_sentiments - valid_sentiments
        print(f"Error: 'manual_sentiment' column contains invalid labels: {invalid_sentiments}.")
        print(f"Allowed labels are {valid_sentiments}. Please correct the annotations.")
        exit()
    else:
         print("Manual sentiment labels validated successfully.")


    # --- Run Predictions ---
    tqdm.pandas() # Enable progress bar for pandas apply functions if used

    all_predictions = {} # Store predictions keyed by model name

    # VADER Predictions
    all_predictions['pred_vader'] = predict_vader(pilot_df.copy())

    # DistilBERT Predictions
    all_predictions['pred_distilbert_sst2'] = predict_distilbert_sst2(pilot_df.copy())

    # DeBERTa ABSA Predictions (Manual Tokenization)
    all_predictions['pred_deberta_absa'] = predict_deberta_absa(pilot_df.copy())

    # Zero-Shot NLI Predictions <--- NEW
    all_predictions['pred_zero_shot_nli'] = predict_zero_shot_nli(pilot_df.copy())


    # --- Combine predictions with original data ---
    for col_name, preds in all_predictions.items():
        if len(preds) == len(pilot_df):
             pilot_df[col_name] = preds
        else:
             print(f"Warning: Length mismatch for {col_name}. Skipping column addition.")


    # --- Evaluate Models ---
    y_true = pilot_df['manual_sentiment'].tolist()
    results_summary = {}
    target_labels = ['Positive', 'Negative', 'Neutral'] # Define expected labels for reporting

    # Updated mapping
    model_mapping = {
         'pred_vader': f"VADER (Sentence)",
         'pred_distilbert_sst2': f"{MODEL_DISTILBERT_SST2} (Sentence)",
         'pred_deberta_absa': f"DeBERTa ABSA ({MODEL_DEBERTA_ABSA})",
         'pred_zero_shot_nli': f"Zero-Shot NLI ({MODEL_ZERO_SHOT})" # <-- NEW
    }

    for pred_col, model_name in model_mapping.items():
        if pred_col in pilot_df.columns:
             y_pred = pilot_df[pred_col].tolist()
             # Evaluation function call remains the same
             report, acc, f1, n_eval, n_total = evaluate_model(y_true, y_pred, model_name, target_labels)
             if report:
                 results_summary[model_name] = {
                     'accuracy': acc,
                     'f1_weighted': f1,
                     'report': report,
                     'evaluated_samples': n_eval,
                     'total_samples': n_total
                 }
        else:
             print(f"Skipping evaluation for {model_name} as prediction column '{pred_col}' not found.")


    # --- Save Results ---
    print(f"\nSaving evaluation results with predictions to: {RESULTS_OUTPUT_FILE}")
    try:
        # Update output columns
        output_cols = ['review_id', 'sentence_text', 'identified_aspect', 'manual_sentiment'] + \
                      sorted([pred_col for pred_col in model_mapping.keys() if pred_col in pilot_df.columns]) + \
                      [col for col in pilot_df.columns if col not in ['review_id', 'sentence_text', 'identified_aspect', 'manual_sentiment'] + list(model_mapping.keys())]
        pilot_df[output_cols].to_csv(RESULTS_OUTPUT_FILE, index=False, encoding='utf-8')
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    # --- Save Summary Report ---
    summary_path = RESULTS_OUTPUT_FILE.parent / f"{ANNOTATED_PILOT_FILE.stem}_evaluation_summary.txt"
    print(f"Saving evaluation summary report to: {summary_path}")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("===================================\n")
            f.write(" ABSA Pilot Model Evaluation Summary\n")
            f.write("===================================\n\n")
            f.write(f"Date: {pd.Timestamp.now()}\n")
            f.write(f"Annotated data file: {ANNOTATED_PILOT_FILE.name}\n")
            f.write(f"Total samples in annotated file: {len(pilot_df)}\n")
            f.write("Note: Evaluation metrics calculated only on samples where prediction was successful \n")
            f.write("      (not 'Error') and the target aspect was found (not 'Not Found' for joint models),\n")
            f.write("      and the manual label was one of ['Positive', 'Negative', 'Neutral'].\n")
            f.write("===================================\n\n")

            if results_summary:
                # Sort summary by F1-score (descending)
                sorted_results = sorted(results_summary.items(), key=lambda item: item[1].get('f1_weighted', 0), reverse=True)

                for model_name, result in sorted_results:
                     f.write(f"--- Model: {model_name} ---\n")
                     if 'accuracy' in result and 'f1_weighted' in result and 'report' in result:
                         eval_samples = result['evaluated_samples']
                         total_samples = result['total_samples']
                         f.write(f"Evaluated Samples: {eval_samples}/{total_samples}\n")
                         f.write(f"Accuracy (on evaluated): {result['accuracy']:.4f}\n")
                         f.write(f"Weighted F1-Score (on evaluated): {result['f1_weighted']:.4f}\n")
                         f.write("Classification Report (on evaluated samples):\n")
                         f.write(result['report'])
                         f.write("\n-----------------------------------\n\n")
                     else:
                         f.write("Evaluation failed or produced no valid results.\n\n")
            else:
                f.write("No models were successfully evaluated.\n")
        print("Summary report saved successfully.")
    except Exception as e:
        print(f"Error saving summary report: {e}")

    print(f"\nEvaluation complete at {pd.Timestamp.now()}. Check '{RESULTS_OUTPUT_FILE.name}' and '{summary_path.name}'.")

