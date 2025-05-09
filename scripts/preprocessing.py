import pandas as pd
import re
import emoji
import spacy
from langdetect import detect, LangDetectException
from tqdm import tqdm
import contractions
import os
from pathlib import Path


# Enable tqdm for pandas
tqdm.pandas()

# Load faster spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stopwords = nlp.Defaults.stop_words.difference({"no", "not", "never", "none", "nor", "neither"})

# Assistant-related keywords
ASSISTANT_KEYWORDS = [
    "alexa", "echo", "echo dot", "echo show", "echo plus", "echo studio", "amazon alexa",
    "alexa app", "alexa device", "alexa skill", "skills", "google assistant", "hey google",
    "ok google", "google home", "nest hub", "google device", "google nest", "google speaker",
    "google action", "actions", "home mini", "nest mini", "nest audio", "nest speaker",
    "assistant", "smart speaker", "voice assistant", "ai assistant", "digital assistant"
]
ASSISTANT_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in ASSISTANT_KEYWORDS) + r")\b", re.IGNORECASE)

def flag_assistant_mentions(text):
    return bool(ASSISTANT_PATTERN.search(text)) if isinstance(text, str) else False

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|@\w+", "", text)  # Remove URLs, mentions
    text = emoji.demojize(text, delimiters=(" ", " "))  # Demojize
    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)  # Keep alphanum and basic punctuation
    return text.strip()

def preprocess_text(text):
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    doc = nlp(cleaned)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in stopwords and not token.is_punct and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

def filter_english(df, text_column="content"):
    def is_english(text):
        try:
            return detect(text) == "en"
        except (LangDetectException, Exception):
            return False
    return df[df[text_column].progress_apply(lambda x: is_english(x) if isinstance(x, str) else False)]

def process_reviews(input_path, output_path, min_length=10):
    input_path, output_path = Path(input_path), Path(output_path)
    dataset_name = input_path.stem

    print(f"\n📥 Loading {dataset_name}...")
    df = pd.read_csv(input_path)
    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'])

    original_count = len(df)
    print(f"🔢 Original reviews: {original_count}")

    df = df.dropna(subset=["content"])

    print("🔍 Filtering English reviews...")
    df = filter_english(df)

    print("🏷️ Flagging assistant mentions...")
    df["has_assistant_mention"] = df["content"].progress_apply(flag_assistant_mentions)

    print("🧹 Cleaning and preprocessing text...")
    df["clean_content"] = df["content"].progress_apply(preprocess_text)

    df = df[df["clean_content"].str.len() > min_length]
    final_count = len(df)
    print(f"📊 Final reviews: {final_count} ({100 * final_count / original_count:.2f}%)")

    df.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")

    if not df.empty:
        print(f"\n💡 Sample cleaned review:\n{df['clean_content'].iloc[0][:200]}...")
    
    return df

def main():
    """Main execution"""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    paths = {
        "alexa": data_dir / "alexa_reviews_all.csv",
        "google": data_dir / "google_assistant_reviews_all.csv"
    }

    output_paths = {
        "alexa": data_dir / "alexa_processed.csv",
        "google": data_dir / "google_processed.csv"
    }

    start_date = pd.Timestamp("2017-10-01")
    end_date = pd.Timestamp("2025-03-31")

    for assistant, path in paths.items():
        print(f"\n🔄 Processing {assistant.capitalize()} reviews...")

        df = pd.read_csv(path, parse_dates=['at'])
        df = df[(df['at'] >= start_date) & (df['at'] <= end_date)]

        columns_to_keep = ['reviewId', 'content', 'score', 'at', 'reviewCreatedVersion', 'appVersion']
        df = df[columns_to_keep].dropna(subset=['content']).drop_duplicates(subset=['reviewId'])

        interim_path = data_dir / f"{assistant}_filtered.csv"
        df.to_csv(interim_path, index=False)

        process_reviews(interim_path, output_paths[assistant])

    print("\n✅ All processing complete!")

if __name__ == "__main__":
    main()