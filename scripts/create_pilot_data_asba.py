import pandas as pd
import spacy
import random
from pathlib import Path
import re # Import regular expressions module

# --- Configuration ---
# Choose which dataset to sample from ('alexa' or 'google')
ASSISTANT_NAME = "alexa" 
# Number of reviews to initially sample
N_REVIEWS_TO_SAMPLE = 100 
# Target number of aspect-sentence pairs for annotation
N_ANNOTATION_TARGET = 100 
# Random seed for reproducibility
RANDOM_SEED = 42

# --- Paths ---
# Assume the script is run from a location where THESIS_ROOT can be determined
# If running interactively, you might need to adjust this path definition
try:
    # Assumes script is in thesis/scripts or similar
    BASE_DIR = Path(__file__).resolve().parent 
    THESIS_ROOT = BASE_DIR.parent
except NameError:
    # Fallback for interactive environments (e.g., Jupyter, Colab)
    # Adjust this path manually if needed
    THESIS_ROOT = Path("./") # Or specify the absolute path to your thesis root directory
    print(f"Warning: Could not determine script location automatically. Using fallback THESIS_ROOT: {THESIS_ROOT.resolve()}")
    # Example manual path: THESIS_ROOT = Path("/path/to/your/thesis/folder")

# Define input file path
input_file = THESIS_ROOT / "results" / f"{ASSISTANT_NAME}_with_topics.csv"

# Define output directory and file path for the pilot data
output_dir = THESIS_ROOT / "results" / "absa_pilot"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{ASSISTANT_NAME}_absa_pilot_annotation.csv"

# --- Taxonomy (Simplified for keyword matching) ---
# Using the same taxonomy structure as provided previously
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

# Flatten taxonomy for quick keyword lookup
keyword_to_aspect = {}
all_keywords_patterns = []
for aspect, keywords in taxonomy.items():
    for keyword in keywords:
        # Use lower case for matching
        kw_lower = keyword.lower()
        keyword_to_aspect[kw_lower] = aspect
        # Create a regex pattern for the keyword, ensuring word boundaries
        # This prevents matching parts of words (e.g., 'art' in 'smart')
        all_keywords_patterns.append(r'\b' + re.escape(kw_lower) + r'\b')

# Combine all keyword patterns into a single regex for efficiency
keyword_regex = re.compile('|'.join(all_keywords_patterns), re.IGNORECASE)


# --- Load spaCy ---
print("Loading spaCy model (en_core_web_sm)...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please install it by running: python -m spacy download en_core_web_sm")
    exit() # Exit if model not available

# --- Main Script ---
print(f"Loading data from: {input_file}")
if not input_file.exists():
    print(f"Error: Input file not found at {input_file}")
    exit()

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} reviews.")

# Ensure 'clean_content' column exists and handle missing values
if 'clean_content' not in df.columns:
    print("Error: 'clean_content' column not found in the input CSV.")
    exit()
df = df.dropna(subset=['clean_content'])
df['clean_content'] = df['clean_content'].astype(str) # Ensure string type

# Sample reviews
print(f"Sampling {N_REVIEWS_TO_SAMPLE} reviews...")
sampled_reviews = df.sample(n=min(N_REVIEWS_TO_SAMPLE, len(df)), random_state=RANDOM_SEED)

# Prepare list to hold pilot data
pilot_data = []

print("Processing reviews to find aspect sentences...")
processed_reviews = 0
# Iterate through sampled reviews to find sentences with aspect keywords
for idx, row in sampled_reviews.iterrows():
    review_id = row.get('review_id', idx) # Use review_id if exists, else use index
    review_text = row['clean_content']
    
    if not review_text or pd.isna(review_text):
        continue # Skip empty reviews

    try:
        doc = nlp(review_text)
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) < 10: # Skip very short sentences
                continue

            # Find all keywords in the sentence using the combined regex
            found_keywords = keyword_regex.findall(sentence_text)

            # Keep track of aspects already added for this sentence to avoid duplicates
            aspects_in_sentence = set()
            
            for keyword in found_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in keyword_to_aspect:
                    aspect = keyword_to_aspect[keyword_lower]
                    # Add only if this aspect hasn't been added for this sentence yet
                    if aspect not in aspects_in_sentence:
                        pilot_data.append({
                            'review_id': review_id,
                            'sentence_text': sentence_text,
                            'identified_aspect': aspect, # Store the broader aspect category
                            'matched_keyword': keyword_lower, # Store the specific keyword found
                            'manual_sentiment': '', # Placeholder for your annotation
                            'model_predicted_sentiment': '' # Placeholder for model output later
                        })
                        aspects_in_sentence.add(aspect)

                        # Stop adding if we have enough data
                        if len(pilot_data) >= N_ANNOTATION_TARGET * 1.5: # Oversample slightly
                             break 
            if len(pilot_data) >= N_ANNOTATION_TARGET * 1.5:
                break
                
    except Exception as e:
        print(f"Warning: Error processing review ID {review_id}: {e}")
        # Continue with the next review if one fails
        continue
        
    processed_reviews += 1
    if processed_reviews % 50 == 0:
        print(f"  Processed {processed_reviews}/{N_REVIEWS_TO_SAMPLE} reviews...")
    if len(pilot_data) >= N_ANNOTATION_TARGET * 1.5:
        print(f"Reached target number of aspect sentences ({len(pilot_data)} found). Stopping review processing.")
        break

print(f"Found {len(pilot_data)} potential aspect-sentence pairs.")

# Convert to DataFrame and shuffle
pilot_df = pd.DataFrame(pilot_data)

# If we found more pairs than needed, sample down to the target
if len(pilot_df) > N_ANNOTATION_TARGET:
    print(f"Sampling down to {N_ANNOTATION_TARGET} pairs for annotation.")
    # Ensure we keep diverse sentences/aspects if possible, shuffle first
    pilot_df = pilot_df.sample(n=N_ANNOTATION_TARGET, random_state=RANDOM_SEED).reset_index(drop=True)
elif len(pilot_df) == 0:
     print("Error: No aspect sentences found with the current settings. Try increasing N_REVIEWS_TO_SAMPLE or checking the taxonomy/data.")
     exit()
else:
    print(f"Using all {len(pilot_df)} found pairs for annotation.")


# Save to CSV
print(f"Saving pilot annotation file to: {output_file}")
pilot_df.to_csv(output_file, index=False, encoding='utf-8')

print("\n--- Pilot Setup Complete ---")
print(f"1. A CSV file named '{output_file.name}' has been created in '{output_dir}'.")
print(f"2. This file contains {len(pilot_df)} sentence-aspect pairs ready for manual annotation.")
print("3. Open this CSV file in a spreadsheet editor (like Excel, Google Sheets, LibreOffice Calc).")
print("4. Follow the annotation guidelines provided separately to fill in the 'manual_sentiment' column.")
print("5. Save the file after annotation. This annotated file will be used to evaluate the ABSA model.")

