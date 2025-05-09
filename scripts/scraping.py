from google_play_scraper import reviews, Sort
import pandas as pd
import time
from pathlib import Path

def fetch_reviews(app_id, app_name, backup_path, final_path,
                  batch_size=100, sleep_time=1, backup_interval=1000):
    all_reviews = []
    count = 0
    continuation_token = None

    print(f"Fetching ALL reviews for {app_name}...")

    while True:
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token
        )

        if not result:
            break

        all_reviews.extend(result)
        count += len(result)
        print(f"Fetched {count} reviews for {app_name} so far...")

        if count % backup_interval < batch_size:
            df_backup = pd.DataFrame(all_reviews)
            df_backup.to_csv(backup_path, index=False)
            print(f"Auto-saved {count} reviews to {backup_path}")

        if continuation_token is None:
            break

        time.sleep(sleep_time)

    df_final = pd.DataFrame(all_reviews)
    df_final.to_csv(final_path, index=False)
    print(f"ALL reviews for {app_name} saved to {final_path}")


# === Define base directory ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)  # Create 'data' folder if it doesn't exist

# === Alexa ===
fetch_reviews(
    app_id="com.amazon.dee.app",
    app_name="Amazon Alexa",
    backup_path=DATA_DIR / "alexa_reviews_backup.csv",
    final_path=DATA_DIR / "alexa_reviews_all.csv"
)

# === Google Assistant ===
fetch_reviews(
    app_id="com.google.android.apps.googleassistant",
    app_name="Google Assistant",
    backup_path=DATA_DIR / "google_assistant_reviews_backup.csv",
    final_path=DATA_DIR / "google_assistant_reviews_all.csv"
)
