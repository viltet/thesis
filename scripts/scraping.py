from google_play_scraper import reviews, Sort
import pandas as pd
import time

# App ID for Amazon Alexa
app_id = "com.amazon.dee.app"

# Absolute paths to save files
backup_filename = "/Users/viltetverijonaite/Desktop/MSC/THESIS/alexa_reviews_backup.csv"
final_filename = "/Users/viltetverijonaite/Desktop/MSC/THESIS/alexa_reviews_all.csv"

all_reviews = []
count = 0
batch_size = 100  # how many reviews per API call
sleep_time = 1   # seconds between requests
backup_interval = 1000  # how often to save progress

print("Fetching ALL reviews for Amazon Alexa...")

continuation_token = None

while True:
    result, continuation_token = reviews(
        app_id,
        lang='en',
        country='us',
        sort=Sort.NEWEST,  # newest first
        count=batch_size,
        continuation_token=continuation_token
    )

    if not result:
        break  # No more reviews

    all_reviews.extend(result)
    count += len(result)

    print(f"Fetched {count} reviews so far...")

    # Backup every backup_interval reviews
    if count % backup_interval < batch_size:
        df_backup = pd.DataFrame(all_reviews)
        df_backup.to_csv(backup_filename, index=False)
        print(f"Auto-saved {count} reviews to {backup_filename}")

    if continuation_token is None:
        break

    time.sleep(sleep_time)

# Final save
df_final = pd.DataFrame(all_reviews)
df_final.to_csv(final_filename, index=False)
print(f"ALL reviews saved to {final_filename}")
