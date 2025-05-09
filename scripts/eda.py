import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.dates as mdates
from collections import defaultdict
import os
from matplotlib.ticker import FuncFormatter


# Output directory
OUTPUT_DIR = "/Users/viltetverijonaite/Desktop/MSC/THESIS"

# Configuration
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})
COLORS = {'alexa': '#0173b2', 'google': '#de8f05'}

def load_data():
    """Load and prepare datasets"""
    print("Loading processed datasets...")
    
    alexa = pd.read_csv(
        f"{OUTPUT_DIR}/alexa_processed.csv",
        parse_dates=['at'],
    )
    google = pd.read_csv(
        f"{OUTPUT_DIR}/google_processed.csv",
        parse_dates=['at'],
    )
    
    # Filter valid scores
    for df in [alexa, google]:
        df.dropna(subset=['score'], inplace=True)
        df['score'] = df['score'].astype(int).clip(1, 5)
    
    print(f"Loaded {len(alexa):,} Alexa reviews and {len(google):,} Google Assistant reviews")
    return alexa, google

def plot_monthly_reviews(alexa, google):
    """Plot monthly review volume timeline"""
    print("Generating monthly review volume plot...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Resample to monthly frequency
    alexa_monthly = alexa.set_index('at').resample('M').size()
    google_monthly = google.set_index('at').resample('M').size()
    
    ax.plot(alexa_monthly.index, alexa_monthly, 
            label='Amazon Alexa', lw=2, color=COLORS['alexa'])
    ax.plot(google_monthly.index, google_monthly, 
            label='Google Assistant', lw=2, color=COLORS['google'])
    
    # Formatting
    ax.set(title='Monthly Review Volume (2017-2025)',
           xlabel='Date', ylabel='Number of Reviews')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_review_volume.png", dpi=300)
    plt.close()
    print(f"✅ Saved monthly review volume plot to {OUTPUT_DIR}/monthly_review_volume.png")

def plot_rating_distribution(alexa, google):
    """Plot star rating distributions with percentages"""
    print("Generating rating distribution plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def _format_thousands(x, pos):
        return f'{int(x):,}'

    formatter = FuncFormatter(_format_thousands)
    
    def _plot_ratings(ax, data, title, color):
        counts = data['score'].value_counts().sort_index()
        percentages = 100 * counts / counts.sum()
        
        sns.barplot(x=counts.index, y=counts.values, ax=ax,
                    color=color, edgecolor='black')
        
        ax.set(title=title, xlabel='Star Rating', ylabel='Count')
        ax.set_yscale('log')  # Keep this if log scale is desired
        ax.yaxis.set_major_formatter(formatter)  # ✨ Apply readable format
        
        # Add percentage labels
        for idx, (score, pct) in enumerate(percentages.items()):
            ax.text(idx, counts[score], f'{pct:.1f}%', 
                    ha='center', va='bottom', fontsize=10)

    _plot_ratings(ax1, alexa, 'Amazon Alexa Rating Distribution', COLORS['alexa'])
    _plot_ratings(ax2, google, 'Google Assistant Rating Distribution', COLORS['google'])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rating_distribution.png", dpi=300)
    plt.close()
    print(f"✅ Saved rating distribution plot to {OUTPUT_DIR}/rating_distribution.png")
    
def generate_wordclouds(alexa, google):
    """Generate comparative word clouds"""
    print("Generating word clouds...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def _create_wordcloud(text, ax, title, color):
        cloud = WordCloud(width=800, height=400,
                          background_color='white',
                          max_words=150,
                          colormap=color + '_r',
                          stopwords={'app', 'assistant'}).generate(text)
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    alexa_text = ' '.join(alexa['clean_content'].dropna())
    google_text = ' '.join(google['clean_content'].dropna())
    
    _create_wordcloud(alexa_text, ax1, 'Amazon Alexa - Frequent Terms', 'Blues')
    _create_wordcloud(google_text, ax2, 'Google Assistant - Frequent Terms', 'Oranges')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/word_clouds.png", dpi=300)
    plt.close()
    print(f"✅ Saved word clouds to {OUTPUT_DIR}/word_clouds.png")

def yearly_sentiment_summary(alexa, google):
    """Generate yearly sentiment summary stats"""
    print("Calculating yearly sentiment statistics...")
    
    # Add year column to both dataframes
    for df in [alexa, google]:
        df['year'] = df['at'].dt.year
    
    # Calculate yearly statistics
    alexa_yearly = alexa.groupby('year')['score'].agg(['mean', 'count', 'std']).reset_index()
    google_yearly = google.groupby('year')['score'].agg(['mean', 'count', 'std']).reset_index()
    
    # Add assistant name for identification
    alexa_yearly['assistant'] = 'Alexa'
    google_yearly['assistant'] = 'Google Assistant'
    
    # Combine and save
    yearly_stats = pd.concat([alexa_yearly, google_yearly])
    yearly_stats.to_csv(f"{OUTPUT_DIR}/yearly_sentiment_stats.csv", index=False)
    print(f"✅ Saved yearly statistics to {OUTPUT_DIR}/yearly_sentiment_stats.csv")
    
    # Plot yearly mean scores
    plt.figure(figsize=(12, 6))
    plt.plot(alexa_yearly['year'], alexa_yearly['mean'], 
             'o-', color=COLORS['alexa'], label='Amazon Alexa')
    plt.plot(google_yearly['year'], google_yearly['mean'], 
             'o-', color=COLORS['google'], label='Google Assistant')
    
    plt.title('Yearly Average Rating (2017-2025)')
    plt.xlabel('Year')
    plt.ylabel('Average Star Rating')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/yearly_ratings.png", dpi=300)
    plt.close()
    print(f"✅ Saved yearly ratings plot to {OUTPUT_DIR}/yearly_ratings.png")

def main():
    """Run all EDA functions"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Load data
    alexa_df, google_df = load_data()
    
    # Generate all visualizations
    plot_monthly_reviews(alexa_df, google_df)
    plot_rating_distribution(alexa_df, google_df)
    generate_wordclouds(alexa_df, google_df)
    yearly_sentiment_summary(alexa_df, google_df)
    
    print("\n✅ All EDA visualizations complete!")

if __name__ == "__main__":
    main()