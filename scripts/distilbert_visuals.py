import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
THESIS_ROOT = BASE_DIR.parent

# Input files
input_files = {
    "alexa": THESIS_ROOT / "results" / "alexa_sentiment.csv",
    "google": THESIS_ROOT / "results" / "google_sentiment.csv"
}

# Output directory
output_visuals_dir = THESIS_ROOT / "results" / "sentiment_visuals"
output_visuals_dir.mkdir(parents=True, exist_ok=True)

# --- Enhanced Visualization Parameters ---
# Set up a professional theme for thesis
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 6
})

# Define consistent colors for platforms and sentiments
platform_colors = {'alexa': '#FF6B35', 'google': '#4285F4'}  # Brand-inspired colors
sentiment_colors = {'positive': '#2ECC71', 'neutral': '#F39C12', 'negative': '#E74C3C'}

# --- Processing and Enhanced Visualization ---
all_platform_data = []
print(f"Looking for sentiment results in: {THESIS_ROOT / 'results'}")
print(f"Saving visualizations to: {output_visuals_dir}")

for name, path in input_files.items():
    print(f"\n📈 Processing results for {name} from {path}...")

    try:
        df = pd.read_csv(path)

        # Data validation
        required_cols = ['at', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns {missing} in {path}. Skipping {name}.")
            continue

        # Convert 'at' column to datetime
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df.dropna(subset=['at'], inplace=True)

        if df.empty:
            print(f"Warning: No valid date entries found in {path}. Skipping {name}.")
            continue

        print(f"Loaded {len(df)} reviews with valid dates.")
        print(f"Date range: {df['at'].min()} to {df['at'].max()}")

        # --- Enhanced Individual Platform Analysis ---
        
        # 1. Quarterly Sentiment Distribution (Stacked Area Chart)
        quarterly_sentiment = df.groupby([df['at'].dt.to_period('Q'), 'sentiment']).size().unstack(fill_value=0)
        quarterly_sentiment_prop = quarterly_sentiment.div(quarterly_sentiment.sum(axis=1), axis=0)
        quarterly_sentiment_prop.index = quarterly_sentiment_prop.index.to_timestamp()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Stacked area plot for proportions
        ax1.stackplot(quarterly_sentiment_prop.index, 
                     quarterly_sentiment_prop['negative'], 
                     quarterly_sentiment_prop['neutral'], 
                     quarterly_sentiment_prop['positive'],
                     labels=['Negative', 'Neutral', 'Positive'],
                     colors=[sentiment_colors['negative'], sentiment_colors['neutral'], sentiment_colors['positive']],
                     alpha=0.8)
        
        ax1.set_ylabel('Proportion of Reviews')
        ax1.set_title(f'Quarterly Sentiment Distribution: {name.capitalize()}', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add trend lines for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in quarterly_sentiment_prop.columns:
                ax1.plot(quarterly_sentiment_prop.index, quarterly_sentiment_prop[sentiment], 
                        color=sentiment_colors[sentiment], linewidth=2, alpha=0.9)

        # 2. Absolute counts
        ax2.plot(quarterly_sentiment.index, quarterly_sentiment['positive'], 
                color=sentiment_colors['positive'], marker='o', label='Positive', linewidth=2.5)
        ax2.plot(quarterly_sentiment.index, quarterly_sentiment['negative'], 
                color=sentiment_colors['negative'], marker='s', label='Negative', linewidth=2.5)
        ax2.plot(quarterly_sentiment.index, quarterly_sentiment['neutral'], 
                color=sentiment_colors['neutral'], marker='^', label='Neutral', linewidth=2.5)
        
        ax2.set_ylabel('Number of Reviews')
        ax2.set_xlabel('Quarter')
        ax2.set_title(f'Quarterly Review Volume by Sentiment: {name.capitalize()}', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        # Improve date formatting
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save enhanced plot
        plot_filename = output_visuals_dir / f"{name}_enhanced_sentiment_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced sentiment analysis to {plot_filename}")
        plt.close()

        # --- Prepare data for comparison plots ---
        quarterly_sentiment_long = quarterly_sentiment_prop.reset_index().melt(
            id_vars=['at'], 
            value_vars=['negative', 'neutral', 'positive'],
            var_name='sentiment_category',
            value_name='proportion'
        )
        quarterly_sentiment_long['platform'] = name
        all_platform_data.append(quarterly_sentiment_long)

        # --- Monthly trend for more granular analysis ---
        monthly_sentiment = df.groupby([df['at'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
        monthly_sentiment_prop = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0)
        monthly_sentiment_prop.index = monthly_sentiment_prop.index.to_timestamp()

        plt.figure(figsize=(16, 8))
        
        # Create smooth trend lines using rolling average
        window_size = 3  # 3-month rolling average
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in monthly_sentiment_prop.columns:
                smoothed = monthly_sentiment_prop[sentiment].rolling(window=window_size, center=True).mean()
                plt.plot(smoothed.index, smoothed, 
                        color=sentiment_colors[sentiment], 
                        label=f'{sentiment.capitalize()} (3-month avg)',
                        linewidth=3, alpha=0.9)
                
                # Add the actual monthly data as lighter lines
                plt.plot(monthly_sentiment_prop.index, monthly_sentiment_prop[sentiment],
                        color=sentiment_colors[sentiment], 
                        alpha=0.3, linewidth=1)
        
        plt.ylabel('Proportion of Reviews')
        plt.xlabel('Month')
        plt.title(f'Monthly Sentiment Trends with Smoothing: {name.capitalize()}', fontweight='bold', pad=20)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Improve date formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save monthly trend plot
        monthly_plot_filename = output_visuals_dir / f"{name}_monthly_sentiment_trends.png"
        plt.savefig(monthly_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved monthly sentiment trends to {monthly_plot_filename}")
        plt.close()

    except FileNotFoundError:
        print(f"Error: Input file not found at {path}. Skipping {name}.")
    except Exception as e:
        print(f"Error processing {path}: {e}. Skipping {name}.")
        import traceback
        traceback.print_exc()

# --- Enhanced Comparison Plots ---
if len(all_platform_data) >= 1:
    combined_df = pd.concat(all_platform_data)
    
    # 1. Side-by-side sentiment distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    for i, platform in enumerate(['alexa', 'google']):
        if platform in [data['platform'].iloc[0] for data in all_platform_data]:
            platform_data = combined_df[combined_df['platform'] == platform]
            
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_data = platform_data[platform_data['sentiment_category'] == sentiment]
                axes[i].plot(sentiment_data['at'], sentiment_data['proportion'], 
                           color=sentiment_colors[sentiment], 
                           marker='o', label=sentiment.capitalize(), linewidth=2.5, markersize=4)
            
            axes[i].set_title(f'{platform.capitalize()} Sentiment Evolution', fontweight='bold', fontsize=14)
            axes[i].set_ylabel('Proportion of Reviews' if i == 0 else '')
            axes[i].set_xlabel('Quarter')
            axes[i].legend(frameon=True, fancybox=True, shadow=True)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
            
            # Format dates
            axes[i].xaxis.set_major_locator(mdates.YearLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Sentiment Evolution Comparison: Google Assistant vs Amazon Alexa', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    comparison_filename = output_visuals_dir / "platform_sentiment_comparison.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"Saved platform comparison to {comparison_filename}")
    plt.close()

    # 2. Overlay comparison for direct comparison
    plt.figure(figsize=(16, 10))
    
    # Create subplot for each sentiment category
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    sentiments = ['positive', 'negative', 'neutral']
    
    for i, sentiment in enumerate(sentiments):
        for platform in ['alexa', 'google']:
            if platform in combined_df['platform'].unique():
                platform_sentiment_data = combined_df[
                    (combined_df['platform'] == platform) & 
                    (combined_df['sentiment_category'] == sentiment)
                ]
                
                if not platform_sentiment_data.empty:
                    axes[i].plot(platform_sentiment_data['at'], platform_sentiment_data['proportion'],
                               color=platform_colors[platform], 
                               marker='o', label=f'{platform.capitalize()}', 
                               linewidth=3, markersize=5, alpha=0.8)
        
        axes[i].set_title(f'{sentiment.capitalize()} Sentiment Comparison', fontweight='bold', fontsize=14)
        axes[i].set_ylabel('Proportion of Reviews')
        axes[i].legend(frameon=True, fancybox=True, shadow=True)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, max(combined_df[combined_df['sentiment_category'] == sentiment]['proportion']) * 1.1)
    
    axes[2].set_xlabel('Quarter')
    
    # Format dates on bottom subplot
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('Detailed Sentiment Category Comparison: Google Assistant vs Amazon Alexa', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    detailed_comparison_filename = output_visuals_dir / "detailed_sentiment_comparison.png"
    plt.savefig(detailed_comparison_filename, dpi=300, bbox_inches='tight')
    print(f"Saved detailed comparison to {detailed_comparison_filename}")
    plt.close()

# --- Enhanced Average Sentiment Score Analysis ---
all_avg_sentiment_data = []

for name, path in input_files.items():
    try:
        df = pd.read_csv(path)
        
        if 'at' not in df.columns or 'sentiment_score' not in df.columns:
            print(f"Warning: Required columns missing in {path} for average score analysis.")
            continue

        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df.dropna(subset=['at'], inplace=True)

        if df.empty:
            continue

        # Calculate both quarterly and monthly averages
        quarterly_avg_score = df.groupby(df['at'].dt.to_period('Q'))['sentiment_score'].agg(['mean', 'std', 'count']).reset_index()
        quarterly_avg_score['at'] = quarterly_avg_score['at'].dt.to_timestamp()
        quarterly_avg_score['platform'] = name
        quarterly_avg_score['period_type'] = 'quarterly'
        
        monthly_avg_score = df.groupby(df['at'].dt.to_period('M'))['sentiment_score'].agg(['mean', 'std', 'count']).reset_index()
        monthly_avg_score['at'] = monthly_avg_score['at'].dt.to_timestamp()
        monthly_avg_score['platform'] = name
        monthly_avg_score['period_type'] = 'monthly'
        
        all_avg_sentiment_data.extend([quarterly_avg_score, monthly_avg_score])

    except Exception as e:
        print(f"Error calculating average score for {name}: {e}")

if all_avg_sentiment_data:
    combined_avg_df = pd.concat(all_avg_sentiment_data, ignore_index=True)
    
    # Plot quarterly averages with confidence intervals
    quarterly_data = combined_avg_df[combined_avg_df['period_type'] == 'quarterly']
    
    plt.figure(figsize=(16, 10))
    
    for platform in quarterly_data['platform'].unique():
        platform_data = quarterly_data[quarterly_data['platform'] == platform]
        
        # Calculate confidence intervals (assuming normal distribution)
        ci_lower = platform_data['mean'] - 1.96 * (platform_data['std'] / np.sqrt(platform_data['count']))
        ci_upper = platform_data['mean'] + 1.96 * (platform_data['std'] / np.sqrt(platform_data['count']))
        
        # Plot mean with confidence interval
        plt.plot(platform_data['at'], platform_data['mean'], 
                color=platform_colors[platform], 
                marker='o', label=f'{platform.capitalize()} (Mean)', 
                linewidth=3, markersize=6)
        
        plt.fill_between(platform_data['at'], ci_lower, ci_upper, 
                        color=platform_colors[platform], alpha=0.2, 
                        label=f'{platform.capitalize()} (95% CI)')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.xlabel('Quarter', fontsize=12)
    plt.title('Quarterly Average Sentiment Score with Confidence Intervals', 
             fontsize=16, fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    
    # Add horizontal lines for interpretation
    plt.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Positive threshold')
    plt.axhline(y=-0.1, color='red', linestyle=':', alpha=0.5, label='Negative threshold')
    
    # Format dates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    avg_score_filename = output_visuals_dir / "enhanced_average_sentiment_comparison.png"
    plt.savefig(avg_score_filename, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced average sentiment comparison to {avg_score_filename}")
    plt.close()

# --- Summary Statistics ---
summary_filename = output_visuals_dir / "sentiment_analysis_summary.txt"
with open(summary_filename, 'w') as f:
    f.write("SENTIMENT ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    for name, path in input_files.items():
        try:
            df = pd.read_csv(path)
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
            df.dropna(subset=['at'], inplace=True)
            
            f.write(f"{name.upper()} STATISTICS:\n")
            f.write(f"Total reviews: {len(df):,}\n")
            f.write(f"Date range: {df['at'].min().strftime('%Y-%m-%d')} to {df['at'].max().strftime('%Y-%m-%d')}\n")
            
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                f.write("Sentiment distribution:\n")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)\n")
            
            if 'sentiment_score' in df.columns:
                f.write(f"Average sentiment score: {df['sentiment_score'].mean():.3f}\n")
                f.write(f"Sentiment score std: {df['sentiment_score'].std():.3f}\n")
            
            f.write("\n" + "-" * 30 + "\n\n")
            
        except Exception as e:
            f.write(f"Error processing {name}: {e}\n\n")

print(f"\nSaved summary statistics to {summary_filename}")
print("\n--- Enhanced Visualization Script Complete ---")