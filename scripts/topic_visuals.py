import pandas as pd
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import os
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
THESIS_ROOT = BASE_DIR.parent

results_dir = THESIS_ROOT / "results"
topic_model_output_dir = results_dir / "topic_models"
static_plots_output_dir = topic_model_output_dir / "topic_evolution_visuals"
static_plots_output_dir.mkdir(parents=True, exist_ok=True)

platforms = ["alexa", "google"]
num_topics_for_static_plot = 9

# --- Descriptive Topic Names ---
alexa_topic_id_to_name_map = {
    0: "Echo Device & Connectivity",
    1: "Amazon Ecosystem & Account",
    2: "Voice Recognition & Commands",
    3: "Positive App Experience",
    4: "Smart Home: Lights",
    5: "General Alexa App Feedback",
    6: "Software Updates & Issues",
    7: "Music Playback Control",
    8: "App Performance: Slowness"
}

google_topic_id_to_name_map = {
    0: "Gemini AI Transition",
    1: "Music & Song Playback",
    2: "Device Lock/Unlock Issues",
    3: "Positive Sentiment",
    4: "Comparisons to Other VAs",
    5: "Google Pixel Device Issues",
    6: "App Ratings & Stars",
    7: "Microphone Issues",
    8: "Updates & Google App Int."
}

# --- !!! ADJUSTED: Enhanced Visualization Parameters !!! ---
try:
    # This style has a good default color cycle (tab10) for distinct lines
    plt.style.use('seaborn-v0_8-darkgrid')
except IOError:
    print("Warning: 'seaborn-v0_8-darkgrid' not found. Using 'ggplot'.")
    plt.style.use('ggplot')

# Further increased font sizes for maximum readability
plot_title_fontsize = 24
axis_label_fontsize = 21
legend_title_fontsize = 20
legend_text_fontsize = 18
tick_label_fontsize = 17
plot_linewidth = 3.0


def create_static_topic_evolution_plot(topics_over_time_df, topic_id_to_name_map, platform_name, num_topics_to_plot, output_path):
    """
    Generates and saves a static line chart for topic evolution with larger fonts and distinct colors.
    """
    if topics_over_time_df.empty:
        print(f"No data in topics_over_time_df for {platform_name}. Skipping static plot.")
        return

    df_plot = topics_over_time_df[topics_over_time_df['Topic'] != -1].copy()
    df_plot['Topic_Name'] = df_plot['Topic'].map(topic_id_to_name_map)
    df_plot = df_plot.dropna(subset=['Topic_Name'])

    if df_plot.empty:
        print(f"No topics to plot for {platform_name} after mapping names. Check your topic_id_to_name_map.")
        return

    # Determine which of the named topics to display based on overall frequency
    total_freq_mapped_topics = df_plot.groupby('Topic_Name')['Frequency'].sum().sort_values(ascending=False)
    topics_to_display_names = total_freq_mapped_topics.head(num_topics_to_plot).index.tolist()

    df_plot_selected = df_plot[df_plot['Topic_Name'].isin(topics_to_display_names)]

    if df_plot_selected.empty:
        print(f"No data for selected top named topics for {platform_name}. Skipping static plot.")
        return
        
    plt.figure(figsize=(18, 11))
    
    pivot_df = df_plot_selected.pivot_table(index='Timestamp', columns='Topic_Name', values='Frequency', fill_value=0)

    columns_to_plot_in_pivot = [name for name in topics_to_display_names if name in pivot_df.columns]
    if not columns_to_plot_in_pivot:
        print(f"None of the selected topic names are in the pivot table columns for {platform_name}. Skipping static plot.")
        return
    
    # Let the default style's color cycle ('tab10') handle colors for better distinction
    pivot_df[columns_to_plot_in_pivot].plot(ax=plt.gca(), linewidth=plot_linewidth)

    plt.title(f'Evolution of Top {len(columns_to_plot_in_pivot)} Review Themes: {platform_name.capitalize()}', fontsize=plot_title_fontsize, fontweight='bold')
    plt.xlabel('Date (Quarterly Bins)', fontsize=axis_label_fontsize, fontweight='bold')
    plt.ylabel('Topic Frequency', fontsize=axis_label_fontsize, fontweight='bold')
    
    plt.legend(title='Topics', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., 
               fontsize=legend_text_fontsize, title_fontsize=legend_title_fontsize)
    
    ax = plt.gca()
    try:
        pivot_df.index = pd.to_datetime(pivot_df.index)
    except Exception as e_dt:
        print(f"Warning: Could not convert pivot_df index to datetime for {platform_name}: {e_dt}")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha='right', fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', axis='x', linestyle=':', alpha=0.4)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.78, 0.95]) # Adjust margin for external legend
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved static topic evolution plot to {output_path}")
    plt.close()


# --- Main script execution loop ---
# (The main loop remains the same as the previous version)
for name in platforms:
    print(f"\n📊 Generating temporal topic analysis for {name}...")

    try:
        model_load_path = topic_model_output_dir / f"{name}_bertopic_model"
        if not model_load_path.exists():
            print(f"Error: Saved BERTopic model not found at {model_load_path}. Skipping {name}.")
            continue
        print(f"Loading saved BERTopic model from {model_load_path}...")
        topic_model = BERTopic.load(str(model_load_path))
        print(f"Model for {name} loaded successfully.")

        df_with_topics_path = results_dir / f"{name}_with_topics.csv"
        if not df_with_topics_path.exists():
            print(f"Error: Dataframe with topics '{df_with_topics_path.name}' not found in {results_dir}. Skipping {name}.")
            continue
        
        print(f"Loading dataframe with topics from {df_with_topics_path}...")
        df_sorted = pd.read_csv(df_with_topics_path, parse_dates=['at'])
        df_sorted['clean_content'] = df_sorted['clean_content'].astype(str)
        df_sorted['topic'] = df_sorted['topic'].astype(int)
        
        valid_indices = df_sorted['at'].notna()
        valid_docs_for_oot_final = df_sorted.loc[valid_indices, 'clean_content'].tolist()
        timestamps_for_oot_final = df_sorted.loc[valid_indices, 'at'].tolist()

        if not timestamps_for_oot_final:
            print(f"Error: Insufficient valid timestamps for {name}. Skipping temporal analysis.")
            continue

        print(f"Proceeding with temporal analysis for {name} using {len(valid_docs_for_oot_final)} documents with valid timestamps.")

        min_date = pd.Series(timestamps_for_oot_final).min()
        max_date = pd.Series(timestamps_for_oot_final).max()
        
        num_quarters = (max_date.year - min_date.year) * 4 + (max_date.month - 1) // 3 - (min_date.month - 1) // 3 + 1 if pd.notna(min_date) and pd.notna(max_date) else 0

        print(f"Calculated date range for {name}: {min_date} to {max_date}, estimated {num_quarters} quarterly bins.")

        topics_over_time_df = topic_model.topics_over_time(
            valid_docs_for_oot_final,
            timestamps_for_oot_final,
            nr_bins=num_quarters if num_quarters > 1 else None
        )

        if topics_over_time_df is None or topics_over_time_df.empty:
            print(f"Warning: topics_over_time returned an empty DataFrame for {name}. No temporal data to save or visualize.")
        else:
            topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp'])
            oot_csv_path = topic_model_output_dir / f"{name}_topics_over_time.csv"
            topics_over_time_df.to_csv(oot_csv_path, index=False)
            print(f"Topics over time data SAVED to {oot_csv_path}")

            # --- Call the function to generate static plot ---
            static_plot_file_path = static_plots_output_dir / f"{name}_static_topic_evolution_top{num_topics_for_static_plot}.png"
            current_topic_map = alexa_topic_id_to_name_map if name == "alexa" else google_topic_id_to_name_map
            
            create_static_topic_evolution_plot(
                topics_over_time_df,
                current_topic_map,
                name,
                num_topics_for_static_plot,
                static_plot_file_path
            )

    except FileNotFoundError as e_fnf:
        print(f"FileNotFoundError for {name}: {e_fnf}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred while processing {name}: {e}")
        traceback.print_exc()

print("\n🖼️ Temporal Topic Analysis & Static Plot Script Finished 🔄")