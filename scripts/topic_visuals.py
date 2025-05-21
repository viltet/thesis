import pandas as pd
from bertopic import BERTopic # Still needed to load the model and use its methods
from pathlib import Path
import os
import traceback
import matplotlib.pyplot as plt # For static plots
import matplotlib.dates as mdates # For date formatting on static plots
import seaborn as sns # For styling static plots

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
THESIS_ROOT = BASE_DIR.parent

results_dir = THESIS_ROOT / "results"
topic_model_output_dir = results_dir / "topic_models"
static_plots_output_dir = topic_model_output_dir / "static_topic_evolution_plots"
static_plots_output_dir.mkdir(parents=True, exist_ok=True)

platforms = ["alexa", "google"]
num_topics_for_static_plot = 9

# --- !!! IMPORTANT: Define Your Descriptive Topic Names Here !!! ---
alexa_topic_id_to_name_map = {
    0: "Ecosystem & App",
    1: "Voice Recognition",
    2: "Music Playback Cmds",
    3: "Shopping & Lists",
    4: "Smart Home: Lights",
    6: "App Performance (Lag)",
    7: "Amazon Music Service",
    9: "Software Updates",
    13: "Alarms & Timers"
    # Ensure you have about 9 topics you want to plot for Alexa
}

google_topic_id_to_name_map = {
    0: "Gemini AI Transition",
    1: "Music Playback",
    2: "Security/Privacy",
    3: "Device Lock/Unlock",
    4: "Language Support",
    5: "Siri/Apple Comparison",
    6: "Voice Retraining",
    7: "Pixel Device Issues",
    9: "Assistant Response"
    # Ensure you have about 9 topics you want to plot for Google
}

# --- Enhanced Visualization Parameters ---
# Matplotlib style for static plots
try:
    plt.style.use('seaborn-v0_8-darkgrid') # A good default style
except IOError:
    print("Warning: 'seaborn-v0_8-darkgrid' not found. Using 'ggplot'.")
    try:
        plt.style.use('ggplot')
    except IOError:
        print("Warning: 'ggplot' not found. Using default Matplotlib style.")

# Further Increased font sizes for better visibility
plot_title_fontsize = 22   
axis_label_fontsize = 19     
legend_title_fontsize = 18    
legend_text_fontsize = 17    
tick_label_fontsize = 15      
plot_linewidth = 2.8


def create_static_topic_evolution_plot(topics_over_time_df, topic_id_to_name_map, platform_name, num_topics_to_plot, output_path):
    """
    Generates and saves a static line chart for topic evolution with further enhanced text visibility.
    """
    if topics_over_time_df.empty:
        print(f"No data in topics_over_time_df for {platform_name}. Skipping static plot.")
        return

    df_plot = topics_over_time_df[topics_over_time_df['Topic'] != -1].copy()
    df_plot['Topic_Name'] = df_plot['Topic'].map(topic_id_to_name_map)
    df_plot = df_plot[df_plot['Topic_Name'].notna()]

    if df_plot.empty:
        print(f"No topics to plot for {platform_name} after mapping names. Check your topic_id_to_name_map.")
        return

    available_named_topics = df_plot['Topic_Name'].unique()
    if len(available_named_topics) <= num_topics_to_plot:
        topics_to_display_names = available_named_topics
    else:
        total_freq_named_topics = df_plot.groupby('Topic_Name')['Frequency'].sum().sort_values(ascending=False)
        topics_to_display_names = total_freq_named_topics.head(num_topics_to_plot).index.tolist()

    df_plot_selected = df_plot[df_plot['Topic_Name'].isin(topics_to_display_names)]

    if df_plot_selected.empty:
        print(f"No data for selected top named topics for {platform_name}. Skipping static plot.")
        return
        
    plt.figure(figsize=(18, 11)) # Slightly adjusted figure size
    
    try:
        pivot_df = df_plot_selected.pivot(index='Timestamp', columns='Topic_Name', values='Frequency').fillna(0)
    except ValueError:
        print(f"Warning: Duplicate (Timestamp, Topic_Name) pairs found for {platform_name}. Aggregating by sum for static plot.")
        pivot_df = df_plot_selected.groupby(['Timestamp', 'Topic_Name'])['Frequency'].sum().unstack(fill_value=0)

    if pivot_df.empty or not any(col in pivot_df.columns for col in topics_to_display_names):
        print(f"Pivot table is empty or does not contain selected topic names for {platform_name}. Skipping static plot.")
        return

    columns_in_pivot = [name for name in topics_to_display_names if name in pivot_df.columns]
    if not columns_in_pivot:
        print(f"None of the selected topic names are in the pivot table columns for {platform_name}. Skipping static plot.")
        return
    

    pivot_df[columns_in_pivot].plot(ax=plt.gca(), linewidth=plot_linewidth) # Removed palette to use style's default

    plt.title(f'Evolution of Top {len(columns_in_pivot)} Review Themes: {platform_name.capitalize()}', fontsize=plot_title_fontsize, fontweight='bold')
    plt.xlabel('Date (Quarterly Bins)', fontsize=axis_label_fontsize, fontweight='bold')
    plt.ylabel('Topic Frequency', fontsize=axis_label_fontsize, fontweight='bold')
    
    plt.legend(title='Topics', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., 
               fontsize=legend_text_fontsize, title_fontsize=legend_title_fontsize)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3)) # Quarterly minor ticks
    plt.xticks(rotation=30, ha='right', fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', axis='x', linestyle=':', alpha=0.4)
    
    # Adjust layout: rect=[left, bottom, right, top] to ensure legend fits
    plt.tight_layout(rect=[0.03, 0.03, 0.80, 0.95]) # Give more space on right for legend, and bit on top for title
    
    plt.savefig(output_path, dpi=300) # Saving with good resolution
    print(f"Saved static topic evolution plot to {output_path}")
    plt.close()


# --- Main script execution loop ---
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
        
        required_cols_loaded_df = ['at', 'clean_content', 'topic']
        if not all(col in df_sorted.columns for col in required_cols_loaded_df):
            missing_cols = [col for col in required_cols_loaded_df if col not in df_sorted.columns]
            print(f"Error: Missing one or more required columns {missing_cols} in {df_with_topics_path.name}. Skipping {name}.")
            continue
            
        docs_for_oot = df_sorted['clean_content'].tolist()
        timestamps_for_oot_raw = df_sorted['at'].tolist()
        timestamps_for_oot = []
        valid_docs_for_oot_final = []
        
        print(f"Validating {len(timestamps_for_oot_raw)} timestamps for {name}...")
        for i, ts_val in enumerate(timestamps_for_oot_raw):
            if pd.notna(ts_val):
                try:
                    timestamps_for_oot.append(pd.to_datetime(ts_val))
                    valid_docs_for_oot_final.append(docs_for_oot[i])
                except Exception:
                    pass
            
        if not timestamps_for_oot or len(valid_docs_for_oot_final) != len(timestamps_for_oot):
            print(f"Error: Insufficient valid timestamps or mismatch for {name} after loading and validation. Processed {len(timestamps_for_oot)} timestamps for {len(valid_docs_for_oot_final)} docs. Skipping temporal analysis.")
            continue

        print(f"Proceeding with temporal analysis for {name} using {len(valid_docs_for_oot_final)} documents with valid timestamps.")

        min_date = pd.Series(timestamps_for_oot).min()
        max_date = pd.Series(timestamps_for_oot).max()
        num_quarters = 0
        if pd.notna(min_date) and pd.notna(max_date) and max_date > min_date:
            num_quarters = len(pd.period_range(start=min_date, end=max_date, freq='Q'))

        topics_over_time_df = None
        if num_quarters > 0:
            print(f"Attempting temporal analysis with {num_quarters} bins (approximating quarters) for {name}.")
            topics_over_time_df = topic_model.topics_over_time(
                valid_docs_for_oot_final,
                timestamps_for_oot,
                nr_bins=num_quarters
            )
        else:
            print(f"Warning: Date range too short for quarterly bins ({num_quarters} quarters found) for {name}. Attempting default BERTopic temporal binning.")
            topics_over_time_df = topic_model.topics_over_time(
                valid_docs_for_oot_final,
                timestamps_for_oot
            )

        if topics_over_time_df is None or topics_over_time_df.empty:
            print(f"Warning: topics_over_time returned an empty DataFrame for {name}. No temporal data to save or visualize.")
        else:
            oot_csv_path = topic_model_output_dir / f"{name}_topics_over_time.csv"
            # topics_over_time_df.to_csv(oot_csv_path, index=False) # Assumed already saved
            # print(f"Topics over time data already available at {oot_csv_path}")

            topic_counts = topic_model.get_topic_info()
            valid_model_topics = topic_counts[topic_counts['Topic'] != -1]['Topic'].tolist()
            top_n_viz_html = 15 
            
            viz_topics_overall_html = []
            if valid_model_topics:
                viz_topics_overall_html = topic_counts[topic_counts['Topic'].isin(valid_model_topics)].sort_values('Count', ascending=False)['Topic'].head(top_n_viz_html).tolist()
                if not viz_topics_overall_html:
                     viz_topics_overall_html = valid_model_topics[:min(top_n_viz_html, len(valid_model_topics))]
            
            topics_in_oot_df = topics_over_time_df['Topic'].unique()
            final_topics_for_oot_viz_html = [t for t in viz_topics_overall_html if t in topics_in_oot_df and t != -1]

            if not final_topics_for_oot_viz_html and any(t != -1 for t in topics_in_oot_df):
                non_outlier_oot_topics = [t for t in topics_in_oot_df if t != -1]
                if non_outlier_oot_topics:
                    oot_topic_counts_in_df = topics_over_time_df[topics_over_time_df['Topic'].isin(non_outlier_oot_topics)]\
                                             .groupby('Topic')['Frequency'].sum().reset_index()
                    final_topics_for_oot_viz_html = oot_topic_counts_in_df.sort_values('Frequency', ascending=False)['Topic'].head(10).tolist()

            if final_topics_for_oot_viz_html:
                print(f"Visualizing HTML evolution for topics: {final_topics_for_oot_viz_html} for {name}")
                try:
                    fig_html = topic_model.visualize_topics_over_time(topics_over_time_df, topics=final_topics_for_oot_viz_html)
                    oot_html_path = topic_model_output_dir / f"{name}_topic_evolution.html"
                    # fig_html.write_html(str(oot_html_path)) # Assumed already saved
                    # print(f"HTML topic evolution plot already available at {oot_html_path}")
                except Exception as e_vis_html:
                    print(f"Error (re)visualizing HTML topics over time for {name}: {e_vis_html}")
            else:
                print(f"No suitable topics found for HTML temporal evolution visualization for {name}.")

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
    except Exception as e:
        print(f"An unexpected error occurred while processing {name}: {e}")
        traceback.print_exc()

print("\n🖼️ Temporal Topic Analysis & Static Plot Script Finished 🔄")