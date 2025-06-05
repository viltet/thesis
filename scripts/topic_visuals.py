import pandas as pd
from bertopic import BERTopic # Still needed to load the model and use its methods
from pathlib import Path
import os
import traceback
import matplotlib.pyplot as plt # For static plots
import matplotlib.dates as mdates # For date formatting on static plots
import seaborn as sns # For styling static plots
import numpy as np
# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
THESIS_ROOT = BASE_DIR.parent

results_dir = THESIS_ROOT / "results"
topic_model_output_dir = results_dir / "topic_models" # Output from previous script
static_plots_output_dir = topic_model_output_dir / "topic_evolution_visuals" # New specific dir for these plots
static_plots_output_dir.mkdir(parents=True, exist_ok=True)

platforms = ["alexa", "google"]
num_topics_for_static_plot = 9 # You want to display top 9 in the static plot

# Updated Descriptive Topic Names Here 
alexa_topic_id_to_name_map = {
    0: "Echo Device & Connectivity",
    1: "Amazon Ecosystem & Account",
    2: "Voice Recognition & Commands",
    3: "Positive App Experience", # Simplified
    4: "Smart Home: Lights",
    5: "General Alexa App Feedback",
    6: "Software Updates & Issues",
    7: "Music Playback Control",
    8: "App Performance: Slowness" # Simplified
    # These are the top 9 based on counts from your full dataset run
}

google_topic_id_to_name_map = {
    0: "Gemini AI Transition", # Simplified
    1: "Music & Song Playback",
    2: "Device Lock/Unlock Issues", # Simplified
    3: "Positive Sentiment", # Simplified
    4: "Comparisons to Other VAs", # Simplified
    5: "Google Pixel Device Issues", # Simplified
    6: "App Ratings & Stars",
    7: "Microphone Issues", # Simplified
    8: "Updates & Google App Int." # Abbreviated for legend space
    # These are the top 9 based on counts from your full dataset run
}

# --- Enhanced Visualization Parameters ---
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except IOError:
    print("Warning: 'seaborn-v0_8-darkgrid' not found. Using 'ggplot'.")
    try:
        plt.style.use('ggplot')
    except IOError:
        print("Warning: 'ggplot' not found. Using default Matplotlib style.")

plot_title_fontsize = 20 # Adjusted for typical thesis figures
axis_label_fontsize = 16
legend_title_fontsize = 14
legend_text_fontsize = 12
tick_label_fontsize = 12
plot_linewidth = 2.5


def create_static_topic_evolution_plot(topics_over_time_df, topic_id_to_name_map, platform_name, num_topics_to_plot, output_path):
    """
    Generates and saves a static line chart for topic evolution with enhanced text visibility.
    """
    if topics_over_time_df.empty:
        print(f"No data in topics_over_time_df for {platform_name}. Skipping static plot.")
        return

    df_plot = topics_over_time_df[topics_over_time_df['Topic'] != -1].copy()
    df_plot['Topic_Name'] = df_plot['Topic'].map(topic_id_to_name_map)
    # Keep only topics that are in our map for plotting, others will be ignored
    df_plot = df_plot[df_plot['Topic_Name'].notna()]


    if df_plot.empty:
        print(f"No topics to plot for {platform_name} after mapping to provided names. Check your topic_id_to_name_map and topic IDs in topics_over_time_df.")
        return

    # Determine which of the named topics to display
    # If the map contains <= num_topics_to_plot, display all of them
    # Otherwise, pick the top N most frequent ones *that are in the map*
    available_mapped_topic_names = sorted(list(df_plot['Topic_Name'].unique()))

    if not available_mapped_topic_names:
        print(f"No topics with provided names found in temporal data for {platform_name}. Skipping plot.")
        return

    if len(available_mapped_topic_names) <= num_topics_to_plot:
        topics_to_display_names = available_mapped_topic_names
    else:
        # Calculate total frequency for mapped topics to select the top N
        total_freq_mapped_topics = df_plot.groupby('Topic_Name')['Frequency'].sum().sort_values(ascending=False)
        topics_to_display_names = total_freq_mapped_topics.head(num_topics_to_plot).index.tolist()

    df_plot_selected = df_plot[df_plot['Topic_Name'].isin(topics_to_display_names)]

    if df_plot_selected.empty:
        print(f"No data for selected top named topics for {platform_name}. Skipping static plot.")
        return
        
    plt.figure(figsize=(16, 9)) # Common 16:9 aspect ratio
    
    try:
        # Pivot: Timestamp on index, Topic_Name on columns, Frequency as values
        pivot_df = df_plot_selected.pivot_table(index='Timestamp', columns='Topic_Name', values='Frequency', fill_value=0)
    except ValueError: # Handles cases with duplicate (Timestamp, Topic_Name) from different original topic IDs mapping to same name
        print(f"Warning: Duplicate (Timestamp, Topic_Name) pairs for {platform_name}. Aggregating by sum for static plot.")
        pivot_df = df_plot_selected.groupby(['Timestamp', 'Topic_Name'])['Frequency'].sum().unstack(fill_value=0)

    # Ensure the columns in pivot_df are only those we intend to plot, and in the desired order
    columns_to_plot_in_pivot = [name for name in topics_to_display_names if name in pivot_df.columns]
    if not columns_to_plot_in_pivot:
        print(f"None of the selected topic names ({topics_to_display_names}) are in the pivot table columns for {platform_name}. Columns available: {pivot_df.columns.tolist()}. Skipping static plot.")
        return
    
    # Using a perceptually uniform colormap if many lines
    if len(columns_to_plot_in_pivot) > 7:
        colors = plt.cm.viridis(np.linspace(0, 1, len(columns_to_plot_in_pivot)))
        pivot_df[columns_to_plot_in_pivot].plot(ax=plt.gca(), linewidth=plot_linewidth, color=colors)
    else:
        pivot_df[columns_to_plot_in_pivot].plot(ax=plt.gca(), linewidth=plot_linewidth)


    plt.title(f'Evolution of Top {len(columns_to_plot_in_pivot)} Review Themes: {platform_name.capitalize()}', fontsize=plot_title_fontsize, fontweight='normal') # Normal weight for title
    plt.xlabel('Date (Quarterly Bins)', fontsize=axis_label_fontsize)
    plt.ylabel('Topic Frequency', fontsize=axis_label_fontsize)
    
    # Place legend outside the plot if many items
    if len(columns_to_plot_in_pivot) > 5:
        plt.legend(title='Topics', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., 
                   fontsize=legend_text_fontsize, title_fontsize=legend_title_fontsize)
        plt.tight_layout(rect=[0.03, 0.03, 0.78, 0.95]) # Adjust right margin for external legend
    else:
        plt.legend(title='Topics', loc='best', fontsize=legend_text_fontsize, title_fontsize=legend_title_fontsize)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95]) # Standard tight layout

    
    ax = plt.gca()
    # Ensure 'Timestamp' is datetime for proper date formatting
    if not pd.api.types.is_datetime64_any_dtype(pivot_df.index):
        try:
            pivot_df.index = pd.to_datetime(pivot_df.index)
        except Exception as e_dt:
            print(f"Warning: Could not convert pivot_df index to datetime for {platform_name}: {e_dt}")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3)) 
    plt.xticks(rotation=30, ha='right', fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', axis='x', linestyle=':', alpha=0.5)
    
    plt.savefig(output_path, dpi=300)
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
        # When loading a model where save_embedding_model=False,
        # you might need to pass the embedding model if BERTopic version requires it.
        # However, BERTopic often can load without it if embeddings were generic.
        # For safety, if it fails, one might need to pass the global embedding_model.
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
            
        # Ensure 'topic' column is integer for mapping
        df_sorted['topic'] = df_sorted['topic'].astype(int)

        docs_for_oot = df_sorted['clean_content'].tolist()
        timestamps_for_oot_raw = df_sorted['at'].tolist() # Already parsed as dates
        
        # Filter out NaT timestamps and corresponding docs before topics_over_time
        valid_indices = [i for i, ts in enumerate(timestamps_for_oot_raw) if pd.notna(ts)]
        valid_docs_for_oot_final = [docs_for_oot[i] for i in valid_indices]
        timestamps_for_oot_final = [timestamps_for_oot_raw[i] for i in valid_indices]

        if not timestamps_for_oot_final or len(valid_docs_for_oot_final) != len(timestamps_for_oot_final):
            print(f"Error: Insufficient valid timestamps or mismatch for {name} after loading and validation. Skipping temporal analysis.")
            continue

        print(f"Proceeding with temporal analysis for {name} using {len(valid_docs_for_oot_final)} documents with valid timestamps.")

        min_date = pd.Series(timestamps_for_oot_final).min()
        max_date = pd.Series(timestamps_for_oot_final).max()
        num_quarters = 0
        if pd.notna(min_date) and pd.notna(max_date) and max_date > min_date:
            # Calculate number of full quarters between min and max date
             num_quarters = (max_date.year - min_date.year) * 4 + (max_date.month - 1) // 3 - (min_date.month - 1) // 3 +1
        
        print(f"Calculated date range for {name}: {min_date} to {max_date}, estimated {num_quarters} quarterly bins.")

        topics_over_time_df = None
        if num_quarters > 1: # Need at least 2 bins for a meaningful evolution plot
            print(f"Generating topics over time with {num_quarters} bins (approximating quarters) for {name}.")
            try:
                topics_over_time_df = topic_model.topics_over_time(
                    valid_docs_for_oot_final,
                    timestamps_for_oot_final,
                    nr_bins=num_quarters, # Use calculated number of quarters
                    datetime_format='%Y-%m-%d %H:%M:%S' # Example, ensure it matches your 'at' column
                )
            except Exception as e_oot:
                print(f"Error generating topics_over_time with {num_quarters} bins: {e_oot}")
                traceback.print_exc()
                print("Attempting default BERTopic temporal binning...")
        
        if topics_over_time_df is None or topics_over_time_df.empty: # If previous failed or num_quarters too low
             print(f"Attempting default BERTopic temporal binning for {name} as quarterly binning failed or was not applicable.")
             try:
                topics_over_time_df = topic_model.topics_over_time(
                    valid_docs_for_oot_final,
                    timestamps_for_oot_final,
                    datetime_format='%Y-%m-%d %H:%M:%S'
                )
             except Exception as e_oot_default:
                print(f"Error generating topics_over_time with default binning: {e_oot_default}")
                traceback.print_exc()


        if topics_over_time_df is None or topics_over_time_df.empty:
            print(f"Warning: topics_over_time returned an empty DataFrame for {name}. No temporal data to save or visualize.")
        else:
            # Ensure Timestamp column is datetime
            topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp'])
            oot_csv_path = topic_model_output_dir / f"{name}_topics_over_time.csv"
            topics_over_time_df.to_csv(oot_csv_path, index=False) 
            print(f"Topics over time data SAVED to {oot_csv_path}")

            topic_counts = topic_model.get_topic_info()
            valid_model_topics = topic_counts[topic_counts['Topic'] != -1]['Topic'].tolist()
            top_n_viz_html = 15 
            
            viz_topics_overall_html = []
            if valid_model_topics:
                viz_topics_overall_html = topic_counts[topic_counts['Topic'].isin(valid_model_topics)].sort_values('Count', ascending=False)['Topic'].head(top_n_viz_html).tolist()
            
            topics_in_oot_df = topics_over_time_df['Topic'].unique()
            final_topics_for_oot_viz_html = [t for t in viz_topics_overall_html if t in topics_in_oot_df and t != -1]

            if not final_topics_for_oot_viz_html and any(t != -1 for t in topics_in_oot_df):
                non_outlier_oot_topics = sorted([t for t in topics_in_oot_df if t != -1]) # Sort for consistency
                if non_outlier_oot_topics:
                    # Select based on overall frequency in topics_over_time_df for relevant topics
                    oot_topic_counts_in_df = topics_over_time_df[topics_over_time_df['Topic'].isin(non_outlier_oot_topics)]\
                                             .groupby('Topic')['Frequency'].sum().reset_index()
                    final_topics_for_oot_viz_html = oot_topic_counts_in_df.sort_values('Frequency', ascending=False)['Topic'].head(min(top_n_viz_html, len(non_outlier_oot_topics))).tolist()


            if final_topics_for_oot_viz_html:
                print(f"Visualizing HTML evolution for topics: {final_topics_for_oot_viz_html} for {name}")
                try:
                    fig_html = topic_model.visualize_topics_over_time(topics_over_time_df, topics=final_topics_for_oot_viz_html)
                    oot_html_path = topic_model_output_dir / f"{name}_topic_evolution.html"
                    fig_html.write_html(str(oot_html_path)) 
                    print(f"HTML topic evolution plot SAVED to {oot_html_path}")
                except Exception as e_vis_html:
                    print(f"Error (re)visualizing HTML topics over time for {name}: {e_vis_html}")
                    traceback.print_exc()
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
        traceback.print_exc() # Added traceback for FileNotFoundError
    except Exception as e:
        print(f"An unexpected error occurred while processing {name}: {e}")
        traceback.print_exc()

print("\n🖼️ Temporal Topic Analysis & Static Plot Script Finished 🔄")