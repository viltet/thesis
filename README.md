# Evolution of Customer Sentiment Toward AI-Driven Virtual Assistants: A Longitudinal Analysis of Google Assistant and Alexa Reviews

## Overview

This repository contains the code and data analysis artifacts for a Master's thesis investigating the evolution of customer sentiment towards AI-driven virtual assistants, specifically Google Assistant and Amazon Alexa. The study employs a comprehensive longitudinal approach using a large dataset of customer reviews from the Google Play Store spanning from October 2017 to March 2025.

The project utilizes a multi-stage Natural Language Processing (NLP) pipeline to:
* Track overall sentiment trends.
* Identify evolving user-discussed themes.
* Analyze sentiment towards predefined product aspects.
* Estimate the causal impact of product updates and key events on user sentiment.

## Research Questions

This study aims to address the following key research questions:

1.  How has customer sentiment toward AI-driven virtual assistants (Amazon Alexa and Google Assistant) evolved from late 2017 to early 2025?
2.  What key themes and concerns emerge from customer reviews for each assistant, and how have these themes and their prominence shifted over time? How does sentiment vary across different predefined aspects of the virtual assistants?
3.  How do specific product updates and external events influence overall and aspect-specific sentiment trends for each virtual assistant?

## Methodology

The research follows an integrated analytical pipeline:

### 1. Data Collection
Automated scraping of user reviews from the Google Play Store for Google Assistant and Amazon Alexa (October 2017 – March 2025).

### 2. Data Preprocessing
Rigorous cleaning of raw review text, including removal of irrelevant characters, handling of emojis, and text normalization.

### 3. Exploratory Data Analysis (EDA)
Initial analysis of review volume, rating distributions, and text characteristics to understand the dataset.

### 4. Analytical Framework

#### 4.1 Sentiment Analysis (for RQ1)
* **Overall Sentiment Classification:** Fine-tuning a DistilBERT model to classify reviews into positive, negative, or neutral categories and to assign a continuous sentiment score.
* **Trend Aggregation:** Aggregating sentiment scores (mean sentiment, proportions of positive/negative/neutral) on a weekly and quarterly basis to analyze longitudinal trends.

#### 4.2 Topic Modeling (for RQ2, part 1)
* **Dynamic Theme Identification:** Applying BERTopic, a dynamic topic modeling technique, to identify coherent themes and track their evolution and prominence over time within the review corpus for each assistant.

#### 4.3 Aspect-Based Sentiment Analysis (ABSA) (for RQ2, part 2)
* **Aspect Category Definition:** Establishing a predefined taxonomy of key product aspects (e.g., "Functionality & Performance," "Voice Recognition," "Privacy & Security").
* **Aspect Sentiment Classification:** Fine-tuning a DeBERTa model for ABSA to identify mentions of these aspects within review sentences and classify the sentiment expressed towards each specific aspect.
* **Aspect Sentiment Aggregation:** Aggregating aspect-specific sentiment scores weekly and quarterly to track their evolution.

#### 4.4 Causal Impact Analysis (for RQ3)
* **Event Identification:** Defining key product updates, feature launches, and external events for both Alexa and Google Assistant.
* **Causal Inference:** Employing Bayesian Structural Time Series (BSTS) models to estimate the causal effect of these identified events on:
    * Overall weekly sentiment scores.
    * Weekly sentiment scores for relevant, predefined aspects.
    * Covariates (e.g., competing assistant's sentiment) are used to control for market-wide trends.

## Repository Structure

**data/**
* `alexa_processed.csv`: Processed review data for Amazon Alexa.
* `google_processed.csv`: Processed review data for Google Assistant.
* `alexa_reviews_all.csv`: Raw scraped review data for Amazon Alexa.
* `google_assistant_reviews_all.csv`: Raw scraped review data for Google Assistant.
* `pilot_subset.csv`: Small, labeled subset of reviews used for initial model testing and threshold calibration.

**results/**
* Contains all outputs from the analysis pipeline, including:
    * Sentiment scores and classifications.
    * Topic modeling results (identified topics, keywords, evolution data).
    * Aspect-Based Sentiment Analysis (ABSA) outputs (aspect sentiment scores).
    * Causal impact analysis (BSTS model summaries and plots).
    * Visualizations (graphs, charts, word clouds) for EDA, sentiment trends, topic evolution, and ABSA.
    * Outputs from pilot testing and model comparison experiments.
    * Specific subdirectories like `absa_pilot/`, `absa_results/`, `absa_visuals_local_all_aspects/`, `bsts_outputs/` (with `absa_sentiment/` and `overall_sentiment/` subfolders), `model_comparison/`, `sentiment_visuals/`, `topic_model_comparison/`, `topic_models/` indicate a well-organized output structure.

**scripts/**
* Contains all Python scripts and Jupyter notebooks used for data collection, preprocessing, analysis, and visualization.

    * **Data Collection & Preparation:**
        * `scraping.py`: Fetches reviews for Amazon Alexa and Google Assistant from the Google Play Store.
        * `preprocessing.py`: Cleans and preprocesses the raw review data.
        * `eda.py`: Performs exploratory data analysis on the processed datasets.
        * `pilot_dataset.py`: Creates a smaller, stratified pilot subset from the processed data for efficient initial model evaluation and testing.
        * `create_pilot_data_asba.py`: Prepares a targeted pilot dataset specifically for Aspect-Based Sentiment Analysis.

    * **Model Testing & Evaluation (Pilot Phase):**
        * `testing_sent_methods.py`: Compares various sentiment analysis models (VADER, DistilBERT, BERT, RoBERTa) on the pilot dataset to select the best performer for the main analysis. Includes threshold calibration.
        * `testing_topic_methods.py`: Evaluates and compares different topic modeling approaches (LDA and BERTopic) to determine the most suitable method for theme identification.
        * `asba_pilot_evaluation.py`: Evaluates different models (VADER, DistilBERT-SST2, DeBERTa ABSA, Zero-Shot NLI) for Aspect-Based Sentiment Analysis on a manually annotated pilot dataset to select the optimal ABSA model.

    * **Main Sentiment Analysis (RQ1):**
        * `distilbert.py`: Applies the chosen DistilBERT model to perform sentiment classification (positive, negative, neutral) on the full preprocessed datasets.
        * `distilbert_visuals.py`: Generates visualizations from the DistilBERT sentiment analysis results, such as quarterly and monthly sentiment distributions and trends.

    * **Topic Modeling & Thematic Analysis (RQ2):**
        * `bertopic_modeling.py`: Implements BERTopic for dynamic topic modeling to identify and characterize latent themes in the review corpus.
        * `topic_visuals.py`: Creates visualizations of the topic modeling results, focusing on the evolution of topic prominence over time (static plots).

    * **Aspect-Based Sentiment Analysis (ABSA) (RQ2):**
        * `asba.ipynb`: (Jupyter Notebook) Contains the main pipeline for full-scale Aspect-Based Sentiment Analysis using the selected DeBERTa model and a predefined aspect taxonomy.
        * `asba_visuals.py`: Generates visualizations for the ABSA results, such as the evolution of sentiment for specific product aspects.

    * **Causal Impact Analysis (RQ3):**
        * `bsts_overall_sentiment.py`: Implements Bayesian Structural Time Series (BSTS) models to estimate the causal impact of product updates and external events on *overall* sentiment trends.
        * `bsts_asba.py`: Applies BSTS models to *aspect-specific* sentiment time series (derived from ABSA) to analyze the causal impact of events on sentiment towards particular product features.
