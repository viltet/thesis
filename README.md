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

##  Repository Structure
data/
Processed input review data (CSV files for Alexa and Google Assistant)

results/
All analysis outputs and visualizations

scripts/
Python scripts for the full NLP and analysis pipeline

requirements.txt
Python dependencies for reproducibility

README.md
Project overview, methodology, and instructions
