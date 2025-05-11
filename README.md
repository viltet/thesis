# Evolution of Customer Sentiment Toward AI-Driven Virtual Assistants: A Longitudinal Analysis of Google Assistant and Alexa Reviews

## Overview

This repository contains the code and data analysis artifacts for a Master's thesis investigating the evolution of customer sentiment towards AI-driven virtual assistants, specifically Google Assistant and Amazon Alexa. The study employs a comprehensive longitudinal approach using a large dataset of customer reviews from the Google Play Store between 2017 and 2025.

The project utilizes a multi-stage Natural Language Processing (NLP) pipeline to track and analyze sentiment trends, identify key themes, analyze sentiment linked to specific product features, and estimate the causal impact of product updates and external events on user perception.

## Research Questions

This study aims to address the following key research questions:

1.  How has customer sentiment toward AI-driven virtual assistants evolved over the past decade (2017-2025)?
2.  What key themes and concerns emerge from customer reviews, and how have they shifted over time?
3.  How do specific product updates and external events influence sentiment trends?

## Methodology

The research follows an integrated analytical pipeline:

1.  **Data Collection & Preprocessing:** Automated scraping of user reviews from the Google Play Store for Google Assistant and Amazon Alexa. Rigorous cleaning and preprocessing of raw review text.
2.  **Sentiment Trend Extraction (RQ1):**
    * Transformer-based sentiment classification (DistilBERT) applied to classify reviews into negative, neutral, and positive categories.
    * Aggregation of sentiment scores over time (quarterly).
    * Visualization of longitudinal sentiment trends for each platform and comparison between them.
    * (Planned for later: Change point detection to identify major trend shifts).
3.  **Topic Modeling & Taxonomy Mapping (RQ2):**
    * Dynamic topic modeling (BERTopic) to identify evolving themes and discussion points within reviews.
    * Mapping identified topics to theoretical taxonomy aspects (e.g., functionality, privacy, usability, etc.).
    * Tracking topic prevalence and emergence over time.
4.  **Aspect-Based Sentiment Analysis (RQ2 Extension):**
    * Integration of topic modeling outputs with sentiment scores.
    * Calculation and tracking of sentiment specifically related to identified aspects/themes.
    * Comparison of aspect importance and sentiment across platforms.
5.  **Causal Impact Analysis (RQ3):**
    * Identification of key events (product updates, controversies, etc.).
    * Application of Bayesian Structural Time Series (BSTS) modeling to estimate the causal effect of these events on overall and aspect-specific sentiment.
    * Quantification of impact magnitude and pattern comparison between platforms.
6.  **Theoretical Integration:**
    * Mapping empirical findings back to theoretical frameworks (e.g., Expectation Confirmation Theory, Privacy Calculus, Attachment Theory).

## 📁 Repository Structure
data/
Processed input review data (CSV files for Alexa and Google Assistant)

results/
All analysis outputs and visualizations


scripts/
Python scripts for the full NLP and analysis pipeline

requirements.txt
Python dependencies for reproducibility

README.md
Project overview, methodology, and setup instructions
