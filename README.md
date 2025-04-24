# spotify_recommendation_syaytem_codeAlpha_task
This project implements a hybrid music recommendation system that combines three different approaches:

Content-based filtering (using song tags and audio features)

Collaborative filtering (using user listening history)

Predictive modeling (XGBoost classifier to predict repeat listens)

The system recommends songs based on a given user and a seed song, combining insights from all three approaches.

Code Explanation
Data Preparation
Data Loading:

Loads music metadata (Music Info.csv) and user listening history (User Listening History.csv)

Merges these datasets on track_id

Sampling:

Takes a random sample of 5,000 records for faster experimentation

Selects relevant features including audio characteristics, tags, and play counts

Feature Engineering:

Creates a binary target repeated_within_month (1 if playcount ≥ 2)

Normalizes tags to lowercase

Adds interaction features like energy_valence and acoustic_speech

Recommendation Components
1. Content-Based Filtering (Tags)
Uses TF-IDF vectorization on song tags

Computes cosine similarity between songs based on tag similarity

2. Collaborative Filtering
Creates a user-song matrix with playcounts as values

Computes cosine similarity between songs based on co-listening patterns

3. Predictive Modeling (XGBoost)
Builds a classifier to predict whether a user will replay a song

Features include:

Audio characteristics (energy, loudness, etc.)

Text features (artist, tags) processed with TF-IDF

User ID encoded as a categorical feature

Uses GPU acceleration for faster training

Hybrid Recommendation Function
The core function hybrid_recommend():

Takes a user ID and seed song name as input

Computes three scores for each song:

Tag similarity score (content-based)

Collaborative filtering score

Repeat probability score (from XGBoost model)

Combines these scores with weights (40% content, 30% collaborative, 30% repeat probability)

Returns the top N most similar songs
Usage
To get recommendations: recommendations = hybrid_recommend(
    user_id='5f2087e8f0171a2b5ed6258e158c75b921411e79', 
    song_name='My Little Corner of the World', 
    top_n=10
)
