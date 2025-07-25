Movie Recommendation System
This repository contains a Jupyter Notebook (model_traing.ipynb) that implements a content-based movie recommendation system. The system leverages movie metadata such as genres, keywords, cast, crew, and overview to suggest similar movies.

Table of Contents
Project Overview

Features

Data Sources

Usage

File Structure

Dependencies

How it Works

Project Overview
The goal of this project is to build a simple movie recommendation engine. It processes movie data, cleans and transforms it, and then uses cosine similarity to find movies that are similar based on their textual features. The core logic is encapsulated within a Jupyter Notebook, which demonstrates the data preprocessing, model training (similarity calculation), and a basic recommendation function.

Features
Content-Based Filtering: Recommends movies based on the similarity of their content (genres, keywords, cast, crew, overview).

Data Preprocessing: Includes steps for handling missing values, extracting relevant information from nested JSON-like strings, and text cleaning (stemming, stopword removal).

Cosine Similarity: Utilizes cosine similarity to measure the likeness between movie feature vectors.

Interactive Notebook: The model_traing.ipynb notebook provides a step-by-step execution of the recommendation system's development.

Model Persistence: Saves the processed movie data and the calculated similarity matrix for later use.

Data Sources
The project uses two CSV files containing movie data:

tmdb_5000_movies.csv: Contains general movie information (budget, genres, homepage, overview, etc.).

tmdb_5000_credits.csv: Contains cast and crew information for movies.

(Note: These files are assumed to be present in the same directory as the notebook or in a specified data directory.)

Usage
Start Jupyter Notebook:

jupyter notebook


Open the notebook:
Navigate to model_traing.ipynb in your browser and open it.

Run all cells:
Execute all cells in the notebook sequentially. This will:

Load and merge the datasets.

Perform data cleaning and feature engineering.

Compute the cosine similarity matrix.

Define the recommend function.

Save the modified_data DataFrame and similary matrix to the artificats directory.

Get Recommendations:
Once all cells are run, you can use the recommend function to get movie suggestions. For example:

recommend('Avatar')


This will print a list of 5 recommended movies.

File Structure
.
├── model_traing.ipynb
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── artificats/
    ├── movie_list.pkl
    └── similarity.pkl


model_traing.ipynb: The main Jupyter Notebook containing the recommendation system logic.

tmdb_5000_movies.csv: Dataset containing movie details.

tmdb_5000_credits.csv: Dataset containing movie cast and crew.

artificats/: Directory where the processed movie list and similarity matrix are saved.

movie_list.pkl: Pickled DataFrame of modified movie data.

similarity.pkl: Pickled cosine similarity matrix.

Dependencies
pandas

numpy

scikit-learn

nltk

ast (built-in Python module)

How it Works
The recommendation system follows these steps:

Data Loading and Merging: tmdb_5000_movies.csv and tmdb_5000_credits.csv are loaded and merged into a single DataFrame.

Feature Engineering:

Relevant columns (movie_id, genres, keywords, cast, crew, overview, title) are selected.

JSON-like strings in genres, keywords, cast, and crew are parsed to extract meaningful information (e.g., genre names, top 3 cast members, director).

Spaces are removed from names (e.g., "Science Fiction" becomes "ScienceFiction") to treat multi-word entities as single tokens.

The overview text is split into words.

Tag Creation: A new tags column is created by concatenating the processed overview, keywords, genres, crew, and cast lists. This column represents the combined textual features of each movie.

Text Preprocessing (Stemming & Stopword Removal): The tags text undergoes stemming (reducing words to their root form, e.g., "loved" to "love") and stopword removal (removing common words like "the", "is", "a") using NLTK's Porter Stemmer.

Vectorization: CountVectorizer is used to convert the tags text into a matrix of token counts. Each row represents a movie, and each column represents a unique word/token from the tags corpus.

Cosine Similarity Calculation: The cosine_similarity function from sklearn.metrics.pairwise is used to compute the similarity between every pair of movie vectors. This results in a similarity matrix where similary[i][j] represents the similarity between movie i and movie j.

Recommendation Function: The recommend(movie_title) function takes a movie title as input, finds its index in the modified_data DataFrame, retrieves its similarity scores from the similary matrix, sorts movies by similarity in descending order, and returns the top 5 most similar movies (excluding the input movie itself).

Model Saving: The modified_data DataFrame and the similary matrix are saved using pickle to artificats/movie_list.pkl and artificats/similarity.pkl respectively, allowing the trained model to be loaded and used without re-running the entire preprocessing and similarity calculation steps.
