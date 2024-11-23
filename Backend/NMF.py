#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import joblib  # Added import for model saving

class AdvancedMovieRecommender:
    def __init__(self, n_components=50, random_state=42, max_iter=500):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.nmf_model = None
        self.user_features_matrix = None
        self.movie_features_matrix = None
        self.user_movie_matrix = None
        self.accuracy_metrics = {}

    def load_data(self, data_path=''):
        """Load ratings data"""
        ratings = pd.read_csv(
            f'{data_path}u.data', 
            sep='\t', 
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Create user-movie rating matrix
        self.user_movie_matrix = ratings.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        return ratings, self.user_movie_matrix

    def preprocess_data(self, user_movie_matrix):
        """Preprocess data for NMF"""
        # Normalize to non-negative range
        scaled_matrix = (user_movie_matrix - user_movie_matrix.min()) / (user_movie_matrix.max() - user_movie_matrix.min())
        return scaled_matrix.values

    def calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy metrics with error handling"""
        # Clip predictions to valid range
        y_pred = np.clip(y_pred, 0, 1)
        y_true = np.clip(y_true, 0, 1)
        
        # Compute metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Binary accuracy (within threshold)
        binary_accuracy = np.mean(np.abs(y_true - y_pred) <= 0.1) * 100
        
        self.accuracy_metrics = {
            'Mean Absolute Error (MAE)': mae,
            'Root Mean Squared Error (RMSE)': rmse,
            'Binary Accuracy (±0.1)': binary_accuracy
        }
        
        return self.accuracy_metrics

    def train_model(self, data_path=''):
        """Train recommendation model"""
        # Suppress NMF warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Load data
        ratings, user_movie_matrix = self.load_data(data_path)
        scaled_matrix = self.preprocess_data(user_movie_matrix)
        
        # Train NMF
        self.nmf_model = NMF(
            n_components=self.n_components, 
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        
        # Fit model
        self.user_features_matrix = self.nmf_model.fit_transform(scaled_matrix)
        self.movie_features_matrix = self.nmf_model.components_
        
        # Reconstruct matrix
        reconstructed_matrix = np.dot(
            self.user_features_matrix, 
            self.movie_features_matrix
        )
        
        # Calculate and print accuracy
        accuracy = self.calculate_accuracy(
            scaled_matrix, 
            reconstructed_matrix
        )
        
        # Print detailed accuracy metrics
        print("\n--- Model Accuracy Metrics ---")
        for metric, value in accuracy.items():
            print(f"{metric}: {value}")
        
        # Save the trained model
        self.save_model('nmf_demo.joblib')
        print("\nModel saved successfully as 'nmf_demo.joblib'")
        
        return self

    def save_model(self, filename):
        """Save the trained model and its components"""
        model_data = {
            'nmf_model': self.nmf_model,
            'user_features_matrix': self.user_features_matrix,
            'movie_features_matrix': self.movie_features_matrix,
            'user_movie_matrix': self.user_movie_matrix,
            'accuracy_metrics': self.accuracy_metrics
        }
        joblib.dump(model_data, filename)

    def predict_rating(self, user_id, movie_id):
        """Predict rating for specific user and movie"""
        try:
            # Verify user and movie exist
            if user_id not in self.user_movie_matrix.index:
                return {"error": f"User {user_id} not found"}
            if movie_id not in self.user_movie_matrix.columns:
                return {"error": f"Movie {movie_id} not found"}
            
            # Find matrix indices
            user_index = list(self.user_movie_matrix.index).index(user_id)
            movie_index = list(self.user_movie_matrix.columns).index(movie_id)
            
            # Predict rating
            prediction = np.dot(
                self.user_features_matrix[user_index], 
                self.movie_features_matrix[:, movie_index]
            )
            
            # Scale prediction to 1-5 range
            prediction = max(1, min(5, prediction * 5))
            
            return {
                'user_id': user_id,
                'movie_id': movie_id,
                'predicted_rating': round(prediction, 2)
            }
        
        except Exception as e:
            return {"error": str(e)}

def main():
    # Initialize and train recommender
    recommender = AdvancedMovieRecommender()
    recommender.train_model()
    
    # Example predictions
    test_cases = [
        (1, 1),    # User 1, Movie 1
        (50, 100), # User 50, Movie 100
        (100, 200),# User 100, Movie 200
        (200, 300) # User 200, Movie 300
    ]
    
    print("\nExample Predictions:")
    for user_id, movie_id in test_cases:
        prediction = recommender.predict_rating(user_id, movie_id)
        print(f"\nPrediction: {prediction}")

if __name__ == "__main__":
    main()


# In[ ]:




