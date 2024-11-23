#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import joblib

class FinalRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.scaler = StandardScaler()
        self.user_item_matrix = None
        self.users_df = None
        self.movies_df = None
        
    def load_data(self):
        """Load all required data files"""
        # Load ratings
        self.ratings_df = pd.read_csv(
            'u.data',
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Load user information
        self.users_df = pd.read_csv(
            'u.user',
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        # Load movie information
        self.movies_df = pd.read_csv(
            'u.item',
            sep='|',
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                  'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                  'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
        )
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        return self.user_item_matrix
    
    def train_model(self):
        """Train model and display metrics"""
        # Load and prepare data
        user_item_matrix = self.load_data()
        
        # Scale the data
        scaled_matrix = self.scaler.fit_transform(user_item_matrix)
        
        # Train SVD model
        user_item_latent = self.svd.fit_transform(scaled_matrix)
        reconstructed = self.svd.inverse_transform(user_item_latent)
        reconstructed = self.scaler.inverse_transform(reconstructed)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(
            self.user_item_matrix.values.flatten(),
            reconstructed.flatten()
        ))
        mae = mean_absolute_error(
            self.user_item_matrix.values.flatten(),
            reconstructed.flatten()
        )
        r2 = r2_score(
            self.user_item_matrix.values.flatten(),
            reconstructed.flatten()
        )
        
        # Calculate accuracy
        tolerance = 1.0
        close_predictions = np.abs(
            self.user_item_matrix.values.flatten() - reconstructed.flatten()
        ) <= tolerance
        accuracy = np.mean(close_predictions) * 100
        
        # Print metrics in single line
        print(f"RMSE = {rmse:.4f} MAE = {mae:.4f} R2 = {r2:.4f} Accuracy = {accuracy:.2f}%")
        
        return self
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        try:
            if user_id not in self.user_item_matrix.index or movie_id not in self.user_item_matrix.columns:
                return None
            
            user_latent = self.svd.transform(
                self.scaler.transform(self.user_item_matrix)
            )
            reconstructed = self.scaler.inverse_transform(
                self.svd.inverse_transform(user_latent)
            )
            
            reconstructed_df = pd.DataFrame(
                reconstructed, 
                index=self.user_item_matrix.index, 
                columns=self.user_item_matrix.columns
            )
            
            prediction = reconstructed_df.loc[user_id, movie_id]
            return np.clip(prediction, 1, 5)
            
        except Exception:
            return None

def main():
    # Initialize and train
    recommender = FinalRecommender()
    recommender.train_model()
    
    # Save model
    joblib.dump(recommender, 'svd_demo.joblib')
    
    # Example predictions
    print("\nExample predictions")
    test_cases = [
        (1, 1),    
        (50, 100), 
        (100, 200)
        
    ]
    
    for i, (user_id, movie_id) in enumerate(test_cases, 1):
        pred = recommender.predict_rating(user_id, movie_id)
        if pred is not None:
            print(f"{i}.User id: {user_id}, Movie id: {movie_id}, Predicted = {pred:.2f}")

if __name__ == "__main__":
    main()

