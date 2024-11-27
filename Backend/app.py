import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from NMF import AdvancedMovieRecommender
import logging
import numpy as np

# Prometheus Metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Prometheus Metrics Setup
REQUEST_COUNT = Counter(
    'app_requests_total', 
    'Total Request Count', 
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds', 
    'Request Latency', 
    ['method', 'endpoint']
)

PREDICTION_PROCESSING_TIME = Histogram(
    'prediction_processing_time_seconds', 
    'Time spent processing predictions'
)

def track_metrics(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        request_latency = REQUEST_LATENCY.labels(request.method, request.path)
        start_time = time.time()
        
        try:
            response = f(*args, **kwargs)
            status_code = response[1] if isinstance(response, tuple) else 200
            
            REQUEST_COUNT.labels(request.method, request.path, status_code).inc()
            request_latency.observe(time.time() - start_time)
            
            return response
        except Exception as e:
            REQUEST_COUNT.labels(request.method, request.path, 500).inc()
            raise
    return wrapper

def load_model(model_path='NMF.joblib'):
    """Load and reconstruct the model from saved components"""
    try:
        logger.info(f"Attempting to load model from {model_path}")
        
        # Load the saved dictionary
        model_data = joblib.load(model_path)
        logger.debug(f"Loaded model keys: {model_data.keys()}")
        
        # Create a new recommender instance
        recommender = AdvancedMovieRecommender()
        
        # Restore all the components
        recommender.nmf_model = model_data['nmf_model']
        recommender.user_features_matrix = model_data['user_features_matrix']
        recommender.movie_features_matrix = model_data['movie_features_matrix']
        recommender.user_movie_matrix = model_data['user_movie_matrix']
        recommender.accuracy_metrics = model_data['accuracy_metrics']
        
        logger.info("Model loaded successfully")
        return recommender
    except FileNotFoundError:
        logger.error(f"Model file {model_path} not found")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load the model when the app starts
recommender = load_model()

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
@track_metrics
def predict_rating():
    if recommender is None:
        return jsonify({"error": "Model not loaded correctly"}), 500
    
    data = request.json
    logger.debug(f"Received prediction request: {data}")
    
    if not data or 'user_id' not in data or 'movie_id' not in data:
        return jsonify({"error": "Missing user_id or movie_id"}), 400
    
    try:
        user_id = int(data['user_id'])
        movie_id = int(data['movie_id'])
        
        # Track prediction processing time
        with PREDICTION_PROCESSING_TIME.time():
            # Get prediction
            prediction = recommender.predict_rating(user_id, movie_id)
            logger.debug(f"Raw prediction result: {prediction}")
        
        # Handle different prediction formats
        if isinstance(prediction, dict):
            if 'rating' in prediction:
                prediction_value = prediction['rating']
            else:
                prediction_value = prediction.get('predicted_rating', None)
        else:
            prediction_value = prediction
            
        # Convert numpy types to Python native types
        if isinstance(prediction_value, (np.floating, np.integer)):
            prediction_value = float(prediction_value)
            
        # Validate prediction value
        if prediction_value is None or not isinstance(prediction_value, (int, float)) or np.isnan(prediction_value):
            raise ValueError("Invalid prediction value received from model")
        
        # Format response
        response = {
            "prediction": prediction_value,
            "user_id": user_id,
            "movie_id": movie_id
        }
        
        logger.debug(f"Sending response: {response}")
        return jsonify(response)
    
    except ValueError as ve:
        error_msg = f"Validation error: {str(ve)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy" if recommender is not None else "unhealthy",
        "model_loaded": recommender is not None,
        "model_details": None
    }
    
    if recommender is not None:
        status["model_details"] = {
            "n_components": recommender.n_components,
            "accuracy_metrics": recommender.accuracy_metrics,
            "matrix_shape": recommender.user_movie_matrix.shape if recommender.user_movie_matrix is not None else None
        }
    
    return jsonify(status)

@app.route('/model-info', methods=['GET'])
def model_info():
    """Additional endpoint to get detailed model information"""
    if recommender is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        return jsonify({
            "user_count": recommender.user_movie_matrix.shape[0],
            "movie_count": recommender.user_movie_matrix.shape[1],
            "n_components": recommender.n_components,
            "accuracy_metrics": recommender.accuracy_metrics,
            "sparsity": (recommender.user_movie_matrix == 0).sum() / recommender.user_movie_matrix.size * 100
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if recommender is None:
        logger.error("Failed to load model. Please check if 'NMF.joblib' exists and is valid.")
    else:
        logger.info("Model loaded successfully. Starting server...")
        app.run(host='0.0.0.0', port=5000)