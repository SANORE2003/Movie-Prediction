from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import threading
import logging

app = Flask(__name__)
CORS(app)

# Define Prometheus Metrics
REQUEST_COUNT = Counter(
    'app_requests_total', 
    'Total count of requests', 
    ['method', 'endpoint', 'http_status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 
    'Time spent making predictions',
    ['model_type']
)

PREDICTION_COUNT = Counter(
    'predictions_total', 
    'Total number of predictions made',
    ['user_id', 'movie_id']
)

PREDICTION_RATING = Gauge(
    'predicted_rating', 
    'Predicted rating for a movie',
    ['user_id', 'movie_id']
)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint to expose Prometheus metrics"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    # Start tracking request metrics
    REQUEST_COUNT.labels(method=request.method, endpoint='/predict', http_status=200).inc()
    
    data = request.json
    user_id = data.get('user_id')
    movie_id = data.get('movie_id')
    
    # Track prediction count and rating
    PREDICTION_COUNT.labels(user_id=str(user_id), movie_id=str(movie_id)).inc()
    
    # Use a histogram to track prediction latency
    with PREDICTION_LATENCY.labels(model_type='NMF').time():
        # Your existing prediction logic here
        result = {
            'user_id': user_id, 
            'movie_id': movie_id, 
            'predicted_rating': 5  # Replace with actual prediction
        }
    
    # Update gauge with predicted rating
    PREDICTION_RATING.labels(user_id=str(user_id), movie_id=str(movie_id)).set(result['predicted_rating'])
    
    return jsonify(result)

def start_prometheus_server():
    """Start a separate thread for Prometheus metrics server"""
    start_http_server(8000)  # Metrics will be available at http://localhost:8000/metrics

if __name__ == '__main__':
    # Start Prometheus metrics server in a separate thread
    threading.Thread(target=start_prometheus_server, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000)