import React, { useState } from 'react';
import movieBackground from "../assets/movie.jpg"; // Adjust path based on file location

const MainPage = () => {
  const [userId, setUserId] = useState('');
  const [movieId, setMovieId] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      if (!userId || !movieId) {
        throw new Error('Please enter both User ID and Movie ID');
      }

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: parseInt(userId),
          movie_id: parseInt(movieId),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Server error occurred');
      }

      if (data && typeof data.prediction === 'number' && !isNaN(data.prediction)) {
        setPrediction(data.prediction);
      } else {
        throw new Error('Invalid prediction response');
      }
    } catch (err) {
      console.error('Prediction Error:', err);
      setError(err.message || 'An unexpected error occurred');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>
        {`
          body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            background: url(${movieBackground}) no-repeat center center fixed;
            background-size: cover;
          }

          .page-header {
            background-color: rgba(26, 54, 93, 0.8);
            color: white;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          }

          .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
          }

          .header-subtitle {
            font-size: 1.25rem;
            opacity: 0.9;
            margin-top: 0.5rem;
          }

          .main-content {
            min-height: calc(100vh - 116px);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
          }

          .card {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 400px;
          }

          .input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            font-size: 1rem;
            color: #2d3748;
            background-color: #ffffff;
            transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
          }

          .input:focus {
            border-color: #3182ce;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.5);
            outline: none;
          }

          .button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1rem;
            font-weight: 600;
            color: white;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s ease-in-out, transform 0.2s ease-in-out;
          }

          .button:disabled {
            background-color: #cbd5e0;
            cursor: not-allowed;
          }

          .button:hover:not(:disabled) {
            background-color: #2b6cb0;
            transform: scale(1.02);
          }

          .alert {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
          }

          .alert.error {
            background-color: #fff5f5;
            border: 1px solid #fed7d7;
            color: #e53e3e;
          }

          .alert.success {
            background-color: #f0fff4;
            border: 1px solid #c6f6d5;
            color: #38a169;
          }
        `}
      </style>

      <header className="page-header">
        <h1 className="header-title">Movie Rating Prediction</h1>
        <p className="header-subtitle">Predict user ratings for movies using AI</p>
      </header>

      <div className="main-content">
        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                User ID:
              </label>
              <input
                type="number"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Enter user ID"
                min="1"
                required
                className="input"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Movie ID:
              </label>
              <input
                type="number"
                value={movieId}
                onChange={(e) => setMovieId(e.target.value)}
                placeholder="Enter movie ID"
                min="1"
                required
                className="input"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`button ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600'}`}
            >
              {loading ? 'Predicting...' : 'Predict Rating'}
            </button>
          </form>

          {error && (
            <div className="alert error">
              <p>{error}</p>
            </div>
          )}

          {prediction !== null && (
            <div className="alert success">
              <h3>Prediction Result</h3>
              <p className="text-2xl">{prediction.toFixed(2)} / 5.0</p>
              <p>Predicted rating for User {userId} and Movie {movieId}</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default MainPage;
