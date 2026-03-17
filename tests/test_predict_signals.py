"""
Tests for predict_signals.py
-------------------------------
Validates the inference pipeline using mocked model and database.
"""

import pytest
import sys
import os
import tempfile

import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

from predict_signals import load_model


@pytest.fixture
def mock_model():
    """Create a simple trained XGBClassifier for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.choice([0, 1], 100)
    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    return model


class TestLoadModel:
    def test_load_model_from_file(self, mock_model):
        """Model should load successfully from a valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "xgb_model.pkl")
            joblib.dump(mock_model, model_path)
            
            # Patch MODELS_DIR so no metadata file is found
            import predict_signals as ps
            original = ps.MODELS_DIR
            ps.MODELS_DIR = tmpdir
            try:
                loaded, version = load_model(model_path)
                assert loaded is not None
                assert version == "unknown"  # No metadata file in tmpdir
            finally:
                ps.MODELS_DIR = original
    
    def test_load_model_with_metadata(self, mock_model):
        """Model version should be read from metadata file."""
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "xgb_model.pkl")
            joblib.dump(mock_model, model_path)
            
            # Create metadata file
            meta = {"model_version": "v20240101_120000"}
            meta_path = os.path.join(tmpdir, "model_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            
            # Patch MODELS_DIR so metadata can be found
            import predict_signals as ps
            original = ps.MODELS_DIR
            ps.MODELS_DIR = tmpdir
            try:
                loaded, version = load_model(model_path)
                assert version == "v20240101_120000"
            finally:
                ps.MODELS_DIR = original
    
    def test_load_model_file_not_found(self):
        """Should raise FileNotFoundError for missing model file."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pkl")


class TestModelPredictions:
    def test_model_outputs_binary_predictions(self, mock_model):
        """Model predictions should be 0 or 1."""
        X = np.random.rand(10, 5)
        preds = mock_model.predict(X)
        assert set(preds).issubset({0, 1})
    
    def test_model_outputs_probabilities(self, mock_model):
        """Probabilities should be between 0 and 1."""
        X = np.random.rand(10, 5)
        probs = mock_model.predict_proba(X)[:, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_prediction_shape(self, mock_model):
        """Should produce one prediction per input row."""
        X = np.random.rand(25, 5)
        preds = mock_model.predict(X)
        assert len(preds) == 25
