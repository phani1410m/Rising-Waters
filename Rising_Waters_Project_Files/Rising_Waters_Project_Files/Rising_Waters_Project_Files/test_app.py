# -*- coding: utf-8 -*-
"""
Rising Waters: AI-Powered Flood Prediction System
Functional Testing & QA Verification Suite

This module runs automated unit tests against the Flask application routes,
input validation parameters, model inference routing, and HTML status outputs.
"""

import unittest
from app import app, validate_inputs


class TestRisingWatersApp(unittest.TestCase):
    def setUp(self):
        # Configure the Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        """Verify that the landing page renders successfully."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Rising Waters', response.data)
        self.assertIn(b'Pravalika', response.data)

    def test_predict_route(self):
        """Verify that the prediction form renders successfully."""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Input', response.data)
        self.assertIn(b'Meteorological Parameters', response.data)

    def test_validation_missing_parameters(self):
        """Verify that validation flags missing parameters."""
        form_data = {
            "annual_rainfall": "1200.5",
            "cloud_coverage": "75.0",
            # missing jun_sep and others
        }
        features, err = validate_inputs(form_data)
        self.assertEqual(features, [])
        self.assertIn("Missing input parameter", err)

    def test_validation_non_numeric(self):
        """Verify that validation rejects non-numeric entries."""
        form_data = {
            "annual_rainfall": "abc",
            "cloud_coverage": "75.0",
            "jun_sep": "800.0",
            "mar_may": "120.0",
            "oct_dec": "150.0",
            "jan_feb": "30.0"
        }
        features, err = validate_inputs(form_data)
        self.assertEqual(features, [])
        self.assertIn("must represent valid floating-point numbers", err)

    def test_validation_negative_rainfall(self):
        """Verify that validation rejects negative rainfall parameters."""
        form_data = {
            "annual_rainfall": "-100.0",
            "cloud_coverage": "75.0",
            "jun_sep": "800.0",
            "mar_may": "120.0",
            "oct_dec": "150.0",
            "jan_feb": "30.0"
        }
        features, err = validate_inputs(form_data)
        self.assertEqual(features, [])
        self.assertIn("must be non-negative values", err)

    def test_validation_cloud_coverage_bounds(self):
        """Verify that validation rejects cloud coverage values outside [0, 100]."""
        form_data = {
            "annual_rainfall": "1200.5",
            "cloud_coverage": "120.0",
            "jun_sep": "800.0",
            "mar_may": "120.0",
            "oct_dec": "150.0",
            "jan_feb": "30.0"
        }
        features, err = validate_inputs(form_data)
        self.assertEqual(features, [])
        self.assertIn("must sit between 0% and 100%", err)

    def test_valid_input_no_flood_prediction(self):
        """Verify that low precipitation values route to the safe forecast screen."""
        form_data = {
            "annual_rainfall": "10.0",
            "cloud_coverage": "5.0",
            "jun_sep": "5.0",
            "mar_may": "2.0",
            "oct_dec": "2.0",
            "jan_feb": "1.0"
        }
        response = self.app.post('/result', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Safe Forecast: No Flood Risk', response.data)
        self.assertIn(b'Stable precipitation metrics', response.data)

    def test_valid_input_flood_prediction(self):
        """Verify that high precipitation parameters route to the warning screen."""
        form_data = {
            "annual_rainfall": "3200.0",
            "cloud_coverage": "85.0",
            "jun_sep": "2200.0",
            "mar_may": "300.0",
            "oct_dec": "300.0",
            "jan_feb": "150.0"
        }
        response = self.app.post('/result', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'WARNING: Flood Risk Detected', response.data)
        self.assertIn(b'Extreme precipitation ratios', response.data)

    def test_invalid_post_handles_gracefully(self):
        """Verify that posting invalid inputs to the result route yields bad request errors."""
        form_data = {
            "annual_rainfall": "invalid_value",
            "cloud_coverage": "85.0",
            "jun_sep": "2200.0",
            "mar_may": "300.0",
            "oct_dec": "300.0",
            "jan_feb": "150.0"
        }
        response = self.app.post('/result', data=form_data)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Error:', response.data)


if __name__ == '__main__':
    unittest.main()
