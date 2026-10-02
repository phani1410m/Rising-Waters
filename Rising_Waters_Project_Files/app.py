"""
Rising Waters: AI-Powered Flood Prediction System
Core Flask Server Application

This module handles web routing, input validation, and real-time inference
using a pre-trained XGBoost classification model and StandardScaler pipeline.
"""

import logging
import os
from typing import Dict, Any, List, Tuple
from flask import Flask, render_template, request
import joblib
import numpy as np

# Configure system logs for production audit trail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RisingWatersApp")

app = Flask(__name__)

# Define file system paths to serialized model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "floods.save")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Initialize global placeholders for ML pipeline
model = None
scaler = None


def load_artifacts() -> None:
    """
    Loads the trained model and StandardScaler from disk.
    Gracefully logs warnings if artifacts are missing.
    """
    global model, scaler
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                "Model files missing. Train the model by executing "
                "'python notebooks/04_model_training.py' first."
            )
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Machine learning model and scaler loaded successfully.")
    except Exception as err:
        logger.error("Failed to initialize model artifacts: %s", err)


# Load artifacts at server startup
load_artifacts()


def validate_inputs(form: Dict[str, str]) -> Tuple[List[float], str]:
    """
    Validates form input parameters to ensure data integrity.

    Args:
        form: Form data dictionary from request.form.

    Returns:
        A tuple of (parsed_features, error_message). If valid, error_message is empty.
    """
    required_fields = [
        "annual_rainfall",
        "cloud_coverage",
        "jun_sep",
        "mar_may",
        "oct_dec",
        "jan_feb"
    ]
    
    # Check for missing values
    for field in required_fields:
        if field not in form or not form[field].strip():
            return [], f"Missing input parameter: {field.replace('_', ' ').title()}"

    try:
        # Convert values to float and check bounds
        annual_rainfall = float(form["annual_rainfall"])
        cloud_coverage = float(form["cloud_coverage"])
        jun_sep = float(form["jun_sep"])
        mar_may = float(form["mar_may"])
        oct_dec = float(form["oct_dec"])
        jan_feb = float(form["jan_feb"])

        if min(annual_rainfall, jun_sep, mar_may, oct_dec, jan_feb) < 0:
            return [], "Rainfall parameters must be non-negative values."

        if not (0 <= cloud_coverage <= 100):
            return [], "Cloud coverage must sit between 0% and 100%."

        # Compile features in the required model order
        features = [
            annual_rainfall,
            cloud_coverage,
            jun_sep,
            mar_may,
            oct_dec,
            jan_feb
        ]
        return features, ""

    except ValueError:
        return [], "All inputs must represent valid floating-point numbers."


def generate_mitigation_data(prediction: int, probability: float) -> List[Dict[str, str]]:
    """
    Generates structured mitigative tasks and advisory status messages
    based on model class prediction and calculated probability.

    Args:
        prediction: Binary label (1 for Flood Risk, 0 for No Flood Risk).
        probability: Score between 0.0 and 1.0.

    Returns:
        A list of dictionaries containing 'icon' and 'text'.
    """
    if prediction == 1:
        if probability >= 0.90:
            return [
                {"icon": "📢", "text": "CRITICAL: Issue immediate red alerts to regional civil bodies."},
                {"icon": "🏕️", "text": "EVACUATE: Trigger mandatory evacuations in low-lying hazard zones."},
                {"icon": "🚁", "text": "RESCUE: Position State Disaster Response Forces and medical assets."},
                {"icon": "🚧", "text": "BLOCK: Restrict transport lines across known floodplains."}
            ]
        elif probability >= 0.70:
            return [
                {"icon": "⚠️", "text": "WARNING: Send orange warning alerts to district administrations."},
                {"icon": "🏕️", "text": "PREPARE: Advise readiness for evacuations in vulnerable sectors."},
                {"icon": "🚑", "text": "MOBILIZE: Alert local emergency response teams and stock shelters."},
                {"icon": "📡", "text": "MONITOR: Track watershed water levels and upstream gauges."}
            ]
        else:
            return [
                {"icon": "🟡", "text": "WATCH: Post active flood watch alerts for local riverbeds."},
                {"icon": "📋", "text": "CHECK: Request drainage maintenance logs from municipal teams."},
                {"icon": "🚯", "text": "MAINTAIN: Clear trash blockages from critical urban storm drains."},
                {"icon": "🌤️", "text": "UPDATE: Keep track of satellite cloud cover updates."}
            ]
    else:
        # Safe scenarios: probability here is the certainty of 'No Flood' (1.0 - flood_probability)
        if probability >= 0.90:
            return [
                {"icon": "📈", "text": "Precipitation and cloud indexes sit within normal limits."},
                {"icon": "🌤️", "text": "Atmospheric conditions show stable regional parameters."},
                {"icon": "📋", "text": "Hydrological sensor log schedules remain on weekly baselines."},
                {"icon": "✅", "text": "Water reservoir capacity remains safe with full buffer margins."}
            ]
        elif probability >= 0.70:
            return [
                {"icon": "🌤️", "text": "No atmospheric anomalies detected in seasonal models."},
                {"icon": "📋", "text": "Hydrological telemetry logs daily standard runs."},
                {"icon": "🌊", "text": "River basin levels are stable and within capacity bounds."},
                {"icon": "✅", "text": "No special actions or warnings are required at this time."}
            ]
        else:
            return [
                {"icon": "🟡", "text": "Elevated Watch: Seasonal rain is slightly elevated but stable."},
                {"icon": "📋", "text": "Check storm gutters and clean drainage zones proactively."},
                {"icon": "🌊", "text": "Reservoir inflows are elevated but within normal boundaries."},
                {"icon": "📡", "text": "Monitor minor weather alerts for sudden rainfall changes."}
            ]


@app.route("/")
def home():
    """Renders the landing homepage."""
    return render_template("index.html")


@app.route("/predict")
def predict():
    """Renders the weather data input form."""
    return render_template("input.html")


@app.route("/result", methods=["POST"])
def result():
    """
    Handles form submission, validates input parameters, scales data,
    performs classification, and renders the result templates.
    """
    if model is None or scaler is None:
        logger.error("Inference requested but model artifacts are not initialized.")
        return "Prediction service is currently offline. Artifacts failed to load.", 503

    # Validate inputs
    features_list, err_msg = validate_inputs(request.form)
    if err_msg:
        logger.warning("Submission validation failed: %s", err_msg)
        return f"Error: {err_msg}", 400

    try:
        # Scale input features using a DataFrame with correct feature names
        import pandas as pd
        FEATURE_COLS = [
            'ANNUAL_RAINFALL', 
            'CLOUD_COVERAGE', 
            'JUN-SEP', 
            'MAR-MAY', 
            'OCT-DEC', 
            'JAN-FEB'
        ]
        features_df = pd.DataFrame([features_list], columns=FEATURE_COLS)
        features_scaled = scaler.transform(features_df)

        # Run classification
        prediction = int(model.predict(features_scaled)[0])
        
        # Calculate probability/confidence score
        probabilities = model.predict_proba(features_scaled)[0]
        flood_prob = float(probabilities[1])
        
        # Format the user inputs for presentation as an explanation element
        input_data = {
            "Annual Rainfall": f"{features_list[0]:,.1f} mm",
            "Cloud Coverage": f"{features_list[1]:.1f}%",
            "June - Sept Rainfall": f"{features_list[2]:,.1f} mm",
            "March - May Rainfall": f"{features_list[3]:,.1f} mm",
            "Oct - Dec Rainfall": f"{features_list[4]:,.1f} mm",
            "Jan - Feb Rainfall": f"{features_list[5]:,.1f} mm",
        }
        
        if prediction == 1:
            confidence_percentage = flood_prob * 100
            confidence_str = f"{confidence_percentage:.2f}%"
            recommendations = generate_mitigation_data(1, flood_prob)
            
            logger.info(
                "Prediction run complete. Result: FLOOD RISK (%s confidence)", 
                confidence_str
            )
            return render_template(
                "result_flood.html", 
                confidence=confidence_str, 
                recommendations=recommendations,
                inputs=input_data
            )
        else:
            safe_prob = float(probabilities[0])
            confidence_percentage = safe_prob * 100
            confidence_str = f"{confidence_percentage:.2f}%"
            recommendations = generate_mitigation_data(0, safe_prob)
            
            logger.info(
                "Prediction run complete. Result: NO FLOOD RISK (%s confidence)", 
                confidence_str
            )
            return render_template(
                "result_no_flood.html", 
                confidence=confidence_str, 
                recommendations=recommendations,
                inputs=input_data
            )

    except Exception as err:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Inference runtime error: %s", err)
        logger.error("Full Traceback:\n%s", error_traceback)
        return f"Internal Server Error: {str(err)}<br><br><strong>Traceback:</strong><br><pre>{error_traceback}</pre>", 500


if __name__ == "__main__":
    # Host on all interfaces and bind to env PORT or default 5000
    server_port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Rising Waters server on port %s", server_port)
    app.run(host="0.0.0.0", port=server_port)
