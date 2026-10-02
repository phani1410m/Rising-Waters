# 🌊 Rising Waters: AI-Powered Flood Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/Flask-2.3.3-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-006600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

**An intelligent, machine-learning-driven flood forecasting application that delivers real-time hazard risk assessments using predictive meteorological features through an interactive web-based interface.**

---

## 📌 Project Overview

**Rising Waters** is a state-of-the-art predictive platform designed to assess and forecast regional flood probabilities. By analyzing critical weather variables—including annual precipitation, cloud cover, and seasonal rainfall breakdowns—the system runs historical data through advanced machine learning classifiers. 

Equipped with a clean and responsive web dashboard built on Flask, the system provides instant, user-friendly risk assessments. This bridges the gap between complex statistical meteorology and actionable early-warning alerts, supporting disaster preparedness, relief organizations, and local communities.

---

## ⚠️ Problem Statement

Extreme flooding incidents cause catastrophic socio-economic damage, human displacement, and loss of life annually. Existing meteorological predictive workflows are often slow, computationally demanding, and inaccessible to the general public. Furthermore, standard forecasting relies heavily on physical simulations that struggle to scale efficiently across regional terrains or process short-term variations dynamically.

There is a vital requirement for an accessible, low-latency machine learning tool that transforms localized precipitation records and atmospheric factors into reliable risk metrics, allowing stakeholders to take pre-emptive safety measures.

---

## 🎯 Objectives

- **Develop a robust ML Pipeline**: Preprocess and scale weather indicators to feed multiple predictive classifiers.
- **Model Evaluation & Optimization**: Train and contrast models (KNN, Decision Trees, Random Forests, XGBoost) to determine the most accurate predictor.
- **Provide Real-Time Inference**: Host the selected champion model on a Flask server for immediate prediction output.
- **Visual Analytics**: Generate comprehensive plots highlighting underlying data trends and relationships.
- **Achieve Production Quality**: Secure low-latency performance and high reliability, validated through performance benchmark testing.

---

## ⚡ Features

| Feature | Details |
| :--- | :--- |
| **🏆 Champion XGBoost Model** | Achieves a top-tier accuracy of **95.42%** for highly dependable classifications. |
| **🤖 Multi-Classifier Engine** | Compares Decision Trees, Random Forests, KNN, and XGBoost classifiers. |
| **📊 Automated Preprocessing** | Handles outlier capping via IQR, handles missing values, and fits `StandardScaler` profiles. |
| **📈 Interactive Visualizations** | Integrates correlation matrices, distribution histograms, and scatter plots. |
| **💻 Responsive Web UI** | Features a premium Glassmorphism-style UI for entering features and viewing predictions. |
| **🚀 Stress-Tested Backend** | Benchmarked under high-load conditions using Apache JMeter (~17ms average response time). |

---

## 🛠️ Technologies Used

### Core Programming & Scripting
- **Python (v3.10+)**: Language powering the data science pipeline and server backend.

### Machine Learning & Data Processing
- **XGBoost (v2.0.3)**: High-performance gradient boosting library.
- **Scikit-Learn (v1.3.2)**: Core machine learning algorithms, scaling utilities, and validation metrics.
- **Pandas (v2.1.4)** & **NumPy (v1.26.4)**: Structured data manipulation, cleaning, and mathematical operations.
- **Joblib (v1.3.2)**: Efficient serialization and deserialization of the trained models and scaler pipelines.

### Data Visualization
- **Matplotlib (v3.8.4)** & **Seaborn (v0.13.2)**: Static and statistical graphing engines.

### Web Server & Interface
- **Flask (v2.3.3)**: Lightweight WSGI micro-web framework.
- **Gunicorn (v23.0.0)**: Production-grade WSGI HTTP Server.
- **HTML5 & CSS3**: Responsive interfaces with modern Outfit & Inter typography.

### Testing & Infrastructure
- **Apache JMeter**: Heavy-load verification and performance auditing.

---

## 📸 Project Screenshots

Visual walk-through of the interface, predictive calculations, and pipeline outputs:

### 1. Web Application Interface
!<img width="1920" height="874" alt="image" src="https://github.com/user-attachments/assets/fa04c1b4-3767-47da-b691-f02f9875d05f" />
](screenshots/home-page.png)

### 2. Flood Prediction Result
!<img width="1920" height="861" alt="image" src="https://github.com/user-attachments/assets/101b691b-58ba-44ae-98c6-fe7005c873ab" />
](screenshots/prediction-result.png)

### 3. Model Visualization
!<img width="1920" height="868" alt="image" src="https://github.com/user-attachments/assets/fe81c3f6-2b5e-48f0-bac5-b88b0f9a18cb" />
](screenshots/model-output.png)

---

## 🎥 Live Demo

A comprehensive walk-through demonstrating the real-time predictive capability and operational pipeline:

[Deploy link](https://rising-waters-hh79.onrender.com/)

[Demo Video](https://drive.google.com/file/d/15d2GuuEyP912i229FnlPPryshDNQVJhJ/view?usp=drivesdk)

The demo showcases the complete workflow:
* **User Input**: Inputting local meteorological and seasonal rainfall metrics.
* **Flood Prediction Process**: Real-time feature scaling and inference processing through the champion XGBoost Classifier.
* **Prediction Result**: Displays the warning or safe forecast screen along with calculated prediction confidence rates and pre-emptive advisories.
* **Visualization Output**: View of generated exploratory charts (distribution plots, boxplots, correlation heatmap) and model comparisons.

---

## 📂 Folder Structure

```text
Rising-Waters/
├── 1.Brainstorming & Ideation/          # Phase 1: Problem Definition & Ideation Docs
│   ├── Brainstorming & Idea Prioritization (1) (1).pdf
│   ├── Define Problem Statements.pdf
│   └── Empathy Map.pdf
├── 2. Requirement Analysis/             # Phase 2: User Journeys & Tech Stack Spec
│   ├── Customer Journey Map.pdf
│   ├── Data Flow Diagram.pdf
│   ├── Solution Requirements.pdf
│   └── Technology Stack.pdf
├── 3. Project Design Phase/              # Phase 3: Architecture & System Design
│   ├── Problem - Solution Fit.pdf
│   ├── Proposed Solution.pdf
│   └── Solution Architecture.pdf
├── 4. Project Planning Phase/            # Phase 4: Planning & Schedule Docs
│   └── Project Planning.pdf
├── 5. Project Development Phase/         # Phase 5: Implementation reports & layout
│   ├── Code-Layout, Readability and Reusability.pdf
│   ├── Coding & Solution.pdf
│   └── No. of Functional Features Included in the Solution.pdf
├── 6.Project Testing/                   # Phase 6: Quality Assurance & Performance Reports
│   └── Performance Testing.pdf
├── 7.Project Documentation/             # Phase 7: Installation & Executable manuals
│   ├── Project Executable Files.pdf
│   └── Sample Project Documentation.pdf
├── 8.Project Demonstration/             # Phase 8: Demo planning, scalability & communication
│   ├── Communication.pdf
│   ├── Demonstration of Proposed Features.pdf
│   ├── Project Demo Planning.pdf
│   ├── Scalability & Future Plan.pdf
│   └── Team Involvement in Demonstration.pdf
├── demo/                                # Live demo video assets
│   └── rising-waters-demo.mp4
├── screenshots/                         # Portfolio application screenshots
│   ├── home-page.png
│   ├── model-output.png
│   └── prediction-result.png
└── Rising_Waters_Project_Files/         # Core codebase & project resources
    ├── app.py                           # Flask web application entry point
    ├── requirements.txt                 # Project dependencies
    ├── dataset/                         # Dataset folder
    │   └── flood_data.csv               # Historical weather & flood training data
    ├── outputs/                         # Pipeline output assets
    │   └── plots/                       # Generated evaluation plots and charts
    │       ├── boxplots_by_floods.png
    │       ├── class_distribution.png
    │       ├── correlation_heatmap.png
    │       ├── feature_importance.png
    │       ├── model_comparison.png
    │       ├── pairplot.png
    │       ├── scatter_rainfall_cloud.png
    │       └── univariate_distributions.png
    ├── models/                          # Serialized model artifacts
    │   ├── floods.save                  # Saved Champion XGBoost model
    │   └── scaler.pkl                   # Saved StandardScaler object
    ├── notebooks/                       # Modular scripts for training pipeline
    │   ├── 01_data_loading.py           # Dataset loading & initial checks
    │   ├── 02_visualization.py          # EDA & generation of charts
    │   ├── 03_preprocessing.py          # Data cleaning, outlier capping & scaling
    │   └── 04_model_training.py         # Complete model comparison & saving script
    ├── static/                          # Static assets for web UI
    │   └── css/
    │       └── style.css                # Custom glassmorphism stylesheet
    └── templates/                       # HTML template pages
        ├── index.html                   # Landing homepage
        ├── input.html                   # Prediction input form
        ├── result_flood.html            # Results page showing flood risk
        └── result_no_flood.html         # Results page showing safe forecast
```

---

## ⚙️ Installation

Set up the environment and install dependencies locally by following these steps:

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10** or higher
- **pip** (Python package installer)
- **Git**

### 2. Clone the Repository
Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/Rising-Waters.git
cd Rising-Waters/Rising_Waters_Project_Files
```

### 3. Initialize a Virtual Environment
Create a clean environment to manage dependencies:
```bash
# Create the environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Activate the environment (macOS/Linux)
source venv/bin/activate
```

### 4. Install Project Packages
Install the required packages listed in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Retraining & Saving the Models (Optional)
The project comes with pre-trained models. However, if you want to rerun the full pipeline, execute:
```bash
python notebooks/04_model_training.py
```
This script will:
- Clean and clip the training data.
- Train the KNN, Decision Tree, Random Forest, and XGBoost classifiers.
- Save comparison metrics and model plots to `outputs/plots/`.
- Save the champion XGBoost classifier (`floods.save`) and `scaler.pkl` to the `models/` directory.

### Generating Visualizations
To produce and view the Exploratory Data Analysis graphs independently:
```bash
python notebooks/02_visualization.py
```

### Starting the Web Server
Launch the Flask development server:
```bash
python app.py
```
After the server initializes, open your browser and navigate to:
```
https://rising-waters-hh79.onrender.com/
```

---

## 🧠 Model Workflow

The machine learning workflow maps data from source files to real-time predictions:

```mermaid
graph TD
    A[Raw Weather Data: flood_data.csv] --> B[Data Preprocessing]
    B --> B1[Median/Mode Imputation]
    B --> B2[IQR Outlier Capping]
    B --> B3[Select 6 Primary Inputs]
    B3 --> C[Fit & Save StandardScaler]
    C --> D[Data Split: 80% Train / 20% Test]
    D --> E[Multi-Model Training]
    E --> E1[K-Nearest Neighbors: ~92.92%]
    E --> E2[Decision Tree: ~90.42%]
    E --> E3[Random Forest: ~94.58%]
    E --> E4[XGBoost Classifier: ~95.42%]
    E4 --> F[Select Best Model: XGBoost]
    F --> G[Serialize Artifacts: floods.save & scaler.pkl]
    G --> H[Flask Server Loads Artifacts]
    H --> I[User Inputs Features via Web App]
    I --> J[Scaler Transform & Model Predict]
    J --> K[Real-Time Risk Output Shown to User]
```

### Key Prediction Parameters
The model leverages six major inputs for forecasting:
1. **Annual Rainfall**: Total cumulative regional rainfall (mm).
2. **Cloud Coverage**: Average atmospheric cloud coverage percentage (%).
3. **JUN-SEP Rainfall**: Total monsoon-period rainfall (mm).
4. **MAR-MAY Rainfall**: Total pre-monsoon-period rainfall (mm).
5. **OCT-DEC Rainfall**: Total post-monsoon-period rainfall (mm).
6. **JAN-FEB Rainfall**: Total winter-period rainfall (mm).

---

## 🌐 Deployment

For cloud environments and scaling, follow these guidelines:

### Production Web Server
While the Flask development server is excellent for local use, it is not optimized for production. It is recommended to deploy using **Gunicorn** to handle concurrent requests:
```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

### Cloud Platforms
The system can be deployed directly to cloud services such as **Render**, **Heroku**, or **AWS Elastic Beanstalk**:
- Set the build command to: `pip install -r requirements.txt`
- Set the start command to: `gunicorn app:app`
- Ensure the `PORT` environment variable is exposed so the application binds dynamically to the host port.

---

## 🔮 Future Scope

- **Real-Time API Integration**: Connect with live services like OpenWeatherMap to query current weather variables automatically based on GPS location.
- **GIS Heatmap Overlay**: Develop interactive map interfaces that color code regional risk levels.
- **Mobile Companion App**: Deploy mobile clients featuring push notifications and hazard alarms.
- **User Management**: Add authentication to allow users to save past predictions and monitor custom zones.
- **Spatio-Temporal Models**: Implement Deep Learning LSTM models to predict flood timelines based on time-series records.

---

## ✍️ Author

**Pravalika Simma** (Lead Developer)  
Created with a focus on leveraging data science and machine learning for disaster mitigation, early warning systems, and community safety.

For support, feedback, or collaborations, feel free to raise an issue or reach out through the project repository.
