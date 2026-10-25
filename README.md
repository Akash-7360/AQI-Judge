# 🍃 Air Quality Index (AQI) Machine Learning Dashboard

An interactive Machine Learning dashboard and predictive analytics system for Indian Air Quality Index (AQI) monitoring and category classification based on official Central Pollution Control Board (CPCB) standards.

---

## 📁 Project Structure

```text
├── app.py                                              # Streamlit interactive web dashboard
├── model.joblib                                        # Serialized pre-trained classification models & evaluation cache
├── aqi_model.pkl                                       # Serialized Random Forest AQI regressor
├── requirements.txt                                    # Python dependencies
├── data/
│   └── air-quality-india.csv                          # Historical Indian air quality records (74,640 rows)
├── train.py                                            # Standalone model training & evaluation pipeline
├── assets/                                             # Dashboard visual themes & backgrounds
│   ├── air_dark_bg.jpg
│   └── air_light_bg.jpg
├── .streamlit/                                         # Streamlit server and layout configuration
│   └── config.toml
├── Air_Quality_Classification_System_Documentation.pdf # Technical system documentation
└── README.md                                           # Project overview & instructions
```


---

## ⚙️ Features & Architecture

- **Machine Learning Models**:
  - 🌲 **Decision Tree Classifier (Champion Model)**: Fast, interpretable, high-accuracy category classification.
  - 👥 **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classification.
  - 📉 **Gaussian Naive Bayes**: Probabilistic classifier with class likelihood estimation.
  - 🌳 **Random Forest Regressor** (`train.py`): Time-series lag-based AQI numeric prediction with $R^2 \approx 0.98$.
- **CPCB NAQI Standards**:
  - Categorization into: *Good* (0–30), *Satisfactory* (31–60), *Moderate* (61–90), *Poor* (91–120), *Very Poor* (121–250), and *Severe* (250+ $\mu g/m^3$).
- **Modern UI & Visualization**:
  - Built with Streamlit and Plotly.
  - Atmospheric dark and light theme toggles.
  - Real-time parameter tuning and health precaution advisories.
  - Model performance comparison, confusion matrices, and detailed classification reports.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested up to Python 3.14)
- Virtual environment (recommended)

### 2. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Run Training Pipeline
To retrain models from scratch using the dataset in `data/air-quality-india.csv`:
```bash
python train.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Dataset Details

The dataset located in `data/air-quality-india.csv` contains hourly records across Indian monitoring stations:
- **Timestamp**: Date and time of observation
- **PM2.5**: Fine particulate matter concentration in $\mu g/m^3$
- **Hour / Month / Day / Year**: Temporal features for seasonal analysis
