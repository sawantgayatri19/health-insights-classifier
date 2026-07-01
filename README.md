# Health Insights Classifier

**An end-to-end machine learning system for health risk prediction with LLM-powered explanations**

## Overview

Health Insights Classifier is a production-ready ML pipeline that predicts health risk levels from structured health intake data. The system achieves **86% accuracy** while providing LLM-powered personalized explanations.

**Key Features:**
- 🎯 **86% Accuracy** - RandomForestClassifier with optimized features
- 🤖 **LLM Integration** - OpenAI API with prompt engineering
- 📊 **RAG System** - Evidence-backed recommendations
- 🔄 **ETL Pipeline** - Robust data preprocessing
- 🧪 **Test-Driven Development** - pytest coverage
- 🚀 **Production Ready** - Flask REST API

## Tech Stack

- **ML Framework:** scikit-learn, pandas, numpy
- **LLM Integration:** OpenAI API, prompt engineering, RAG
- **Backend API:** Flask, REST APIs
- **Testing:** pytest, TDD
- **Data Storage:** CSV, serialized models (joblib)

## Project Structure

health-insights-classifier/
├── data_pipeline.py          # ETL and data preprocessing
├── model.py                  # ML model training (86% accuracy)
├── llm_explainer.py          # OpenAI API + RAG integration
├── app.py                    # Flask REST API
├── requirements.txt          # Python dependencies
├── models/
│   └── classifier.pkl        # Trained RandomForestClassifier
├── data/
│   └── synthetic_health_data.csv
└── README.md
## Installation

### Prerequisites
- Python 3.14+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/sawantgayatri19/health-insights-classifier.git
cd health-insights-classifier

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### 1. Data Pipeline & Model Training

```python
from data_pipeline import HealthDataPipeline
from model import train_model, evaluate_model

# Initialize and prepare data
pipeline = HealthDataPipeline()
X_train, X_test, y_train, y_test = pipeline.prepare_data()

# Train model
model = train_model(X_train, y_train)

# Evaluate
accuracy = evaluate_model(model, X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")  # 86%
```

### 2. LLM-Powered Explanations

```python
from llm_explainer import HealthExplainer

explainer = HealthExplainer(model)

# Get prediction with explanation
result = explainer.predict_with_explanation(user_data)
print(result['explanation'])
print(result['recommendations'])
```

### 3. REST API

```bash
python app.py
# API runs on http://localhost:5000
```

**API Endpoints:**
- `POST /predict` - Health risk prediction
- `POST /explain` - LLM-powered explanation
- `POST /predict-and-explain` - Combined prediction + explanation
- `GET /health` - Service health check
- `GET /features` - Required feature names

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 86% |
| **Precision** | 0.84 |
| **Recall** | 0.85 |
| **F1-Score** | 0.845 |

**Model:** RandomForestClassifier (100 estimators)  
**Features:** 15 engineered health attributes  
**Data:** Synthetic health intake data (500+ samples)

## Key Components

### data_pipeline.py
- HealthDataPipeline class for ETL
- Synthetic data generation
- Feature engineering and scaling
- Train/test split with stratification

### model.py
- RandomForestClassifier training
- Cross-validation and hyperparameter tuning
- Model serialization (joblib)
- Feature importance analysis

### llm_explainer.py
- OpenAI API integration with error handling
- Retrieval-Augmented Generation (RAG) system
- Risk level categorization
- Fallback mechanisms for API failures

### app.py
- Flask REST API with 5 endpoints
- Input validation and error handling
- Comprehensive logging
- CORS support for frontend integration

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_model.py -v
```

## Architecture
User Input (Health Data)
↓
Data Pipeline (Preprocessing)
↓
ML Model (RandomForest - 86%)
↓
Prediction + Probability
↓
LLM Explainer (OpenAI + RAG)
↓
Natural Language Explanation
↓
REST API (Flask) → User Response
## Features in Development

- [x] LLM Integration - Full OpenAI implementation
- [x] Flask REST API - Production endpoints
- [ ] Database - PostgreSQL integration
- [ ] Deployment - Docker containerization
- [ ] Advanced RAG - Vector embeddings
- [ ] Web Dashboard - Frontend interface

## Future Roadmap

**Phase 2 (Months 1-3):**
- Cloud deployment (AWS/GCP)
- User authentication
- Database persistence

**Phase 3 (Months 3-6):**
- Advanced RAG with embeddings
- Model monitoring and drift detection
- User feedback loops

**Phase 4 (Months 6+):**
- Personalized insights
- Healthcare provider integration
- Mobile app

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open Pull Request


## Author

**Gayatri Sawant**
- Email: sawant.g@northeastern.edu
- GitHub: [@sawantgayatri19](https://github.com/sawantgayatri19)
- LinkedIn: [Gayatri Sawant](https://linkedin.com/in/sawantgayatri)

## Contact & Support

- GitHub Issues: [Open an issue](https://github.com/sawantgayatri19/health-insights-classifier/issues)
- Email: sawant.g@northeastern.edu

---

**Last Updated:** July 1, 2026  
**Status:** Active Development (Phase 1 ✓ Complete)
