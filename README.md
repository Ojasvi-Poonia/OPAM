# OPAM - Expense Prediction System

A comprehensive AI-powered expense prediction and analysis platform using ensemble machine learning for financial forecasting and Management.

## Overview

OPAM (Optimized Prediction and Analysis of Monthly Expenses) is a full-stack expense prediction system that analyzes transaction data to forecast future spending patterns with 98%+ accuracy. The system includes fraud detection, anomaly detection, user clustering, and budget recommendations.

## Features

### Machine Learning Models

- 6 ensemble ML models: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, and Ensemble
- Time series analysis with lag features and rolling windows
- 50+ engineered features from raw transaction data
- Category-wise spending predictions
- Statistical confidence intervals

### Analytics & Insights

- Spending pattern analysis (daily, weekly, monthly)
- Category and merchant intelligence
- Anomaly detection using multiple algorithms
- User behavior clustering and segmentation
- Budget optimization with AI-powered recommendations

### Security & Fraud Detection

- ML-based fraud scoring (0-100 scale)
- Real-time risk assessment
- Pattern recognition for suspicious transactions
- Multi-level risk classification

### Dashboard

- Interactive Streamlit web interface
- Real-time data visualization with Plotly
- Multiple analysis modules
- Export capabilities (CSV, Excel, PDF)

## Project Structure

```
opam/
├── back/                          # Backend and data generation
│   ├── generate_sample_data.py    # Sample data generator
│   ├── fix_data.py               # Data cleaning utilities
│   └── requirements.txt          # Backend dependencies
├── model/                         # ML models and analysis
│   ├── expense_predictor.py      # Main prediction model
│   ├── budget_recommender.py     # Budget optimization
│   ├── fraud_detector.py         # Fraud detection
│   ├── anomaly_detector_simple.py # Anomaly detection
│   ├── user_clusterer.py         # User segmentation
│   ├── run_all_systems.py        # Master execution script
│   └── visualize_*.py            # Visualization modules
├── data/                         # Data storage
│   └── transactions.csv          # Transaction data
├── results/                      # Model outputs
│   ├── predictions.csv
│   ├── fraud_scores.csv
│   ├── anomalies.csv
│   └── budget_recommendations.csv
├── charts/                       # Generated visualizations
├── opam_dashboard.py            # Main dashboard
├── advanced_features.py         # Advanced dashboard features
├── requirements.txt             # Full dependencies
├── requirements-minimal.txt     # Core dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.11 or 3.12 (recommended)
- pip package manager
- Virtual environment support

> **⚠️ Important**: This project requires Python 3.11 or 3.12. Python 3.14 is not currently supported due to compatibility issues with PyArrow (a dependency of Streamlit). Pre-built wheels for PyArrow are not yet available for Python 3.14, which causes installation failures.

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate virtual environment:

```bash
# Create virtual environment with Python 3.12 (recommended)
python3.12 -m venv .venv

# Or use Python 3.11
python3.11 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

3. Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Troubleshooting Installation

**Issue: PyArrow build failure**

If you encounter an error like `ERROR: Failed building wheel for pyarrow`, ensure you're using Python 3.11 or 3.12:

```bash
# Check your Python version
python --version

# If using Python 3.14 or newer, recreate your virtual environment:
deactivate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative Solution (macOS with Homebrew):**

If you must use a newer Python version, install Apache Arrow C++ library first:

```bash
brew install apache-arrow
export ARROW_HOME=$(brew --prefix apache-arrow)
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run opam_dashboard.py
```

Open browser to `http://localhost:8501`

## Usage

### Generate Sample Data

```bash
cd back
python3 generate_sample_data.py
cd ..
```

### Train ML Models

```bash
cd model
python3 expense_predictor.py
cd ..
```

### Run Dashboard

```bash
streamlit run opam_dashboard.py
```

Open browser to `http://localhost:8501`

### Run All Systems

```bash
cd model
python3 run_all_systems.py
cd ..
```

This executes all modules: prediction, fraud detection, anomaly detection, clustering, budget recommendations, and visualizations.

## ML Model Performance

| Model             | RMSE     | R² Score | MAPE   | Training Time |
| ----------------- | -------- | -------- | ------ | ------------- |
| Linear Regression | ₹81,949  | 0.9792   | 1.49%  | <1s           |
| Ridge Regression  | ₹77,891  | 0.9812   | 1.57%  | <1s           |
| Random Forest     | ₹303,232 | 0.7155   | 6.31%  | ~10s          |
| Gradient Boosting | ₹83,542  | 0.9784   | 1.68%  | ~15s          |
| XGBoost           | ₹550,783 | 0.0614   | 11.52% | ~20s          |
| Ensemble          | ₹147,532 | 0.9327   | 2.55%  | ~30s          |

Best Model: Ridge Regression (lowest RMSE)

Production Model: Ensemble (balanced performance)

## Key Components

### Expense Predictor

Trains 6 ML models on historical transaction data to predict future monthly expenses with feature engineering, lag features, and rolling statistics.

### Fraud Detector

Calculates fraud risk scores (0-100) for each transaction using multiple detection methods including time-based analysis, amount-based scoring, and pattern recognition.

### Anomaly Detector

Identifies unusual transactions using statistical methods (Z-score), Isolation Forest, and high-value detection algorithms.

### Budget Recommender

Generates personalized budget recommendations by category with savings opportunities and spending optimization strategies.

### User Clusterer

Segments users into behavioral groups using K-Means clustering based on spending patterns, transaction frequency, and category preferences.

## Dashboard Pages

1. Overview - Key metrics and transaction summary
2. Predictions - ML model forecasts and confidence intervals
3. Budget - AI-powered budget recommendations
4. Fraud Detection - Risk scores and suspicious transactions
5. Anomalies - Unusual transaction patterns
6. User Segments - Behavioral clustering results
7. Analytics - Advanced spending analysis

## Configuration

### Data Format

The system expects CSV files with the following columns:

- date: Transaction date (YYYY-MM-DD HH:MM:SS)
- amount: Transaction amount (numeric)
- category: Spending category (string)
- merchant: Merchant name (string)
- description: Transaction description (string)
- payment_method: Payment type (string)
- is_recurring: Recurring transaction flag (0/1)

### Model Parameters

Edit model parameters in respective Python files:

- Number of estimators for tree-based models
- Learning rate for gradient boosting
- Number of clusters for user segmentation
- Fraud detection thresholds

## Dependencies

### Core Requirements

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning models
- xgboost: Gradient boosting
- streamlit: Dashboard framework
- plotly: Interactive visualizations

### Optional Requirements

- tensorflow: Deep learning (LSTM models)
- lightgbm: Additional ML models
- openpyxl: Excel export
- reportlab: PDF generation

See requirements.txt for complete list.

## Performance

- Data Loading: ~2s for 1.3M transactions
- Feature Engineering: ~5s
- Model Training: ~30s (all 6 models)
- Prediction: <0.1s
- Dashboard Load: ~3s

Tested with 5M transactions, ~2GB RAM usage, 50K transactions/second processing speed.

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=model --cov-report=html
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

Built with:

- Streamlit - Dashboard framework
- scikit-learn - Machine learning
- XGBoost - Gradient boosting
- Plotly - Interactive visualizations
- Pandas - Data manipulation

## Contact

For questions or support, please open an issue on GitHub.

Refered database file is https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset
