# Data Science & Time Series Projects - Master's in Applied Artificial Intelligence

This repository contains data science and time series analysis projects completed as part of my Master's program in Applied Artificial Intelligence. Each project demonstrates different aspects of data science methodology, from exploratory data analysis to advanced deep learning techniques for temporal data.

## Projects Overview

## Data Science Projects

### Unit 1: Data Manipulation and Analysis Fundamentals
**Files:** `solucion_enunciado_iep_iaa_ds_u1.py`

#### Project 1: Movie Ratings Analysis
- **Dataset:** MovieLens dataset (movies.csv, ratings.csv)
- **Objective:** Analyze movie ratings and user preferences
- **Key Tasks:**
  - Data loading and merging of multiple datasets
  - Statistical analysis of movie ratings
  - Identification of top-rated movies with statistical significance
  - Time series analysis of rating patterns
  - Genre-based filtering and analysis

**Key Findings:**
- Analyzed 45,447 movies with 500,000 ratings
- "Shawshank Redemption" ranked as highest-rated movie (4.42/5) with sufficient sample size
- Rating period spanned from 1996 to 2017

#### Project 2: Student Performance Analysis
- **Dataset:** UCI Student Performance dataset
- **Objective:** Analyze factors affecting student academic performance
- **Key Tasks:**
  - Data preprocessing and cleaning
  - Student performance tracking across multiple terms
  - Demographic analysis (age, gender effects)
  - Pass/fail rate calculations
  - Duplicate detection and handling

**Key Findings:**
- 167 students passed all three courses, 61 failed all courses
- 15-year-old male students showed highest average performance (12.57/20)
- Identified data quality issues with 15 duplicate records

---

## Time Series Analysis Projects

### Unit 1: Time Series Fundamentals and Exploration
**Files:** `solucion_iep_iaa_ts_caso_practico_u1.py`

#### AirPassengers Time Series Analysis
- **Dataset:** Classic AirPassengers dataset (monthly airline passenger numbers)
- **Objective:** Master fundamental time series analysis techniques
- **Key Tasks:**
  - Time series visualization and trend identification
  - Seasonal decomposition (multiplicative model)
  - Autocorrelation analysis and pattern detection
  - Monthly and yearly seasonality exploration

**Key Findings:**
- Clear increasing trend over the time period
- Strong seasonal patterns with consistent annual peaks
- Multiplicative relationship between trend and seasonality
- High autocorrelation confirming temporal dependencies

**Technical Skills:**
- `statsmodels` seasonal decomposition
- Autocorrelation plotting and interpretation
- Time series data preprocessing and visualization

---

### Unit 3: Deep Learning for Time Series Forecasting
**Files:** `solucion_caso_practico_iep_iaa_ts_u3.py`

#### Energy Consumption Forecasting with Neural Networks
- **Dataset:** PJME hourly energy consumption data
- **Objective:** Compare deep learning architectures for time series prediction
- **Models Implemented:**
  - Recurrent Neural Networks (RNN)
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Units (GRU)

**Advanced Preprocessing:**
- Time delay embedding with 24-hour lag windows
- Calendar feature engineering (hour, day, month, day of week)
- Holiday detection and encoding
- MinMax scaling for neural network optimization
- PyTorch tensor preparation and DataLoader implementation

**Key Results:**
- LSTM achieved superior performance: R² ≈ 0.974, RMSE ≈ 966, MAE ≈ 645
- Demonstrated effectiveness of combining PCA preprocessing with t-SNE
- Hybrid approach (PCA + LSTM) improved computational efficiency
- Calendar features significantly enhanced prediction accuracy

**Technical Contributions:**
- Custom PyTorch model architectures
- Comprehensive optimization class for training and evaluation
- Performance benchmarking across multiple architectures
- Advanced feature engineering for temporal data

---

### Applied Project: Disease Surveillance Forecasting
**Files:** `solución_caso_práctico_iep_iaa_ts_pa.py`

#### Flu Cases Prediction in Burgundy Region
- **Dataset:** Weekly flu cases in Burgundy, France (2004-2014)
- **Objective:** Develop ML vs DL forecasting pipeline for epidemiological surveillance
- **Comprehensive Analysis Pipeline:**
  - Stationarity testing using Augmented Dickey-Fuller (ADF) test
  - Seasonal decomposition and pattern identification
  - ACF/PACF analysis for model selection guidance
  - Feature engineering with 12-week lag windows

**Model Comparison:**
- **Machine Learning:** Random Forest Regressor
- **Deep Learning:** LSTM Neural Network

**Results Comparison:**

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | 0.0335 | 0.0667 | 0.4720 |
| LSTM | 0.0284 | 0.0616 | 0.5491 |

**Key Insights:**
- LSTM outperformed Random Forest across all metrics
- Strong seasonal patterns confirmed epidemiological expectations
- Non-stationary nature of the series validated use of modern ML approaches
- 12-week lag window effectively captured seasonal dependencies

**Business Impact:**
- Demonstrated applicability for public health surveillance systems
- Provided foundation for early warning systems in disease outbreaks
- Validated deep learning approaches for epidemiological forecasting

---

### Unit 2: Exploratory Data Analysis (EDA)
**Files:** `solucion_iaa_ds_u2.py`

#### Diabetes Hospital Readmission Analysis
- **Dataset:** Diabetes Hospital dataset from Fairlearn
- **Objective:** Predict 30-day hospital readmission risk for diabetes patients
- **Methodology:**
  - Comprehensive univariate analysis of categorical and numerical variables
  - Bivariate analysis including correlation studies
  - Chi-square tests for categorical variable significance
  - Statistical significance testing for readmission predictors

**Key Techniques:**
- Missing value analysis and treatment
- Categorical variable encoding and cleaning
- Correlation matrix analysis
- Statistical hypothesis testing (Chi-square)
- Outlier detection using boxplots and histograms

**Key Findings:**
- Most categorical variables showed significant relationship with readmission
- Variable `medicaid` showed no statistical significance and was recommended for removal
- No high multicollinearity detected between numerical features

---

### Unit 3: Dimensionality Reduction Techniques
**Files:** `solucion_iep_iaa_ds_u3.py`

#### Comparative Analysis: PCA vs t-SNE
- **Datasets:** Iris dataset, MNIST handwritten digits
- **Objective:** Compare and optimize dimensionality reduction techniques
- **Methods Implemented:**
  - Principal Component Analysis (PCA)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Hybrid approach: PCA + t-SNE optimization

**Technical Contributions:**
- Demonstrated PCA effectiveness on linear data (Iris)
- Showed t-SNE superiority for non-linear data (MNIST)
- Developed optimization strategy combining PCA preprocessing with t-SNE
- Performance benchmarking and computational complexity analysis

**Key Insights:**
- PCA explained variance analysis for optimal component selection
- t-SNE better preserved local structure in high-dimensional data
- Hybrid approach improved computational efficiency without quality loss

---

### Applied Project: Fraud Detection System
**Files:** `solución_iep_iaa_ds_pa.py`

#### Banking Fraud Detection
- **Dataset:** Custom banking transaction dataset
- **Objective:** Develop predictive model for fraudulent banking activities
- **Pipeline Implemented:**
  - Data preprocessing and quality assessment
  - Feature engineering and selection
  - Correlation analysis and multicollinearity removal
  - Dimensionality reduction using PCA
  - Model preparation for fraud classification

**Technical Implementation:**
- Automated duplicate and null value handling
- Statistical correlation analysis for feature selection
- PCA implementation with 90% variance retention
- Data standardization and scaling
- Prepared datasets for machine learning model training

**Business Impact:**
- Created clean, analysis-ready dataset for fraud detection
- Reduced feature space while maintaining information content
- Established foundation for real-world fraud prevention systems

---

## Technical Skills Demonstrated

### Programming & Libraries
- **Python:** pandas, numpy, matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Deep Learning:** PyTorch, TensorFlow/Keras
- **Time Series:** statsmodels, seasonal_decompose
- **Statistical Analysis:** scipy, ADF testing

### Data Science Methodologies
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Dimensionality reduction techniques
- Data preprocessing and cleaning
- Feature engineering and selection

### Time Series Analysis Techniques
- Seasonal decomposition (additive/multiplicative)
- Stationarity testing (Augmented Dickey-Fuller)
- Autocorrelation and Partial Autocorrelation analysis
- Time delay embedding and lag feature creation
- Calendar feature engineering

### Advanced Modeling Techniques
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Recurrent Neural Networks (RNN, LSTM, GRU)
- Ensemble methods (Random Forest)
- Neural network optimization and regularization

---

## Project Structure

Each project follows a structured approach:

1. **Data Loading & Exploration:** Initial dataset examination
2. **Data Preprocessing:** Cleaning, transformation, and preparation
3. **Exploratory Analysis:** Statistical and visual analysis
4. **Method Implementation:** Application of specific techniques
5. **Results Interpretation:** Analysis of findings and implications
6. **Documentation:** Comprehensive code comments and explanations

---

## Academic Context

These projects were completed as part of the Master's program in Applied Artificial Intelligence, demonstrating progression across two key domains:

**Data Science Track:** Evolution from fundamental data manipulation to advanced machine learning preprocessing techniques, covering exploratory data analysis, statistical testing, and dimensionality reduction methods.

**Time Series Analysis Track:** Progression from basic temporal pattern recognition to sophisticated deep learning architectures for forecasting, including classical statistical methods and modern neural network approaches.

Each project builds upon previous knowledge while introducing new concepts and methodologies essential for modern data science and time series analysis practice. The combination of both tracks provides a comprehensive foundation for tackling real-world problems involving both static and temporal data structures.

---

## Repository Contents

```
Data Science Projects:
├── solucion_enunciado_iep_iaa_ds_u1.py    # Unit 1: Data Fundamentals
├── solucion_iaa_ds_u2.py                  # Unit 2: Exploratory Data Analysis  
├── solucion_iep_iaa_ds_u3.py              # Unit 3: Dimensionality Reduction
├── solución_iep_iaa_ds_pa.py              # Applied Project: Fraud Detection

Time Series Projects:
├── solucion_iep_iaa_ts_caso_practico_u1.py    # Unit 1: Time Series Fundamentals
├── solucion_caso_practico_iep_iaa_ts_u3.py    # Unit 3: Deep Learning for TS
├── solución_caso_práctico_iep_iaa_ts_pa.py    # Applied Project: Disease Forecasting
└── README.md                                   # This file
```

---

## Getting Started

To run these projects:

1. Clone the repository
2. Install required dependencies:
   ```bash
   # Core data science libraries
   pip install pandas numpy matplotlib seaborn scikit-learn scipy fairlearn
   
   # Time series specific libraries
   pip install statsmodels plotly
   
   # Deep learning libraries
   pip install torch tensorflow keras
   
   # Additional utilities
   pip install holidays
   ```
3. Run individual Python files in your preferred environment (Jupyter, Google Colab, etc.)

---

## License

This project is part of academic coursework for educational purposes.

---

## Contact

For questions about these projects or collaboration opportunities, please feel free to reach out through GitHub.
