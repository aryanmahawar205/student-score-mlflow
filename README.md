# 🎓 Student Score Prediction — End-to-End ML Pipeline with MLflow

An **end-to-end production-ready data science pipeline** for predicting student math scores using structured academic and demographic data.  
The project follows **industry best practices** with modular code, automated setup, experiment tracking, and model registry using **MLflow + Dagshub**.

---

## 🚀 Key Highlights

- 📦 **Fully modular ML pipeline** (data ingestion → transformation → training → evaluation)
- 🧱 **Package-based architecture** with reusable components
- ⚙️ **Automated project scaffolding** using `template.py`
- 🗄️ **PostgreSQL (Supabase) data ingestion**
- 📊 **Multi-model training & hyperparameter tuning**
- 📈 **Experiment tracking & model registry** with MLflow + Dagshub
- 🪵 Centralized **logging & custom exception handling**
- 💾 Persisted artifacts (preprocessor & trained model)

---

## 🏗️ Project Architecture
```
student-score-mlflow/
│
├── src/student_score_mlflow/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipelines/
│ │ ├── training_pipeline.py
│ │ └── prediction_pipeline.py
│ │
│ ├── exception.py
│ ├── logger.py
│ └── utils.py
│
├── artifacts/
│ ├── train.csv
│ ├── test.csv
│ ├── preprocessor.pkl
│ └── model.pkl
│
├── app.py
├── template.py
├── setup.py
├── requirements.txt
└── Dockerfile
```


---

## 🔄 Pipeline Overview

### 1️⃣ Data Ingestion
- Reads student data from **PostgreSQL (Supabase)** or local CSV
- Splits dataset into train & test sets
- Stores raw and processed data as versioned artifacts

### 2️⃣ Data Transformation
- Handles missing values using imputers
- Applies scaling to numerical features
- Encodes categorical features using One-Hot Encoding
- Saves the preprocessing pipeline for reuse

### 3️⃣ Model Training & Evaluation
- Trains multiple regression models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - CatBoost
  - AdaBoost
- Performs **GridSearchCV** for hyperparameter tuning
- Selects best model based on **R² score**

### 4️⃣ Experiment Tracking (MLflow + Dagshub)
- Logs:
  - Model parameters
  - RMSE, MAE, R² metrics
  - Trained models
- Registers best model to **Dagshub MLflow Registry**

---

## 🧠 Models Evaluated

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- CatBoost Regressor  
- AdaBoost Regressor  

---

## ⚙️ Tech Stack

**Core**
- Python, NumPy, Pandas
- Scikit-learn

**MLOps**
- MLflow
- Dagshub
- DVC

**Data**
- PostgreSQL (Supabase)

**Utilities**
- Logging
- Custom Exception Handling
- setup.py based packaging

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```
git clone https://github.com/aryanmahawar205/student-score-mlflow.git
cd student-score-mlflow
```

### 2️⃣ Create Virtual Environment
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies
```
pip install -requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a .env file:
```
DB_HOST=your_host
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_db
DB_PORT=5432
```

### 5️⃣ Run Training Pipeline
```
python app.py
```

📊 MLflow Tracking

Experiments and models are tracked using Dagshub MLflow:
```
https://dagshub.com/aryanmahawar205/student-score-mlflow.mlflow
```

You can visualize:

- Experiment runs
- Metrics comparison
- Model artifacts
- Registered models

🧪 Best Model Selection

- Automatically selects the best performing model based on R² score
- Saves the trained model to artifacts/model.pkl
- Saves preprocessing pipeline to artifacts/preprocessor.pkl

📌 Future Improvements

- CI/CD integration
- Model inference API
- Data validation with Great Expectations
- Monitoring & drift detection

👤 Author

Aryan Mahawar
📧 aryanmahawar205@gmail.com
