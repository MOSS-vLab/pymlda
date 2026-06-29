# PyMLDA — Machine Learning for Damage Assessment

PyMLDA is a modular machine learning framework for **structural health monitoring (SHM)** and **damage assessment** using clustering, classification, and regression models.

The framework integrates signal-derived features, unsupervised learning (KMeans), and supervised models (SVM, Random Forest, KNN, XGBoost, SVR) into a unified pipeline.

---

## 🧠 System Overview

PyMLDA follows a modular architecture:

- Feature extraction (external modules)
- Signal processing (external modules)
- ML Pipeline (clustering + classification + regression)
- Evaluation and visualization tools

---

## ⚙️ Pipeline Architecture

The framework is built around a unified pipeline:


---

## 🚀 Main Components

### 📦 Pipeline
- `PyMLDAPipeline`: central orchestration class

### 📦 Models
- SVM
- Random Forest
- KNN
- Naive Bayes
- XGBoost
- SVR (regression)

### 📦 Evaluation
- Accuracy, F1-score, Precision, Recall
- Cross-validation
- Confusion matrix

---

## 📊 Example Usage

```python
from ml.pipeline.ml_pipeline import PyMLDAPipeline
from ml.models.model_factory import get_models
from sklearn.cluster import KMeans

pipe = PyMLDAPipeline(
    cluster_model=KMeans(n_clusters=4),
    classifiers=get_models(),
    regressor=None
)

pipe.fit(X_train, y_train)

results = pipe.classify(X_test, y_test)
print(results)

📁 Project Structure
PyMLDA/
├── ml/
├── utils/
├── features/
├── signal_processing/
├── main.py

📌 Applications
Structural Health Monitoring (SHM)
Damage detection in mechanical systems
Vibration-based diagnostics
Smart structures and metamaterials
AI-driven condition monitoring

👩‍🔬 Authors
Jefferson Coelho
Marcela Machado
Amanda Aryda