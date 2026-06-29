from utils.io import load_excel
from utils.data_split import split_data
from ml.pipeline.ml_pipeline import PyMLDAPipeline
from ml.models.model_factory import get_models
from sklearn.cluster import KMeans
from ml.evaluation.metrics import compute_classification_metrics


# =========================
# 1. LOAD DATA
# =========================
dataset = load_excel(
    path="data/DI_FRAC_Exp-estimation.xlsx",
    sheet_name="DI_FRAC"
)

# limpeza básica (aqui pode ficar leve no main)
dataset = dataset.drop(['Mass loss [%]', 'Multiclass classification'], axis=1)

X = dataset.iloc[:, :2]
y = dataset.iloc[:, 2]


# =========================
# 2. SPLIT
# =========================
X_train, X_test, y_train, y_test = split_data(X, y)


# =========================
# 3. DEFINE MODELS
# =========================

cluster_model = KMeans(n_clusters=4, random_state=0)

classifiers = get_models()  
# ex: SVM, RF, KNN, XGB (tudo centralizado no factory)

regressor = None  # ou SVR se quiser ativar depois


# =========================
# 4. BUILD PIPELINE
# =========================
pipe = PyMLDAPipeline(
    cluster_model=cluster_model,
    classifiers=classifiers,
    regressor=regressor,
    evaluator=compute_classification_metrics
)


# =========================
# 5. TRAIN
# =========================
pipe.fit(X_train, y_train)


# =========================
# 6. CLUSTERING RESULTS
# =========================
clusters = pipe.cluster(X_test)
print("\nClusters:\n", clusters)


# =========================
# 7. CLASSIFICATION RESULTS
# =========================
results = pipe.classify(X_test, y_test)

print("\nClassification Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    print(metrics)


# =========================
# 8. OPTIONAL REGRESSION
# =========================
# damage = pipe.regress(X_test)
# print(damage)