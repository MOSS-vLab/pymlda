class PyMLDAPipeline:

    def __init__(self, cluster_model, classifiers, regressor=None, evaluator=None):
        self.cluster_model = cluster_model
        self.classifiers = classifiers
        self.regressor = regressor
        self.evaluator = evaluator

    # =========================
    # 1. FIT (TREINO GERAL)
    # =========================
    def fit(self, X, y):

        # --- clustering ---
        self.cluster_model.fit(X)
        self.cluster_labels = self.cluster_model.labels_

        # adiciona cluster como feature
        X_aug = X.copy()
        X_aug["cluster"] = self.cluster_labels

        # --- classification ---
        self.fitted_classifiers = {}

        for name, model in self.classifiers.items():
            model.fit(X_aug, y)
            self.fitted_classifiers[name] = model

        return self

    # =========================
    # 2. CLUSTERING
    # =========================
    def cluster(self, X):
        return self.cluster_model.predict(X)

    # =========================
    # 3. CLASSIFICATION
    # =========================
    def classify(self, X, y_true=None):

        X_aug = X.copy()
        X_aug["cluster"] = self.cluster(X)

        results = {}

        for name, model in self.fitted_classifiers.items():
            y_pred = model.predict(X_aug)

            if self.evaluator and y_true is not None:
                results[name] = self.evaluator.compute_classification_metrics(
                    y_true, y_pred
                )
            else:
                results[name] = y_pred

        return results

    # =========================
    # 4. REGRESSION
    # =========================
    def regress(self, X):
        if self.regressor is None:
            raise ValueError("Regressor not defined")

        return self.regressor.predict(X)