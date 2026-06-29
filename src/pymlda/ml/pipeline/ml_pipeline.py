class MLDA:

    def __init__(
        self,
        cluster_model,
        classifiers=None,
        regressor=None,
        evaluator=None,
    ):

        self.cluster_model = cluster_model
        self.classifiers = classifiers or {}
        self.regressor = regressor
        self.evaluator = evaluator

        self.X = None
        self.y = None

        self.cluster_labels = None
        self.X_clustered = None

        self.fitted_classifiers = {}

    # ==========================================
    # Fit
    # ==========================================
    def fit(self, X):

        """
        Store the feature table for subsequent analyses.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        self
        """

        self.X = X.copy()

        return self
    # ==========================================
    # Cluster
    # ==========================================
    def cluster(self):

        """
        Perform clustering using the selected clustering model.
        """

        if self.X is None:
            raise RuntimeError("Run fit(X) before cluster().")

        self.cluster_model.fit(self.X)

        self.cluster_labels = self.cluster_model.labels_

        self.X_clustered = self.X.copy()
        self.X_clustered["cluster"] = self.cluster_labels

        return self.cluster_labels    
    # ==========================================
    # Classify
    # ==========================================
    def classify(self, y):

        """
        Train and evaluate all classifiers.
        """

        if self.cluster_labels is None:
            raise RuntimeError(
                "Run cluster() before classify()."
            )

        self.y = y

        self.fitted_classifiers = {}

        results = {}

        for name, model in self.classifiers.items():

            model.fit(self.X_clustered, y)

            self.fitted_classifiers[name] = model

            if self.evaluator is not None:

                y_pred = model.predict(self.X_clustered)

                results[name] = self.evaluator.compute_classification_metrics(
                    y,
                    y_pred,
                )
        return results
    # ==========================================
    # Regression
    # ==========================================
    def regress(self, damage):

        """
        Train the regression model.
        """
        if self.regressor is None:
            raise RuntimeError(
                "No regression model defined."
            )
        self.regressor.fit(self.X_clustered, damage)

        prediction = self.regressor.predict(self.X_clustered)

        return prediction