import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class ClusteringModel:
    """Wrapper for KMeans clustering."""
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.inertia_ = None

    def fit_predict(self, X):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)
        self.inertia_ = self.model.inertia_
        return self.labels_

    def elbow_plot(self, X):
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=self.random_state)
            km.fit(X)
            inertia.append(km.inertia_)
        return K_range, inertia

class ClassificationModel:
    """Wrapper for Random Forest classification."""
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    def confusion_matrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)

    def feature_importance(self, feature_names):
        importances = pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return importances 