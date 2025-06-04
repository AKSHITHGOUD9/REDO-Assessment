# --- Streamlit App: Anti-Reproductive Rights Crimes Analysis ---
# This app analyzes anti-reproductive rights crimes in California using clustering and classification.
# It follows ethical, transparent, and responsible data science practices, inspired by Redo.io and DataKind.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import silhouette_samples, silhouette_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.manifold import TSNE

from models import ClusteringModel, ClassificationModel
from utils import load_data, clean_and_engineer_features, plot_countplot
from config import REQUIRED_COLUMNS, PALETTE_CREST, PALETTE_HUSL, PALETTE_VIRIDIS, PALETTE_MAGMA, PALETTE_CUBEHELIX, PALETTE_TAB10

# --- App Config & Project Motivation ---
st.set_page_config(page_title="Anti-Reproductive Rights Crimes Analysis", layout="wide")
sns.set_palette(PALETTE_HUSL)
st.title("Anti-Reproductive Rights Crimes Analysis")
st.markdown("""
**Why this project?**  
Understanding anti-reproductive rights crimes is crucial for public safety, policy, and advocacy. This project analyzes real incident data from California to:
- Identify patterns and risk factors using clustering and classification.
- Support stakeholders (law enforcement, policymakers, advocates) with actionable insights.
- Demonstrate ethical, transparent, and responsible data science, inspired by best practices from Redo.io and DataKind.
""")

# --- Data Upload & Loading ---
st.sidebar.header("Data Upload & Options")
with st.sidebar.expander("Upload Data & Sample Format", expanded=True):
    st.markdown("**How to use this app:** Upload your CSV file or use the default dataset. For best results, your file should have the following columns:")
    st.markdown("\n".join([f"- {col}" for col in REQUIRED_COLUMNS]))
    sample_df = pd.DataFrame({col: ["sample"] for col in REQUIRED_COLUMNS})
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download sample CSV format", sample_csv, "sample_format.csv")
    uploaded_file = st.file_uploader("Upload your CSV file (must match required columns)", type=["csv"])
    st.caption("If your file does not match the required columns, the app will still try to run, but results may not be accurate.")

df = load_data(uploaded_file)
columns_match = all(col in df.columns for col in REQUIRED_COLUMNS)
if not columns_match:
    st.warning(f"Your file does not match the expected columns. The results may not be accurate, but you can still explore the analysis. Try uploading a file with the recommended columns for best results.")

# --- Data Quality Assessment ---
st.header("1. Data Quality Assessment")
try:
    if st.checkbox("Show raw data sample"):
        st.dataframe(df.head())
        st.caption("This table shows the first few rows of your data. Each row is an incident, and each column is a feature about that incident.")
    quality_report = pd.DataFrame({
        'Total Values': df.count(),
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique(),
        'Data Type': df.dtypes
    })
    st.dataframe(quality_report.astype(str))
    st.caption("This table summarizes the quality of your data, including missing values and data types for each column.")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    if not missing_percent.empty:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.barplot(y=missing_percent.index, x=missing_percent.values, ax=ax, color=sns.color_palette(PALETTE_CREST)[2])
        ax.set_title('Missing Values (%) by Column', fontsize=10)
        ax.set_xlabel('Percent Missing (%)', fontsize=9)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        st.caption("This compact bar chart shows which columns have missing data and how much is missing.")
except Exception as e:
    st.error(f"An error occurred during data quality assessment: {e}. Try uploading a file with the recommended columns for best results.")

# --- Data Cleaning & Feature Engineering ---
df_clean = clean_and_engineer_features(df)

# --- EDA ---
st.header("2. Exploratory Data Analysis (EDA)")
try:
    col1, col2 = st.columns(2)
    with col1:
        # Custom figure size for 'Incidents per Year' plot
        st.pyplot(plot_countplot(df_clean, 'YEAR', title='Incidents per Year', palette=PALETTE_VIRIDIS, figsize=(10, 5)))
        st.caption("This plot shows how many incidents happened each year. Each bar is a year, and the height shows the number of incidents recorded.")
    with col2:
        st.pyplot(plot_countplot(df_clean, 'MONTH', title='Incidents per Month', palette=PALETTE_MAGMA))
        st.caption("This plot shows how incidents are distributed across months. Peaks may indicate seasonal trends.")
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_countplot(df_clean, 'LOCATION TYPE', orient='h', title='Incidents by Location Type', palette=PALETTE_CUBEHELIX))
        st.caption("This plot shows which types of locations are most affected by incidents. Each bar is a location type.")
    with col4:
        top_offenses = df_clean['DESCRIPTION'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=top_offenses.values, y=top_offenses.index, ax=ax, color=sns.color_palette(PALETTE_CREST)[2])
        ax.set_title('Top 10 Offense Descriptions')
        st.pyplot(fig)
        st.caption("This plot shows the most common types of offenses in the data.")
except Exception as e:
    st.error(f"An error occurred during EDA: {e}. Try uploading a file with the recommended columns for best results.")

# --- Clustering Section ---
st.header("3. Clustering Analysis")
st.markdown("""
**Why clustering?**  
Clustering helps us find hidden patterns and group similar incidents together. This can reveal risk factors, hotspots, and trends that are not obvious from raw data. KMeans is used for its simplicity and interpretability, but other methods are available for more complex patterns.
""")
features = [
    'YEAR', 'MONTH', 'VALUE', 'QUANTITY',
    'LOCATION TYPE_ENC', 'TYPE OF LOSS_ENC', 'PROPERTY CATEGORY_ENC',
    'S.RACE_ENC', 'S.GENDER_ENC', 'V.RACE_ENC', 'V.GENDER_ENC',
    'VICTIM TYPE_ENC', 'WEAPON_ENC'
]
X = df_clean[features].astype(float)
X_scaled = (X - X.mean()) / X.std()  # Standardize
clusterer_for_elbow = ClusteringModel(n_clusters=4)
K_range, inertia = clusterer_for_elbow.elbow_plot(X_scaled)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(K_range, inertia, 'bo-')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method For Optimal k')
st.pyplot(fig)
st.caption("The 'elbow' in this plot helps you choose the best number of clusters. The ideal k is where the curve bends and adding more clusters doesn't improve much.")
k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)
clustering_model = ClusteringModel(n_clusters=k)
try:
    df_clean['CLUSTER'] = clustering_model.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_clean['CLUSTER'], palette=PALETTE_TAB10, ax=ax)
    ax.set_title('Clusters Visualized with PCA')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    st.pyplot(fig)
    st.caption("This plot shows how the data is grouped into clusters. Each color is a different group found by the algorithm.")
    st.subheader("Cluster Feature Means")
    st.dataframe(df_clean.groupby('CLUSTER')[features].mean())
    st.caption("This table shows the average value of each feature for each cluster. It helps you understand what makes each group unique.")
    for col in ['LOCATION TYPE', 'DESCRIPTION', 'S.RACE', 'S.GENDER']:
        st.markdown(f"**{col} distribution by cluster:**")
        st.dataframe(pd.crosstab(df_clean['CLUSTER'], df_clean[col]))
        st.caption(f"This table shows how {col.lower()} is distributed across clusters.")
except Exception as e:
    st.error(f"An error occurred during clustering: {e}. Try uploading a file with the recommended columns for best results.")

# --- Classification Section ---
st.header("4. Classification & Prediction")
st.markdown("""
**Why classification?**  
Classification helps us predict which incidents are likely to be violent, supporting prevention and response. Random Forest is chosen for its robustness, interpretability, and ability to handle complex, structured data.
""")
df_clean['IS_VIOLENT'] = df_clean['DESCRIPTION'].str.contains('VIOLENCE|ASSAULT|BATTERY|MURDER|THREAT', case=False, na=False).astype(int)
features_cls = features
X_cls = df_clean[features_cls].astype(float)
y_cls = df_clean['IS_VIOLENT']
classification_model = ClassificationModel()
st.markdown("**Test set size for classification:**")
test_size = st.slider("Test size (fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
st.caption("""
**What does the test size slider do?**
- Increasing the test size means more data is used to test the model, making evaluation more reliable but leaving less data for training.
- Decreasing the test size means more data is used for training, which can help the model learn better, but the evaluation may be less reliable.
- Typical values are 0.2 (20%) for testing and 0.8 (80%) for training.
""")
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=test_size, random_state=42, stratify=y_cls)
try:
    classification_model.fit(X_train, y_train)
    y_pred = classification_model.predict(X_test)
    st.markdown("**Classification Report:**")
    if (
        len(y_pred) > 0
        and len(y_test) > 0
        and len(np.unique(y_test)) > 1
        and len(np.unique(y_pred)) > 1
    ):
        report_text = classification_report(y_test, y_pred)
        st.text(report_text)
        st.caption("""
        **How to read this report:**
        - **Accuracy:** Overall fraction of correct predictions.
        - **Precision:** Of all predicted positives, how many were correct.
        - **Recall:** Of all actual positives, how many were found.
        - **F1-score:** Balance between precision and recall.
        - **Support:** Number of true samples for each class.
        Macro/weighted averages summarize overall performance.
        """)
        st.markdown("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(classification_model.confusion_matrix(y_test, y_pred)))
        st.caption("This table shows how many incidents were correctly or incorrectly classified as violent or not violent.")
        importances = classification_model.feature_importance(features_cls)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=importances.values, y=importances.index, ax=ax, color=sns.color_palette(PALETTE_CREST)[3])
        ax.set_title('Feature Importances (Random Forest)')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig)
        st.caption("This plot shows which features are most important for predicting violence. 'ENC' means the feature is encoded as a number for modeling.")
    else:
        st.warning("Classification report could not be generated. This may be due to insufficient data, missing or incorrect columns, or all predictions being of a single class. Please check your data and try again.")
except Exception as e:
    st.error(f"An error occurred during classification: {e}. Try uploading a file with the recommended columns for best results.")

# --- Experimental: Try Different Models & Visualizations ---
st.header("5. Try Different Models & Visualizations (Experimental)")
st.markdown("""
Experiment with different clustering and classification models to see how results and visualizations change. This section lets you compare a few common alternatives to the recommended models above. These are for exploration and may not be as robust as the main workflow.
""")

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import silhouette_samples, silhouette_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.manifold import TSNE

# Tabs for Clustering and Classification
tab1, tab2 = st.tabs(["Clustering Models", "Classification Models"])

with tab1:
    clustering_options = {
        'KMeans': ClusteringModel,
        'DBSCAN': lambda n_clusters: DBSCAN(eps=st.session_state.get('dbscan_eps', 1.0), min_samples=st.session_state.get('dbscan_min_samples', 5)),
        'Agglomerative': lambda n_clusters: AgglomerativeClustering(n_clusters=n_clusters),
    }
    clustering_vis_options = {
        'KMeans': ['PCA plot', 't-SNE plot', 'Elbow plot', 'Silhouette plot'],
        'DBSCAN': ['PCA plot', 't-SNE plot', 'Cluster size histogram', 'Silhouette plot'],
        'Agglomerative': ['Dendrogram', 'PCA plot', 't-SNE plot', 'Silhouette plot'],
    }
    selected_clustering = st.selectbox("Choose clustering model:", list(clustering_options.keys()), key='alt_clustering_model')
    vis_choice = st.selectbox("Choose visualization:", clustering_vis_options[selected_clustering], key='alt_clustering_vis')
    # Sliders for model params if needed
    if selected_clustering == 'KMeans':
        k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, key='alt_k')
        cluster_model = ClusteringModel(n_clusters=k)
        alt_labels = cluster_model.fit_predict(X_scaled)
    elif selected_clustering == 'DBSCAN':
        eps = st.slider("DBSCAN: eps (neighborhood size)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='alt_dbscan_eps')
        min_samples = st.slider("DBSCAN: min_samples", min_value=2, max_value=10, value=5, key='alt_dbscan_min_samples')
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        alt_labels = cluster_model.fit_predict(X_scaled)
    elif selected_clustering == 'Agglomerative':
        k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, key='alt_agg_k')
        cluster_model = AgglomerativeClustering(n_clusters=k)
        alt_labels = cluster_model.fit_predict(X_scaled)
    try:
        if vis_choice in ['PCA plot', 't-SNE plot']:
            if vis_choice == 'PCA plot':
                reducer = PCA(n_components=2)
                X_vis = reducer.fit_transform(X_scaled)
                title = f'Clusters Visualized with PCA ({selected_clustering})'
            else:
                reducer = TSNE(n_components=2, random_state=42)
                X_vis = reducer.fit_transform(X_scaled)
                title = f'Clusters Visualized with t-SNE ({selected_clustering})'
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=alt_labels, palette=PALETTE_TAB10, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            st.pyplot(fig)
            st.caption(f"This plot shows how the data is grouped into clusters using {selected_clustering}.")
        elif vis_choice == 'Elbow plot' and selected_clustering == 'KMeans':
            K_range, inertia = cluster_model.elbow_plot(X_scaled)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(K_range, inertia, 'bo-')
            ax.set_xlabel('Number of clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method For Optimal k')
            st.pyplot(fig)
            st.caption("The 'elbow' in this plot helps you choose the best number of clusters. The ideal k is where the curve bends and adding more clusters doesn't improve much.")
        elif vis_choice == 'Dendrogram' and selected_clustering == 'Agglomerative':
            Z = linkage(X_scaled, method='ward')
            fig, ax = plt.subplots(figsize=(8, 4))
            dendrogram(Z, ax=ax)
            ax.set_title('Dendrogram (Agglomerative Clustering)')
            st.pyplot(fig)
            st.caption("This dendrogram shows how clusters are merged step by step. Useful for hierarchical relationships.")
        elif vis_choice == 'Silhouette plot':
            if len(set(alt_labels)) > 1:
                n_clusters = len(set(alt_labels))
                fig, ax = plt.subplots(figsize=(7, 5))
                silhouette_avg = silhouette_score(X_scaled, alt_labels)
                sample_silhouette_values = silhouette_samples(X_scaled, alt_labels)
                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = sample_silhouette_values[np.array(alt_labels) == i]
                    ith_cluster_silhouette_values.sort()
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    color = plt.cm.nipy_spectral(float(i) / n_clusters)
                    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10
                ax.set_title("Silhouette plot for the clusters")
                ax.set_xlabel("Silhouette coefficient values")
                ax.set_ylabel("Cluster label")
                ax.axvline(x=silhouette_avg, color="red", linestyle="--")
                st.pyplot(fig)
                st.caption("This plot shows how well each point fits within its cluster. Higher silhouette values mean better clustering.")
            else:
                st.warning("Silhouette plot requires at least 2 clusters. Try increasing the number of clusters or using a different clustering method.")
        elif vis_choice == 'Cluster size histogram' and selected_clustering == 'DBSCAN':
            labels, counts = np.unique(alt_labels, return_counts=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(labels, counts, color=sns.color_palette(PALETTE_CREST))
            ax.set_xlabel('Cluster Label')
            ax.set_ylabel('Size')
            ax.set_title('Cluster Size Histogram (DBSCAN)')
            st.pyplot(fig)
            st.caption("This histogram shows the size of each cluster found by DBSCAN. -1 means noise/outliers.")
        else:
            st.warning("This visualization is not implemented for this model.")
    except Exception as e:
        st.warning(f"Could not run {selected_clustering} visualization: {e}")

with tab2:
    classification_options = {
        'Random Forest': ClassificationModel,
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000),
        'SVM': lambda: SVC(probability=True),
        'XGBoost': lambda: XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    }
    classification_vis_options = {
        'Random Forest': ['Feature Importance', 'Confusion Matrix', 'ROC Curve', 'PR Curve'],
        'Logistic Regression': ['Confusion Matrix', 'ROC Curve', 'PR Curve', 'Decision Boundary'],
        'SVM': ['Confusion Matrix', 'ROC Curve', 'PR Curve', 'Decision Boundary'],
        'XGBoost': ['Feature Importance', 'Confusion Matrix', 'ROC Curve', 'PR Curve'],
    }
    selected_classifier = st.selectbox("Choose classification model:", list(classification_options.keys()), key='alt_classifier_model')
    vis_choice_cls = st.selectbox("Choose visualization:", classification_vis_options[selected_classifier], key='alt_classifier_vis')
    # Fit model
    if selected_classifier == 'Random Forest':
        clf_model = ClassificationModel()
        clf_model.fit(X_train, y_train)
        y_pred_alt = clf_model.predict(X_test)
    elif selected_classifier == 'Logistic Regression':
        clf_model = LogisticRegression(max_iter=1000)
        clf_model.fit(X_train, y_train)
        y_pred_alt = clf_model.predict(X_test)
    elif selected_classifier == 'SVM':
        clf_model = SVC(probability=True)
        clf_model.fit(X_train, y_train)
        y_pred_alt = clf_model.predict(X_test)
    elif selected_classifier == 'XGBoost':
        clf_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        clf_model.fit(X_train, y_train)
        y_pred_alt = clf_model.predict(X_test)
    try:
        if (
            len(y_pred_alt) > 0
            and len(y_test) > 0
            and len(np.unique(y_test)) > 1
            and len(np.unique(y_pred_alt)) > 1
        ):
            if vis_choice_cls == 'Feature Importance' and selected_classifier in ['Random Forest', 'XGBoost']:
                if selected_classifier == 'Random Forest':
                    importances = clf_model.feature_importance(features_cls)
                else:
                    importances = pd.Series(clf_model.feature_importances_, index=features_cls).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=importances.values, y=importances.index, ax=ax, color=sns.color_palette(PALETTE_CREST)[3])
                ax.set_title(f'Feature Importances ({selected_classifier})')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                st.pyplot(fig)
                st.caption("This plot shows which features are most important for predicting violence. 'ENC' means the feature is encoded as a number for modeling.")
            elif vis_choice_cls == 'Confusion Matrix':
                st.markdown("**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred_alt)))
                st.caption("This table shows how many incidents were correctly or incorrectly classified as violent or not violent.")
            elif vis_choice_cls == 'ROC Curve':
                if hasattr(clf_model, "predict_proba"):
                    y_score = clf_model.predict_proba(X_test)[:, 1]
                elif hasattr(clf_model, "decision_function"):
                    y_score = clf_model.decision_function(X_test)
                else:
                    st.warning("ROC Curve is not available for this model.")
                    y_score = None
                if y_score is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
            elif vis_choice_cls == 'PR Curve':
                if hasattr(clf_model, "predict_proba"):
                    y_score = clf_model.predict_proba(X_test)[:, 1]
                elif hasattr(clf_model, "decision_function"):
                    y_score = clf_model.decision_function(X_test)
                else:
                    st.warning("PR Curve is not available for this model.")
                    y_score = None
                if y_score is not None:
                    precision, recall, _ = precision_recall_curve(y_test, y_score)
                    avg_precision = average_precision_score(y_test, y_score)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.step(recall, precision, color='b', alpha=0.2, where='post')
                    ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlim([0.0, 1.0])
                    ax.set_title(f'Precision-Recall curve: AP={avg_precision:.2f}')
                    st.pyplot(fig)
            elif vis_choice_cls == 'Decision Boundary' and selected_classifier in ['Logistic Regression', 'SVM']:
                X_vis = X_test.iloc[:, :2].values
                y_vis = y_test.values
                h = .02
                x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
                y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = clf_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                fig, ax = plt.subplots(figsize=(7, 5))
                cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
                ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
                scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='bwr', edgecolor='k', s=40)
                ax.set_title(f"Decision Boundary ({selected_classifier}) - First 2 Features")
                st.pyplot(fig)
                st.caption("This plot shows how the model separates violent from non-violent incidents using the first two features.")
            else:
                st.warning("This visualization is not implemented for this model.")
        else:
            st.warning("Classification report could not be generated for this model. This may be due to insufficient data, missing or incorrect columns, or all predictions being of a single class.")
    except Exception as e:
        st.warning(f"Could not run {selected_classifier} visualization: {e}")

# --- Stakeholder Context & References ---
st.header("6. Stakeholder Context & References")
st.markdown("""
**Stakeholder Context & Impact:**
Stakeholders include policymakers, law enforcement, advocacy groups, and affected communities. The analysis aims to inform prevention and response strategies, highlight risk factors, and support resource allocation. Care was taken to avoid stigmatization and to ensure findings are communicated responsibly.

**References:**
- Data Source: California Department of Justice (example placeholder)
- Redo.io Responsible Data Science: https://redoio.info/
- DataKind Ethics: https://www.datakind.org/2022/01/11/our-ethics-responsible-data-science-practices-at-datakind/
- Scikit-learn documentation: https://scikit-learn.org/
- Seaborn documentation: https://seaborn.pydata.org/
""")
