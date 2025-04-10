import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

st.set_page_config(page_title="Clustering Tool", layout="wide")
st.title("Clustering Software with PCA, KMeans, and GMM")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your dataset:")
    st.dataframe(df.head())

    # Allow the user to select the ID column
    id_column = st.selectbox("Select the ID column (optional):", options=["None"] + list(df.columns), index=0)

    # Preserve the ID column if selected
    if id_column != "None":
        id_data = df[[id_column]]  # Preserve the ID column
        df = df.drop(columns=[id_column])  # Remove the ID column from the main DataFrame
    else:
        id_data = None

    # Check for non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        st.warning(
            f"The dataset contains non-numeric columns: {non_numeric_columns}. "
            "Please click on 'Remove non-numeric columns' to continue."
        )

    # Option to remove non-numeric columns
    remove_non_numeric = st.checkbox("Remove non-numeric columns")
    if remove_non_numeric:
        df = df.select_dtypes(include=[np.number])

    # Proceed only if all columns are numeric
    if df.select_dtypes(exclude=[np.number]).shape[1] == 0:
        all_columns = df.columns.tolist()

        # Feature Selection
        selected_columns = st.multiselect("Select features for clustering", all_columns, default=all_columns)

        if selected_columns:
            selected_df = df[selected_columns]

            # Option to remove rows with missing values
            missing_rows_count = selected_df[selected_df.isnull().any(axis=1)].shape[0]
            if missing_rows_count > 0:
                st.warning(f"The dataset contains {missing_rows_count} rows with missing values. Please click on 'Remove rows with missing values (NaN)' to continue.")
            
            remove_missing = st.checkbox("Remove rows with missing values (NaN)")
            if remove_missing:
                selected_df = selected_df.dropna()
                if id_data is not None:
                    id_data = id_data.loc[selected_df.index]  # Update ID data to match the cleaned DataFrame
                df = selected_df  # Update the main DataFrame to reflect the cleaned version

            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(selected_df)

            # Show the preprocessed data for clustering
            st.write("### Preprocessed Data for Clustering:")
            preprocessed_df = pd.DataFrame(scaled_data, columns=selected_columns)
            st.dataframe(preprocessed_df.head())

            # Show the size of the preprocessed data
            st.write(f"**Size of the preprocessed data:** {preprocessed_df.shape[0]} rows and {preprocessed_df.shape[1]} columns")

            # PCA Option
            apply_pca = st.checkbox("Apply PCA")

            if apply_pca:
                max_components = min(len(selected_columns), len(selected_df))
                n_components = st.slider("Number of Principal Components", 1, max_components, 2)

                pca = PCA(n_components=n_components)
                pca_data = pca.fit_transform(scaled_data)

                st.write("### PCA Result (first 5 rows):")
                pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_components)])
                st.dataframe(pca_df.head())

                st.write("### Explained Variance Ratio:")
                explained_variance_ratio = pca.explained_variance_ratio_

                # Create a bar chart with custom x-axis labels
                fig, ax = plt.subplots()  # Default figure size
                x_labels = [f"PC{i+1}" for i in range(len(explained_variance_ratio))]
                ax.bar(x_labels, explained_variance_ratio, color="skyblue")
                ax.set_xlabel("Principal Components")
                ax.set_ylabel("Explained Variance Ratio")
                ax.set_title("Explained Variance Ratio by Principal Component")
                st.pyplot(fig)

                # Calculate and display cumulative variance
                cumulative_variance = np.cumsum(explained_variance_ratio)
                st.write("### Cumulative Variance Explained:")
                for i, var in enumerate(cumulative_variance, start=1):
                    st.write(f"PC{i}: {var:.2%}")

                data_for_clustering = pca_data
                clustering_columns = [f"PC{i+1}" for i in range(n_components)]  # Use PCA column names
            else:
                data_for_clustering = scaled_data
                clustering_columns = selected_columns  # Use original column names

            # Clear session state for helper plots and clustering results when PCA changes
            if "apply_pca_state" not in st.session_state or st.session_state["apply_pca_state"] != apply_pca:
                st.session_state["apply_pca_state"] = apply_pca
                st.session_state.pop("aics", None)
                st.session_state.pop("bics", None)
                st.session_state.pop("inertias", None)
                st.session_state.pop("kmeans_inertias", None)  # Clear KMeans helper plot data
                st.session_state.pop("gmm_aics", None)        # Clear GMM AIC data
                st.session_state.pop("gmm_bics", None)        # Clear GMM BIC data
                st.session_state.pop("sparse_kmeans_inertias", None)  # Clear Sparse KMeans helper plot data
                st.session_state.pop("clustered_df", None)    # Clear clustered data
                st.session_state.pop("cluster_labels", None)  # Clear cluster labels

        # --- Clustering Section ---
        st.subheader("Choose Clustering Algorithm")
        algorithm = st.selectbox(
            "Clustering algorithm",
            ["Select an algorithm", "KMeans", "Gaussian Mixture Model", "DBSCAN", "KMeans with Variance Threshold", "Agglomerative Clustering", "Spectral Clustering"]
        )

        if algorithm != "Select an algorithm":
            # Define session state keys for each algorithm to track changes
            if algorithm == "KMeans":
                max_k = st.slider("Maximum number of clusters to evaluate", 2, 15, 10, key="kmeans_max_k")
                n_clusters = st.slider("Final number of clusters", 2, max_k, 3, key="kmeans_n_clusters")

                # Elbow Method for KMeans
                if "kmeans_inertias" not in st.session_state or st.session_state.get("kmeans_max_k_prev") != max_k:
                    st.session_state["kmeans_max_k_prev"] = max_k
                    st.session_state["kmeans_inertias"] = []

                    for k in range(1, max_k + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(data_for_clustering)
                        st.session_state["kmeans_inertias"].append(kmeans.inertia_)

                # Plot the Elbow Method
                st.write("### Elbow Method for KMeans")
                fig, ax = plt.subplots()
                ax.plot(range(1, max_k + 1), st.session_state["kmeans_inertias"], marker='o')
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Method")
                st.pyplot(fig)

            elif algorithm == "Gaussian Mixture Model":
                max_k = st.slider("Maximum number of components to evaluate", 2, 15, 10, key="gmm_max_k")
                n_clusters = st.slider("Final number of components", 2, max_k, 3, key="gmm_n_clusters")
                covariance_type = st.selectbox(
                    "Choose covariance type for GMM",
                    ["full", "tied", "diag", "spherical"],
                    index=0,
                    key="gmm_covariance_type"
                )

                # AIC/BIC for GMM
                if "gmm_aics" not in st.session_state or "gmm_bics" not in st.session_state or \
                        st.session_state.get("gmm_max_k_prev") != max_k or st.session_state.get("gmm_covariance_type_prev") != covariance_type:
                    st.session_state["gmm_max_k_prev"] = max_k
                    st.session_state["gmm_covariance_type_prev"] = covariance_type
                    st.session_state["gmm_aics"] = []
                    st.session_state["gmm_bics"] = []

                    for k in range(1, max_k + 1):
                        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
                        gmm.fit(data_for_clustering)
                        st.session_state["gmm_aics"].append(gmm.aic(data_for_clustering))
                        st.session_state["gmm_bics"].append(gmm.bic(data_for_clustering))

                # Plot AIC and BIC
                st.write("### AIC/BIC for GMM")
                fig, ax = plt.subplots()
                ax.plot(range(1, max_k + 1), st.session_state["gmm_aics"], label="AIC", marker='o')
                ax.plot(range(1, max_k + 1), st.session_state["gmm_bics"], label="BIC", marker='s')
                ax.set_xlabel("Number of components")
                ax.set_ylabel("Score")
                ax.set_title(f"AIC/BIC for GMM (Covariance Type: {covariance_type})")
                ax.legend()
                st.pyplot(fig)

            elif algorithm == "DBSCAN":
                eps = st.slider("Epsilon (eps)", 0.1, 10.0, 0.5, step=0.1, key="dbscan_eps")
                min_samples = st.slider("Minimum Samples", 1, 20, 5, key="dbscan_min_samples")

                # Add k-distance plot
                if st.checkbox("Show k-distance Plot", key="dbscan_k_distance_plot"):
                    with st.spinner("Generating k-distance plot..."):
                        neighbors = NearestNeighbors(n_neighbors=min_samples)
                        neighbors_fit = neighbors.fit(data_for_clustering)
                        distances, _ = neighbors_fit.kneighbors(data_for_clustering)
                        distances = np.sort(distances[:, -1])  # Sort distances to the k-th nearest neighbor

                        fig, ax = plt.subplots()
                        ax.plot(distances)
                        ax.set_title("k-distance Plot")
                        ax.set_xlabel("Points sorted by distance to k-th nearest neighbor")
                        ax.set_ylabel("Distance to k-th nearest neighbor")
                        st.pyplot(fig)

            elif algorithm == "KMeans with Variance Threshold":
                variance_threshold = st.slider(
                    "Variance Threshold for Feature Selection", 0.0, 1.0, 0.1, step=0.01, key="kmeans_variance_threshold"
                )
                max_k = st.slider("Maximum number of clusters to evaluate", 2, 15, 10, key="kmeans_variance_max_k")
                n_clusters = st.slider("Final number of clusters", 2, max_k, 3, key="kmeans_variance_n_clusters")

                # Apply variance threshold for feature selection
                with st.spinner("Applying variance threshold..."):
                    selector = VarianceThreshold(threshold=variance_threshold)
                    sparse_data = selector.fit_transform(data_for_clustering)

                    # Dynamically adjust selected features based on PCA or original data
                    if apply_pca:
                        adjusted_columns = clustering_columns  # Use PCA column names
                    else:
                        adjusted_columns = selected_columns  # Use original column names

                    selected_features = np.array(adjusted_columns)[selector.get_support()]

                    # Provide a meaningful message about the selected features
                    if len(selected_features) > 0:
                        st.success(f"Variance threshold applied successfully. {len(selected_features)} features were selected for clustering: {', '.join(selected_features)}.")
                    else:
                        st.warning("No features were selected after applying the variance threshold. Please adjust the threshold or select different features.")

                # Elbow Method for KMeans with Variance Threshold
                if "kmeans_variance_inertias" not in st.session_state or st.session_state.get("kmeans_variance_max_k_prev") != max_k:
                    st.session_state["kmeans_variance_max_k_prev"] = max_k
                    st.session_state["kmeans_variance_inertias"] = []

                    for k in range(1, max_k + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(sparse_data)
                        st.session_state["kmeans_variance_inertias"].append(kmeans.inertia_)

                # Plot the Elbow Method
                st.write("### Elbow Method for KMeans with Variance Threshold")
                fig, ax = plt.subplots()
                ax.plot(range(1, max_k + 1), st.session_state["kmeans_variance_inertias"], marker='o')
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Method for KMeans with Variance Threshold")
                st.pyplot(fig)

            elif algorithm == "Spectral Clustering":
                n_clusters = st.slider("Number of clusters", 2, 15, 3, key="spectral_n_clusters")
                
                # Add a note about the affinity being used
                st.info("Using 'nearest_neighbors' affinity for Spectral Clustering.")

                # Slider for the number of neighbors
                n_neighbors = st.slider(
                    "Number of Neighbors", 2, 20, 10, key="spectral_n_neighbors"
                )

                # Create the SpectralClustering model
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    random_state=42
                )

                # Fit the model and get cluster labels
                cluster_labels = model.fit_predict(data_for_clustering)

            if algorithm == "Agglomerative Clustering":
                linkage_method = st.selectbox(
                    "Linkage Criterion",
                    ["ward", "complete", "average", "single"],
                    index=0,  # Default to "ward"
                    key="agg_linkage_method"
                )
                n_clusters = st.slider("Number of clusters", 2, 15, 3, key="agg_n_clusters")

                # Add a checkbox to show the dendrogram
                show_dendrogram = st.checkbox("Show Dendrogram", key="agg_dendrogram")

                if show_dendrogram:
                    from scipy.cluster.hierarchy import dendrogram, linkage

                    # Slider to adjust the depth of the dendrogram
                    truncate_depth = st.slider(
                        "Dendrogram Depth (truncate mode)", 0, 20, n_clusters, key="agg_dendrogram_depth"
                    )

                    # Generate the linkage matrix
                    with st.spinner("Generating dendrogram..."):
                        linkage_matrix = linkage(data_for_clustering, method=linkage_method)
                        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
                        dendrogram(
                            linkage_matrix,
                            ax=ax,
                            color_threshold=linkage_matrix[-(n_clusters - 1), 2],  # Highlight clusters
                            truncate_mode="level" if truncate_depth > 0 else None,  # Apply truncation if depth > 0
                            p=truncate_depth  # Depth of the dendrogram
                        )
                        ax.set_title("Dendrogram")
                        ax.set_xlabel("Samples or Clusters")
                        ax.set_ylabel("Distance")
                        st.pyplot(fig)

                # Fit the Agglomerative Clustering model
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                cluster_labels = model.fit_predict(data_for_clustering)

            # Automatically update clustering results when variables change
            if algorithm == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                cluster_labels = model.fit_predict(data_for_clustering)
            elif algorithm == "Spectral Clustering":
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",  # Directly use "nearest_neighbors"
                    n_neighbors=n_neighbors,
                    random_state=42
                )
                cluster_labels = model.fit_predict(data_for_clustering)
            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = model.fit_predict(data_for_clustering)
            elif algorithm == "Gaussian Mixture Model":
                model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
                cluster_labels = model.fit_predict(data_for_clustering)
            elif algorithm == "KMeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = model.fit_predict(data_for_clustering)
            elif algorithm == "KMeans with Variance Threshold":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = model.fit_predict(sparse_data)

            # Add the Cluster column to the cleaned DataFrame
            clustered_df = selected_df.copy()
            clustered_df["Cluster"] = cluster_labels.astype(int)

            # Add the ID column back if it exists
            if id_data is not None:
                clustered_df = pd.concat([id_data.reset_index(drop=True), clustered_df.reset_index(drop=True)], axis=1)

            # Store the clustered DataFrame and cluster labels in session state
            st.session_state["clustered_df"] = clustered_df
            st.session_state["cluster_labels"] = cluster_labels
            st.session_state["data_for_clustering"] = sparse_data if algorithm == "KMeans with Variance Threshold" else data_for_clustering
            st.session_state["selected_columns"] = list(selected_features) if algorithm == "KMeans with Variance Threshold" else selected_columns

            # Calculate and store cluster centroids' means and variances using non-standardized data
            if algorithm not in ["DBSCAN", "Spectral Clustering"]:  # These do not have centroids
                non_standardized_centroids = pd.DataFrame(columns=selected_columns)
                cluster_variances = pd.DataFrame(columns=selected_columns)

                for cluster in range(n_clusters):
                    cluster_data = clustered_df[clustered_df["Cluster"] == cluster][selected_columns]
                    non_standardized_centroids.loc[cluster] = cluster_data.mean()
                    cluster_variances.loc[cluster] = cluster_data.var()

                st.session_state["non_standardized_centroids"] = non_standardized_centroids
                st.session_state["cluster_variances"] = cluster_variances

            # Display Results from Session State
            st.write("### Dataset with Cluster Labels:")
            st.dataframe(st.session_state["clustered_df"].head())

            # Display cluster means and variances
            if algorithm not in ["DBSCAN", "Spectral Clustering"]:
                st.write("### Cluster Centroids' Means and Variances (Non-Standardized Data):")
                st.write("**Cluster Centroids' Means:**")
                st.dataframe(st.session_state["non_standardized_centroids"])
                st.write("**Cluster Centroids' Variances:**")
                st.dataframe(st.session_state["cluster_variances"])

            # Add a download button for the clustered data
            st.download_button(
                label="Download clustered data as CSV",
                data=st.session_state["clustered_df"].to_csv(index=False).encode('utf-8'),
                file_name='clustered_data.csv',
                mime='text/csv',
                key="download_clustered_data"
            )

            # Visualizations
            st.subheader("Cluster Visualization")
            plot_type = st.selectbox(
                "Choose a visualization type",
                ["Pairwise Scatter Plot", "PCA 2D Scatter Plot", "t-SNE", "UMAP"],
                key="visualization_type"
            )

            # Define a colorful palette
            color_palette = sns.color_palette("husl", n_colors=len(np.unique(cluster_labels)))

            if plot_type == "Pairwise Scatter Plot":
                st.write("### Pairwise Scatter Plot (colored by Cluster)")
                # Dynamically set column names based on whether PCA is applied
                viz_columns = clustering_columns if apply_pca else selected_columns
                viz_df = pd.DataFrame(data_for_clustering, columns=viz_columns)
                viz_df["Cluster"] = cluster_labels
                with st.spinner("Generating pairplot..."):
                    fig = sns.pairplot(viz_df, hue="Cluster", diag_kind="kde", palette=color_palette)
                    st.pyplot(fig)

            elif plot_type == "PCA 2D Scatter Plot":
                st.write("### PCA 2D Scatter Plot (colored by Cluster)")
                if data_for_clustering.shape[1] < 2:
                    st.warning("PCA 2D Scatter Plot requires at least 2 dimensions. Please adjust your PCA settings.")
                else:
                    pca_2d = PCA(n_components=2)
                    pca_2d_data = pca_2d.fit_transform(data_for_clustering)
                    pca_2d_df = pd.DataFrame(pca_2d_data, columns=["PCA1", "PCA2"])
                    pca_2d_df["Cluster"] = cluster_labels
                    fig, ax = plt.subplots()
                    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette=color_palette, data=pca_2d_df, ax=ax, legend="full")
                    ax.set_title("PCA 2D Scatter Plot")
                    st.pyplot(fig)

            elif plot_type == "t-SNE":
                st.write("### t-SNE Visualization (colored by Cluster)")
                if data_for_clustering.shape[0] < 2 or data_for_clustering.shape[1] < 2:
                    st.warning("t-SNE requires at least 2 samples and 2 features. Please adjust your dataset or PCA settings.")
                else:
                    perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perplexity")
                    learning_rate = st.slider("Learning Rate", 10, 1000, 200, key="tsne_learning_rate")
                    n_iter = st.slider("Number of Iterations", 250, 1000, 500, key="tsne_n_iter")
                    tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
                    tsne_results = tsne.fit_transform(data_for_clustering)
                    tsne_df = pd.DataFrame(tsne_results, columns=["t-SNE1", "t-SNE2"])
                    tsne_df["Cluster"] = cluster_labels
                    fig, ax = plt.subplots()
                    sns.scatterplot(x="t-SNE1", y="t-SNE2", hue="Cluster", palette=color_palette, data=tsne_df, ax=ax, legend="full")
                    ax.set_title("t-SNE Visualization")
                    st.pyplot(fig)

            elif plot_type == "UMAP":
                st.write("### UMAP Visualization (colored by Cluster)")
                n_neighbors = st.slider("Number of Neighbors", 5, 50, 15, key="umap_n_neighbors")
                min_dist = st.slider("Minimum Distance", 0.0, 1.0, 0.1, key="umap_min_dist")
                umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                umap_results = umap_model.fit_transform(data_for_clustering)
                umap_df = pd.DataFrame(umap_results, columns=["UMAP1", "UMAP2"])
                umap_df["Cluster"] = cluster_labels
                fig, ax = plt.subplots()
                sns.scatterplot(x="UMAP1", y="UMAP2", hue="Cluster", palette=color_palette, data=umap_df, ax=ax, legend="full")
                ax.set_title("UMAP Visualization")
                st.pyplot(fig)

    else:
        st.error("Please remove all non-numeric columns to proceed.")