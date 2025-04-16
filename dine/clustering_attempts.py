import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # Let's keep things clean without warnings

def load_embeddings(file_path):
    """
    Let's get those embeddings from your CSV file!
    
    I'll try to be smart about finding which columns contain your 256-bit vectors.
    """
    print(f"Reading your data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # First, let's try the obvious approach - looking for columns that sound like embeddings
    embedding_cols = [col for col in df.columns if 'embed' in col.lower() or 'vector' in col.lower()]
    
    # If that didn't work, we'll fall back to just taking numeric columns
    if not embedding_cols:
        print("Hmm, couldn't find columns with 'embed' or 'vector' in their names.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        embedding_cols = numeric_cols[:256]  # Grabbing the first 256 since you mentioned 256-bit embeddings
    
    print(f"Great! Found {len(embedding_cols)} dimensions in your embeddings.")
    
    # Create a nice array of just the embeddings
    embeddings = df[embedding_cols].values
    
    return df, embeddings

def determine_optimal_k(embeddings, max_k=20):
    """
    Let's figure out how many clusters we should use!
    
    We'll try the elbow method (looking for where adding more clusters stops helping much)
    and the silhouette method (measuring how well-defined our clusters are).
    """
    print("Hunting for the optimal number of clusters...")
    
    # We'll check different numbers of clusters and see what works best
    inertia_values = []
    silhouette_values = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        print(f"  Trying {k} clusters...", end="\r")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
        
        # Let's also calculate the silhouette score - higher is better!
        if len(np.unique(kmeans.labels_)) > 1:  # Need at least 2 clusters
            silhouette_values.append(silhouette_score(embeddings, kmeans.labels_))
        else:
            silhouette_values.append(-1)
    
    # Let's make some pretty charts to visualize this
    plt.figure(figsize=(12, 5))
    
    # The elbow chart - look for the "bend"
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia_values, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (lower is better)')
    plt.title('Elbow Method - Look for the Bend!')
    plt.grid(True)
    
    # The silhouette chart - higher is better
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_values, 'ro-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score (higher is better)')
    plt.title('Silhouette Method - Higher is Better!')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    
    # Let's pick the best k based on silhouette
    optimal_k = k_values[np.argmax(silhouette_values)]
    print(f"Looks like {optimal_k} clusters is our best bet!")
    
    return {
        'optimal_k': optimal_k,
        'k_values': k_values,
        'inertia_values': inertia_values,
        'silhouette_values': silhouette_values
    }

def apply_kmeans_clustering(embeddings, n_clusters):
    """
    K-means is like dividing your data into n_clusters different groups,
    trying to keep similar items together.
    
    It's great for nice, roundish clusters!
    """
    print(f"Grouping your data into {n_clusters} clusters using K-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Let's see how well we did
    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f} (closer to 1.0 is better)")
    
    return cluster_labels

def apply_dbscan_clustering(embeddings, eps=0.5, min_samples=5):
    """
    DBSCAN is awesome for finding clusters of weird shapes!
    
    It's especially good at finding outliers too - it'll mark them as noise.
    The eps parameter is like the maximum distance between neighbors.
    """
    print(f"Running DBSCAN to find natural clusters in your data...")
    print(f"Using neighborhood radius {eps} and minimum group size {min_samples}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    
    # Let's count up what we found
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"Found {n_clusters} clusters naturally in your data")
    print(f"About {n_noise} points ({n_noise/len(cluster_labels):.2%}) seem to be outliers")
    
    # If we can, let's see how well-defined our clusters are
    if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels:
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f} (closer to 1.0 is better)")
    
    return cluster_labels

def apply_hdbscan_clustering(embeddings, min_cluster_size=5, min_samples=None):
    """
    HDBSCAN is like DBSCAN's smarter cousin!
    
    It handles varying density clusters better, so it's perfect when
    some groups in your data are tightly packed and others are spread out.
    """
    print(f"Using HDBSCAN to find clusters of varying density...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                               min_samples=min_samples,
                               gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Let's count what we found
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"HDBSCAN found {n_clusters} natural clusters")
    print(f"Identified {n_noise} points ({n_noise/len(cluster_labels):.2%}) as outliers")
    
    return cluster_labels

def visualize_clusters(embeddings, labels, method='pca', title='Cluster Visualization'):
    """
    Let's make some pretty pictures of your clusters!
    
    Since we can't easily visualize 256 dimensions, we'll squish everything
    down to 2D using either PCA (faster) or UMAP (better at preserving structure).
    """
    print(f"Creating a {method.upper()} visualization of your clusters...")
    
    # First, let's get our data down to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_var = sum(reducer.explained_variance_ratio_)
        title = f"{title} (PCA captures {explained_var:.2%} of variation)"
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = f"{title} (UMAP preserves local relationships better)"
    
    # Now let's make our plot
    plt.figure(figsize=(12, 10))
    
    # Handle any noise points first
    unique_labels = set(labels)
    if -1 in unique_labels:
        noise_mask = labels == -1
        plt.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1], 
                   color='gray', marker='.', alpha=0.3, label='Noise (outliers)')
        unique_labels.remove(-1)
    
    # Now plot each cluster with a different color
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   marker='.', alpha=0.7, label=f'Cluster {label}')
    
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'clusters_{method}.png')
    print(f"Saved visualization to clusters_{method}.png")

def save_results(df, cluster_labels, output_file='clustered_data.csv'):
    """
    Let's save our clustering results so you can use them later!
    
    We'll add a 'cluster' column to your original data.
    """
    df['cluster'] = cluster_labels
    df.to_csv(output_file, index=False)
    print(f"Saved your clustered data to {output_file}")

def main():
    """
    This is where the magic happens! We'll run through the whole
    clustering pipeline from start to finish.
    """
    # Change this to your actual file path!
    file_path = "your_embeddings.csv"
    
    # Step 1: Load your embedding data
    print("Step 1: Loading your data...")
    df, embeddings = load_embeddings(file_path)
    
    # Step 2: Scale the data so all dimensions are treated equally
    print("\nStep 2: Scaling your embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    print("Done! All dimensions now have mean=0 and std=1")
    
    # Step 3: Find the best number of clusters for K-means
    print("\nStep 3: Finding the optimal number of clusters...")
    cluster_eval = determine_optimal_k(embeddings_scaled, max_k=20)
    optimal_k = cluster_eval['optimal_k']
    
    # Step 4: Run all our clustering algorithms
    print("\nStep 4: Applying different clustering algorithms...")
    print("\n--- K-means clustering ---")
    kmeans_labels = apply_kmeans_clustering(embeddings_scaled, optimal_k)
    
    print("\n--- DBSCAN clustering ---")
    # Note: You might need to play with the eps value!
    dbscan_labels = apply_dbscan_clustering(embeddings_scaled, eps=1.0, min_samples=5)
    
    print("\n--- HDBSCAN clustering ---")
    hdbscan_labels = apply_hdbscan_clustering(embeddings_scaled, min_cluster_size=5)
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating visualizations...")
    print("\nMaking K-means visualizations:")
    visualize_clusters(embeddings_scaled, kmeans_labels, method='pca', title='K-means Clustering')
    visualize_clusters(embeddings_scaled, kmeans_labels, method='umap', title='K-means Clustering')
    
    print("\nMaking DBSCAN visualizations:")
    visualize_clusters(embeddings_scaled, dbscan_labels, method='pca', title='DBSCAN Clustering')
    visualize_clusters(embeddings_scaled, dbscan_labels, method='umap', title='DBSCAN Clustering')
    
    print("\nMaking HDBSCAN visualizations:")
    visualize_clusters(embeddings_scaled, hdbscan_labels, method='pca', title='HDBSCAN Clustering')
    visualize_clusters(embeddings_scaled, hdbscan_labels, method='umap', title='HDBSCAN Clustering')
    
    # Step 6: Save results
    print("\nStep 6: Saving all results...")
    save_results(df, kmeans_labels, output_file='kmeans_clustered_data.csv')
    save_results(df, dbscan_labels, output_file='dbscan_clustered_data.csv')
    save_results(df, hdbscan_labels, output_file='hdbscan_clustered_data.csv')
    
    print("\n🎉 All done! Your data has been clustered three different ways.")
    print("Now you can compare the results and pick the one that works best for your needs!")

if __name__ == "__main__":
    main()