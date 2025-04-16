import numpy as np
import json
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

# For sequence embedding
from transformers import AutoTokenizer, AutoModel
import torch

# For autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# For graph embeddings
from node2vec import Node2Vec

# Sample data
sample_protein = {
    "accession": "P12345",
    "protein_name": "Sample Protein",
    "gene_name": "SPX",
    "organism": "Homo sapiens",
    "sequence": "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWV",
    "domains": "Kinase domain, SH3 domain",
    "ptms": "Phosphorylation at serine residues",
    "go_terms": "GO:0004672, GO:0005524",
    "ppi": "Interacts with protein A, protein B",
    "pathways": "MAPK signaling pathway",
    "disease_info": "Associated with cancer",
    "evidence": "Reviewed"
}

# List of proteins
proteins_list = [sample_protein]

# 1. Sequence-Based Embeddings
def get_sequence_embedding(sequence: str) -> np.ndarray:
    # Get embedding using ProtBERT
    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    model = AutoModel.from_pretrained('Rostlab/prot_bert')

    sequence_spaced = " ".join(list(sequence))
    inputs = tokenizer(sequence_spaced, return_tensors='pt')
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embedding

# Example
sequence_emb = get_sequence_embedding(sample_protein["sequence"])
print("Sequence Embedding shape:", sequence_emb.shape)

# 2. GO-Term Embeddings
def get_go_embedding(go_terms_str: str, mlb: MultiLabelBinarizer = None):
    # Convert GO terms to one-hot encoding
    go_terms = [term.strip() for term in go_terms_str.split(",") if term.strip()]
    if mlb is None:
        mlb = MultiLabelBinarizer()
        mlb.fit([["GO:0004672", "GO:0005524", "GO:0008150", "GO:0005575"]])
    onehot = mlb.transform([go_terms])[0]
    return onehot, mlb

# Example
go_onehot, go_mlb = get_go_embedding(sample_protein["go_terms"])
print("GO-term One-Hot Encoding:", go_onehot)
print("GO-term Vector length:", len(go_onehot))

def reduce_go_embedding(onehot_vector, target_dim=4):
    # Reduce dimensionality with PCA
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(onehot_vector.reshape(1, -1))[0]
    return reduced

# 3. Graph-Based Embeddings
def create_ppi_graph(proteins):
    # Create protein interaction graph
    G = nx.Graph()
    for protein in proteins:
        G.add_node(protein["accession"])
    if len(proteins) > 1:
        for i in range(len(proteins)-1):
            G.add_edge(proteins[i]["accession"], proteins[i+1]["accession"])
    return G

def get_node2vec_embeddings(graph, dimensions=64):
    # Generate node embeddings
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, num_walks=200, workers=2)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv[str(node)] for node in graph.nodes()}
    return embeddings

# Example
ppi_graph = create_ppi_graph(proteins_list)
graph_embeddings = get_node2vec_embeddings(ppi_graph, dimensions=64)
print("Graph Embedding shape:", graph_embeddings[sample_protein["accession"]].shape)

# 4. Multi-Modal Embeddings
def combine_embeddings(emb1: np.ndarray, emb2: np.ndarray, method="concat", alpha=0.5):
    # Combine multiple embeddings
    if method == "concat":
        combined = np.concatenate([emb1, emb2])
    elif method == "weighted_average":
        if emb1.shape != emb2.shape:
            raise ValueError("For weighted average, embeddings must have the same dimension.")
        combined = alpha * emb1 + (1 - alpha) * emb2
    else:
        raise ValueError("Unknown combination method.")
    return combined

# Example
multi_modal_emb = combine_embeddings(sequence_emb, go_onehot.astype(float), method="concat")
print("Combined Embedding shape:", multi_modal_emb.shape)

# 5. Autoencoder-Based Embeddings
def build_autoencoder(input_dim: int, latent_dim: int):
    # Build autoencoder model
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    latent = Dense(latent_dim, activation='relu', name="latent")(encoded)
    decoded = Dense(128, activation='relu')(latent)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Example
input_dim = multi_modal_emb.shape[0]
latent_dim = 32

autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
print("Autoencoder model created")

# Quick training
X_dummy = np.array([multi_modal_emb for _ in range(100)])
autoencoder.fit(X_dummy, X_dummy, epochs=10, verbose=0)
autoencoded_emb = encoder.predict(multi_modal_emb.reshape(1, -1))[0]
print("Autoencoder latent embedding shape:", autoencoded_emb.shape)

# 6. Ensemble Embeddings
def ensemble_embeddings(embeddings_list, method="average", target_dim=None):
    # Combine multiple embedding methods
    embeddings_array = np.array(embeddings_list)
    
    if method == "average":
        ensemble_emb = np.mean(embeddings_array, axis=0)
    elif method == "concat_pca":
        concat_emb = np.concatenate(embeddings_list)
        if target_dim is None:
            target_dim = concat_emb.shape[0] // 2  
        pca = PCA(n_components=target_dim)
        ensemble_emb = pca.fit_transform(concat_emb.reshape(1, -1))[0]
    else:
        raise ValueError("Unknown ensemble method")
    return ensemble_emb

# Examples
ensemble_emb = ensemble_embeddings(
    [sequence_emb, go_onehot.astype(float), autoencoded_emb], 
    method="average"
)
print("Ensemble Embedding shape (average):", ensemble_emb.shape)

ensemble_emb2 = ensemble_embeddings(
    [sequence_emb, go_onehot.astype(float), autoencoded_emb], 
    method="concat_pca", 
    target_dim=64
)
print("Ensemble Embedding shape (concat PCA):", ensemble_emb2.shape)