# Meeting Notes 2/7/2025

## 1. Extracting Relationships Between Proteins
- Utilize LLMs and protein catalogs to discover and evaluate connections between proteins.
- Determine the best information to scrape from UniProt to create the embeddings:
  - GO annotations
  - Others
  - Check my existing ones
- Experiment with different LLMs for embeddings.
- Generate embeddings from protein catalogs through LLMs for further analysis.
- Determine the best embedding possible (be creative).
- **Idea for Later:** Each protein will have a graph representation based on its properties.
  - Graph representation was only mentioned for the other model, which we could use as validation.
  - Each protein will have an embedding based on its catalog page.
- Perform downstream analysis, such as clustering proteins into meaningful groups. Other downstream analyses are welcome.
- Compare the resulting protein relationships (clusters and networks) against existing Protein-Protein Interaction (PPI) networks datasets.

## 2. Clustering & Data Comparison
- Perform clustering on protein embeddings developed through LLMs.
- Research optimal clustering algorithms:
  - Control number of clusters.
  - Adjust for outliers.
  - Manage cluster sizes.
  - Try more algorithms and make an assessment.
- **OBS:** Compare findings with relevant genomic datasets (LATER):
  - We know proteins are regulated by genes, so comparing the networks between proteins and genes might help understand something.
  - **Science Dataset 1** – Complex dataset for a slightly different scope.
  - **Science Dataset 2** – Another complex dataset for a different perspective.

## 3. Brain Disorder Proteins & UniProt Extraction
- Identify brain disorder-related proteins using UniProt.
- Initially, focus on human proteins.
- Extract Gene Ontology (GO) annotations from UniProt.
- Prepare human-language descriptions for each protein to use as LLM input.
- Compare discovered clusters with brain disorder protein interaction networks.
- Ensure my dataset and the comparison dataset have similar proteins.
- If I find a good Schizophrenia-related protein dataset, then I focus on Schizophrenia-related proteins first from the UniProt dataset.
- Experiment with making the dataset larger.

## 4. Graph-Based PPI Networks (Later)
- Build a network from protein embeddings:
  - Measure distance in the embedding space.
  - If the distance between two proteins is below a threshold, create a graph edge.
- Develop a large protein dataset and convert it into a graph-based PPI network.
- Analyze network connectivity:
  - Intra-cluster average linkage.
  - Inter-cluster linkage.

## 5. Graph Neural Networks (GNNs) for PPI Prediction (Later)
- Build a GNN-based model to predict protein-protein interactions.
- Train the model to determine if two proteins are linked.
- Investigate different graph-based approaches for interaction prediction.
- **LLM Input:** Use protein sequences to generate embeddings for the GNN.

## 6. Next Steps
1. Identify the best protein interaction dataset for validation.
2. Determine the best information to scrape from UniProt to create the embeddings:
   - GO annotations
   - Others
   - Check my existing ones
3. Scrape data from UniProt & other relevant protein catalogs and create a dataset.
4. Determine if I host the dataset on Supabase or something else.
5. Experiment with different LLM embeddings.
6. Experiment with different LLMs (ChatGPT, Gemini, DeepSeek, etc.) for embeddings.
7. Create a dataset for embeddings (possibly multiple datasets for multiple styles of embeddings).
8. Experiment with different clustering algorithms.
9. Run clustering and create clustered datasets.
10. Make a graphical representation of these datasets.
11. Compare clustering with other clustering results.
12. Evaluate findings.
13. See later steps.

