# Comprehensive Protein Dataset Notes

## 1. Protein-Protein Interaction (PPI) Networks

### BioGRID
- **Data Content**: Curated repository of protein and genetic interactions across multiple organisms
- **Size**: >1.7 million interactions from 70,000+ publications
- **Format**: Tab-delimited, PSI-MI XML
- **Columns/Fields**:
  - Unique identifiers (gene symbols, Entrez IDs, UniProt IDs)
  - Interactor A and B identifiers
  - Experimental method
  - Publication references (PubMed IDs)
  - Interaction type
  - Source database
  - Confidence/evidence scores
- **Access**: REST API available for programmatic access
- **Validation Strategy**: 
  - Use as ground truth to validate if proteins that cluster together in embeddings have known physical interactions
  - Calculate enrichment of known BioGRID interactions within your predicted clusters compared to random expectation
  - For each cluster, compute the ratio of protein pairs with BioGRID evidence vs. total possible pairs in cluster
  - Compare edge density in your embedding neighborhood to BioGRID network subgraphs

### IntAct
- **Data Content**: Open-source molecular interaction database with curated protein interaction data
- **Size**: >1 million curated human binary PPIs
- **Format**: PSI-MI TAB/XML, JSON
- **Columns/Fields**:
  - UniProt accession IDs for interacting proteins
  - Interaction detection method
  - Publication reference
  - Interaction type
  - Confidence scores (IMEx evidence scores)
  - Cross-references to other databases
- **Access**: Web interface and API available
- **Validation Strategy**:
  - Filter for high-confidence interactions (e.g., those with multiple experimental validations)
  - Check if protein pairs identified as similar in your embeddings match IntAct high-confidence pairs
  - Use IntAct interaction scores to validate distance metrics in your embedding space
  - Evaluate correlation between embedding similarity scores and IntAct confidence scores

### STRING
- **Data Content**: Database of known and predicted protein associations including direct and indirect interactions
- **Size**: ~19.4 million human interactions
- **Format**: TSV edge lists with confidence scores
- **Columns/Fields**:
  - STRING internal IDs (mapped to Ensembl or UniProt)
  - Protein A and B identifiers
  - Combined confidence score (0-1)
  - Separate scores by evidence type:
    - Experimental evidence
    - Database evidence
    - Text mining evidence
    - Co-expression evidence
    - Neighborhood evidence
    - Fusion evidence
    - Co-occurrence evidence
- **Access**: Web interface and REST API
- **Validation Strategy**:
  - Filter by confidence score >0.7 to focus on high-quality interactions
  - Compare pairwise embedding distances to STRING combined scores
  - Calculate Spearman correlation between embedding similarity and STRING confidence
  - Evaluate whether specific evidence types (e.g., co-expression vs experimental) correlate better with your embedding similarities
  - Create precision-recall curves using STRING scores as ground truth for predicted interactions

### DIP (Database of Interacting Proteins)
- **Data Content**: Curated database of experimentally determined PPIs
- **Format**: PSI-MI XML (HUPO MIF) and tab-delimited text
- **Columns/Fields**:
  - Database-specific IDs mapped to UniProt or RefSeq
  - Experimental method
  - Publication references
  - Detection method classifications
- **License**: CC license, freely available
- **Validation Strategy**:
  - Use as independent validation set after training on other PPI datasets
  - Check overlap between DIP interactions and clustering results
  - Calculate precision/recall of your predicted protein-protein associations against DIP

### HPRD (Human Protein Reference Database)
- **Data Content**: Curated human protein information including PPIs, post-translational modifications, and tissue expression
- **Size**: >36,500 binary protein-protein interactions among 25,000 human proteins
- **Format**: Tab-delimited or XML
- **Columns/Fields**:
  - Gene symbols and HPRD IDs (cross-linked to UniProt and OMIM)
  - Interaction type
  - Experimental evidence (in vitro, in vivo, Y2H)
  - PubMed references
- **Note**: Static dataset (not updated since 2007)
- **Validation Strategy**:
  - Use as historical benchmark for well-established interactions
  - Validate core protein complexes in your clustering results
  - Check if your embeddings capture well-known protein complexes from HPRD

### Human Reference Interactome (HuRI)
- **Data Content**: Large-scale human binary interactome from high-throughput yeast two-hybrid screens
- **Size**: ~52,000 novel human PPIs, network of ~8,000 proteins
- **Format**: Edge lists
- **Columns/Fields**:
  - Gene or UniProt identifiers for both interactors
  - Publication references
  - Experimental evidence (primarily Y2H)
- **Access**: Via BioGRID and Human Interactome Atlas website
- **Validation Strategy**:
  - Test if binary Y2H interactions are captured in your embedding space
  - Compare against other PPI datasets to validate consistency across different experimental methods
  - Use for novel interaction discovery validation if your embeddings suggest interactions not in older databases

### BioPlex (Biophysical Interactions of ORFeome)
- **Data Content**: Proteomics-derived human interactome from affinity-purification mass spectrometry (AP-MS)
- **Size**: BioPlex 3.0 covers ~120,000 PPIs among ~15,000 proteins
- **Format**: CSV interaction lists
- **Columns/Fields**:
  - Gene symbols and UniProt IDs
  - Confidence scores
  - Bait-prey pairs (for directed interactions)
  - Cell line information
- **Access**: BioPlex portal (Harvard Medical School)
- **Validation Strategy**:
  - Compare protein complex memberships between BioPlex and your embedding clusters
  - Use cell-line specific interactions to validate tissue-specific predictions
  - Check if proteins that form complexes in BioPlex have similar embedding vectors
  - Evaluate whether proteins that interact with the same baits have similar embeddings

### HIPPIE (Human Integrated Protein-Protein Interaction Reference)
- **Data Content**: Integrated human PPI network with confidence scores from multiple sources
- **Format**: Tab-delimited edge list
- **Columns/Fields**:
  - Ensembl or UniProt IDs for proteins
  - Normalized confidence score (0-1)
  - Evidence sources
  - Experiment types
- **Access**: HIPPIE web interface
- **Validation Strategy**:
  - Use as meta-validation resource since it integrates multiple databases
  - Correlate HIPPIE confidence scores with distances in embedding space
  - Compare clustering performance against HIPPIE's confidence-filtered networks

## 2. Gene/Disease Association Datasets

### OMIM (Online Mendelian Inheritance in Man)
- **Data Content**: Comprehensive catalog of human genes and genetic disorders
- **Format**: Structured text, API access
- **Key Resources**: Morbid Map (gene-disease pairs)
- **Columns/Fields**:
  - Gene symbols and Entrez IDs
  - OMIM IDs for genes and diseases
  - Disease names
  - Inheritance pattern
  - Cytogenetic location
  - Cross-references (UniProt, Ensembl)
- **Access**: OMIM website (free for browsing), licensed downloads, API
- **Validation Strategy**:
  - Use Morbid Map to validate if clustered proteins share inherited disorder associations
  - Check if proteins with similar embeddings are linked to similar Mendelian disorders
  - Calculate disease term enrichment within clusters
  - Evaluate if embedding distances correlate with shared disease phenotypes

### DisGeNET
- **Data Content**: Platform integrating gene-disease associations from multiple sources
- **Size**: >2 million gene-disease associations involving ~30k genes and ~42k diseases/phenotypes
- **Format**: CSV/TSV
- **Columns/Fields**:
  - NCBI Gene IDs (with mappings to UniProt, Ensembl)
  - Disease IDs (MeSH, UMLS, OMIM)
  - Association score (based on evidence)
  - Evidence sources
  - Evidence level (curated, animal model, literature)
  - PubMed IDs
- **Access**: Downloads, REST API, SPARQL endpoint
- **Validation Strategy**:
  - Match UniProt genes to diseases and evaluate if clustered proteins link to the same diseases
  - Filter by association score thresholds for different confidence levels
  - Compute disease semantic similarity between proteins and compare to embedding similarity
  - Test if proteins sharing rare disease associations cluster together
  - Use for cluster annotation enrichment analysis

### Orphadata (Orphanet)
- **Data Content**: Datasets focused on rare diseases and associated genes
- **Format**: CSV/XML
- **Columns/Fields**:
  - Orpha Codes (Orphanet IDs) for diseases
  - Gene symbols/Entrez IDs
  - Type of gene-disease relationship
  - Status of the association (assessed)
- **Access**: Bulk downloads under CC-BY license, API available
- **Validation Strategy**:
  - Analyze if rare disease genes cluster meaningfully in your embedding space
  - Validate if proteins involved in similar rare diseases have similar vector representations
  - Test enrichment of Orphanet disease classes in your clusters

### CTD (Comparative Toxicogenomics Database)
- **Data Content**: Curated database linking genes, chemicals, and diseases
- **Size**: >5,700 curated gene-disease relationships
- **Format**: CSV
- **Columns/Fields**:
  - Official gene symbols/NCBI IDs
  - Disease terms (MeSH)
  - Association context (marker, therapeutic)
  - References
  - Cross-references (OMIM, DO)
- **Access**: CTD portal downloads
- **Validation Strategy**:
  - Use direct gene-disease associations to validate disease-relevant clusters
  - Compare embedding clusters to chemical-mediated gene sets
  - Validate if genes involved in similar toxicological responses have similar embeddings
  - Test if your embedding can predict gene-disease relationships through nearest neighbors

### ClinGen/GenCC
- **Data Content**: Validated gene-disease relationship data for Mendelian conditions
- **Format**: Structured database
- **Columns/Fields**:
  - HGNC gene symbols/Entrez IDs
  - Disease identifiers (OMIM, Orphanet)
  - Evidence level ratings (definitive, strong, limited)
  - Mechanism of disease
  - MOI (Mode of Inheritance)
- **Access**: ClinGen website and API
- **Validation Strategy**:
  - Use as high-confidence benchmark for Mendelian disease associations
  - Filter by evidence levels (definitive/strong) for strongest validation
  - Test if genes with similar pathogenic mechanisms have similar embeddings

### Open Targets Platform
- **Data Content**: Platform integrating gene-disease association evidence
- **Format**: JSON, API access
- **Columns/Fields**:
  - Ensembl gene IDs
  - EFO disease terms
  - Association scores by evidence type:
    - Genetic associations
    - Somatic mutations
    - Drugs
    - Pathways
    - RNA expression
    - Animal models
    - Literature
  - Overall association score
- **Access**: Web interface, REST API, bulk downloads
- **Validation Strategy**:
  - Use overall and evidence-specific scores to validate disease relevance of protein clusters
  - Compare embedding similarity with Open Targets association score for same diseases
  - Test if therapeutic target predictions from embeddings match Open Targets evidence

## 3. Protein Functional Annotation and Expression Datasets

### UniProtKB (Swiss-Prot/TrEMBL) - Human
- **Data Content**: Comprehensive protein annotations including function, domains, localization, and disease
- **Format**: TSV or XML
- **Columns/Fields**:
  - UniProt accessions (primary identifiers)
  - Protein name and gene names
  - Function descriptions
  - GO annotations (BP, MF, CC)
  - Pfam/InterPro domains
  - Subcellular location
  - Pathway involvement
  - Disease associations
  - Cross-references (OMIM, PDB, RefSeq)
- **Access**: Downloads, REST API, SPARQL endpoint
- **Validation Strategy**:
  - Extract GO annotations for functional enrichment analysis of clusters
  - Compare domain composition between proteins in same cluster
  - Validate if proteins with similar functions have similar embeddings
  - Use UniProt subcellular localization to check if co-localized proteins cluster together
  - Test if embedding distances correlate with shared GO terms or protein family membership

### Gene Ontology Annotations (GOA Human)
- **Data Content**: GO term assignments to human gene products
- **Format**: GAF (Gene Association Format) or TSV
- **Columns/Fields**:
  - UniProt ID (or ENSEMBL/Entrez)
  - GO terms (Molecular Function, Biological Process, Cellular Component)
  - Evidence codes (experimental or computational)
  - Reference
  - Assigned_by (source)
- **Access**: EBI downloads, updated frequently
- **Validation Strategy**:
  - Perform GO term enrichment analysis on protein clusters
  - Calculate semantic similarity between proteins using GO annotations and compare to embedding distance
  - Evaluate embedding quality by testing if proteins sharing GO terms cluster together
  - Use information content of GO terms to weight similarity calculations
  - Filter by evidence codes (e.g., EXP, IDA) for higher confidence validations

### InterPro
- **Data Content**: Integrated resource for protein families, domains, and functional sites
- **Format**: XML/TSV
- **Columns/Fields**:
  - UniProt proteins to domain/family IDs
  - Domain boundaries (start/end positions)
  - Domain/family descriptions
  - GO terms associated with domains
- **Access**: Downloadable (XML/TSV) and queryable via REST API
- **Validation Strategy**:
  - Validate shared domains between related proteins in clusters
  - Check if proteins belonging to the same family have similar embeddings
  - Calculate Jaccard similarity of domain composition and compare to embedding distance
  - Use domain architecture as an independent feature for validating protein relationships

### Human Protein Atlas (HPA)
- **Data Content**: Protein expression and localization across human tissues and cells
- **Size**: Covers >10 million immunohistochemistry images
- **Format**: CSV for expression levels
- **Columns/Fields**:
  - Ensembl/UniProt identifiers
  - Tissue expression values (NX values)
  - Qualitative antibody staining (None, Low, Medium, High)
  - Subcellular localization classifications
  - RNA tissue specificity
  - Cell line expression
- **Access**: Web portal and API
- **Validation Strategy**:
  - Validate clusters by co-expression across brain regions or subcellular localization
  - Test if proteins with similar tissue distribution patterns have similar embeddings
  - Use subcellular localization to validate if co-localized proteins cluster together
  - Compare brain-specific expression patterns with neurological disease gene clusters
  - Calculate tissue expression correlation and compare with embedding similarity

### GTEx (Genotype-Tissue Expression)
- **Data Content**: Reference for baseline gene expression across dozens of human tissues
- **Size**: ~17,000 samples from 54 tissue sites
- **Format**: Expression matrices (gene x tissue)
- **Columns/Fields**:
  - Ensembl gene IDs
  - Gene-level expression (TPM/FPKM)
  - Tissue types (54 distinct tissues)
  - Sample metadata
- **Access**: GTEx Portal, downloadable processed expression matrices
- **Validation Strategy**:
  - Check if genes for clustered proteins show tissue-specific expression correlation
  - Calculate correlation of expression profiles between genes and compare to embedding distances
  - Identify tissue-specific gene modules and validate against embedding clusters
  - Test if brain region-specific genes form coherent clusters in embedding space
  - Use as validation for tissue-specific protein function predictions

### Expression Atlas (EMBL-EBI)
- **Data Content**: Open repository of gene expression under various conditions
- **Format**: Tab-delimited matrices
- **Columns/Fields**:
  - Ensembl gene ID
  - Expression values (TPM)
  - Tissue/cell types
  - Experimental conditions
- **Access**: HTTP REST API and bulk downloads
- **Validation Strategy**:
  - Quickly find if a gene is highly expressed in certain tissues or cell types
  - Validate tissue-specific expression patterns against protein clusters
  - Test if proteins with similar expression profiles cluster together
  - Compare disease-relevant expression changes to embedding-based groupings

### Bgee
- **Data Content**: Curated expression from multiple sources across species and developmental stages
- **Format**: Tab-delimited files
- **Columns/Fields**:
  - Ensembl genes
  - Expression presence/absence calls
  - Expression scores
  - Anatomical entity (tissue/organ)
  - Developmental stage
  - Evidence type
- **Validation Strategy**:
  - Complement GTEx/HPA for cross-database validation of expression patterns
  - Validate developmental stage-specific expression against protein clusters
  - Test if proteins expressed during similar developmental windows have similar embeddings

### ProteomicsDB
- **Data Content**: Human proteome database with protein expression across tissues
- **Format**: Protein-tissue abundance matrix
- **Columns/Fields**:
  - UniProt protein IDs
  - Tissues as columns
  - Relative protein abundance values
  - Detection confidence metrics
- **Access**: Bulk download available
- **Validation Strategy**:
  - Validate if proteins are co-expressed or detected in tissues of interest
  - Compare protein abundance correlation with embedding similarity
  - Cross-validate RNA expression patterns from GTEx with protein-level detection
  - Test if proteins with similar abundance profiles have similar embeddings

### PhosphoSitePlus
- **Data Content**: Resource for post-translational modifications
- **Format**: TXT/CSV
- **Columns/Fields**:
  - UniProt ID
  - Modification sites (position)
  - Modification types (phosphorylation, acetylation, etc.)
  - Disease links
  - Site functions
  - Upstream kinases
- **Validation Strategy**:
  - Check if proteins sharing pathway-relevant modifications have similar embeddings
  - Validate if proteins with similar post-translational regulation patterns cluster together
  - Test enrichment of common PTM sites or kinase targets within protein clusters

## 4. Disease-Specific Datasets (Focus: Brain Disorders)

### ROSMAP (Religious Orders Study and Memory and Aging Project)
- **Data Content**: Multi-omic study of Alzheimer's disease and aging from postmortem brains
- **Size**: ~600 RNA-seq samples + proteomics
- **Format**: Expression matrices
- **Columns/Fields**:
  - Ensembl IDs (RNA-seq)
  - UniProt IDs (proteomics)
  - Expression/abundance values
  - Clinical metadata:
    - Diagnosis (AD/Control)
    - Cognitive scores
    - Neuropathology measures
    - Demographics
- **Access**: AMP-AD Knowledge Portal (Synapse)
- **Validation Strategy**:
  - Test if clusters show altered expression in Alzheimer's disease brains
  - Validate disease associations by correlating protein abundances with AD pathology
  - Compare differential expression patterns with protein clusters
  - Cross-validate RNA and protein-level changes in same dataset
  - Test if co-expressed proteins in AD also cluster in embedding space

### AMP-AD Knowledge Portal
- **Data Content**: Repository of multiple Alzheimer's disease studies
- **Key Datasets**:
  - MSBB (Mount Sinai Brain Bank): RNA-seq from multiple brain regions (~300 individuals)
  - Mayo Clinic Brain RNAseq: Temporal cortex and cerebellum
  - Proteomics datasets from multiple centers
- **Format**: Expression matrices, clinical data
- **Columns/Fields**:
  - Ensembl gene IDs (RNA)
  - UniProt IDs (proteins)
  - Expression/abundance values
  - Extensive clinical metadata
- **Access**: Synapse IDs, API
- **Validation Strategy**:
  - Check if protein's expression correlates with AD pathology
  - Validate if coexpression networks are perturbed in Alzheimer's
  - Test region-specific expression changes in disease
  - Correlate protein network modules with clinical variables
  - Compare clusters across multiple independent cohorts for robust validation

### CommonMind Consortium (CMC)
- **Data Content**: Transcriptomic and epigenomic data for psychiatric disorders
- **Size**: RNA-seq of DLPFC from ~1,000 individuals (353 schizophrenia, 120 bipolar, 522 controls)
- **Format**: Gene-level counts/FPKM
- **Columns/Fields**:
  - Gencode/Ensembl gene IDs
  - Expression values
  - Diagnosis
  - Medication status
  - Demographics
  - Technical covariates
- **Access**: CMC Knowledge Portal (Synapse)
- **Validation Strategy**:
  - Compare clusters against schizophrenia/bipolar vs. control differential expression
  - Validate if protein's mRNA is significantly dysregulated in psychiatric disorders
  - Test if co-expression modules are enriched for disease risk genes
  - Calculate correlation between disease effect sizes and embedding clusters
  - Evaluate if proteins with similar embeddings show coordinated expression changes in disease

### PsychENCODE Consortium Data
- **Data Content**: Integrated neuropsychiatric disorder datasets (autism, schizophrenia, bipolar)
- **Size**: Brain RNA-seq from 1,695 individuals
- **Format**: Unified processed expression matrix
- **Columns/Fields**:
  - Entrez or Ensembl IDs with gene symbols
  - Expression values
  - Splicing information
  - Diagnostic categories
  - Brain region
  - Developmental stage
- **Access**: PsychENCODE portal, NIH dbGaP
- **Validation Strategy**:
  - Query if protein-coding genes are differentially expressed in disorders
  - Check if alternatively spliced genes in disorders cluster together
  - Test if disease-relevant co-expression modules match embedding clusters
  - Validate cross-disorder gene expression patterns against protein clusters
  - Compare developmental trajectories of gene expression with embedding relationships

### Brain-Specific Single-Cell Datasets
- **Data Content**: Single-cell or single-nucleus RNA-seq from normal and diseased brains
- **Examples**:
  - Alzheimer's single-nucleus RNA-seq (80k nuclei, Mathys et al.)
  - ASD snRNA-seq from prefrontal cortex
  - Human Cell Atlas brain data
- **Format**: Count matrices (cells x genes)
- **Columns/Fields**:
  - Gene IDs (typically Ensembl)
  - Cell barcodes
  - UMI counts
  - Cell type annotations
  - Disease status
- **Access**: GEO, Synapse, HCA
- **Validation Strategy**:
  - Identify cell-type specific expression patterns of protein-coding genes
  - Dissect which cell types show dysregulation of clustered proteins
  - Test if proteins with similar embeddings are expressed in same cell types
  - Validate cell-type specific disease alterations against protein functional clusters
  - Compare single-cell co-expression networks with protein embedding spaces
