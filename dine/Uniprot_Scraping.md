# Enhanced UniProt Brain Protein CSV Extractor

## Overview

This tool extracts information about human proteins associated with brain-related disorders directly from the UniProt database and saves it to a single CSV file. It's designed to create datasets for research in neuroscience, bioinformatics, and related fields.

## Data Fields

Each protein entry includes:

### Basic Information
- `accession`: UniProt accession number
- `entry_name`: UniProt entry name
- `protein_name`: Full recommended protein name
- `gene_name`: Primary gene name
- `gene_synonyms`: Alternative gene names
- `organism`: Source organism (Homo sapiens)
- `review_status`: Swiss-Prot review status

### Sequence Data
- `sequence_length`: Length of the protein in amino acids
- `sequence`: Complete amino acid sequence
- `sequence_mass`: Molecular weight of the protein

### Functional Annotations
- `go_biological_process`: Gene Ontology biological process terms
- `go_molecular_function`: Gene Ontology molecular function terms
- `go_cellular_component`: Gene Ontology cellular component terms
- `pathways`: Pathway associations

### Structural Features
- `domains_motifs`: Protein domains, motifs, regions, and transmembrane segments
- `ptms`: Post-translational modifications
- `isoforms`: Alternative protein isoforms

### Disease Associations
- `diseases`: Detailed disease associations with descriptions
- `disease_types`: Simplified disease names (e.g., "Alzheimer disease")

### Interaction Data
- `ppi_data`: Protein-protein interaction data

### External References
- `xref_PDB`: Protein Data Bank structures
- `xref_AlphaFoldDB`: AlphaFold structure predictions
- `xref_Pfam`: Protein family information
- `xref_InterPro`: Protein domains and families
- `xref_KEGG`: KEGG pathway database links
- `xref_Reactome`: Reactome pathway database links
- `xref_GeneID`: NCBI Gene IDs
- `xref_OMIM`: Online Mendelian Inheritance in Man disease links
- `other_xrefs`: Additional cross-references

## Example Output

The CSV file will have one row per protein with columns for all extracted fields. Multiple values in a single field (such as multiple domains or GO terms) are separated by pipe (`|`) characters.

## Notes

- The script uses the UniProt REST API and may be subject to rate limiting
- Progress is saved every 5 proteins to prevent data loss
- The default query targets proteins associated with:
  - Common neurological disorders (Alzheimer's, Parkinson's, etc.)
  - Neurotransmission and synaptic function
  - Neuronal signaling and brain development

## Customization

You can modify the script to:
- Add additional query terms in the `construct_queries()` method
- Extract additional data fields in the `extract_protein_info()` method
- Change the progress saving frequency or output format
