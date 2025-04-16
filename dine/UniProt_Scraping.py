
import requests
import pandas as pd
import time
import sys
import re
from collections import defaultdict


class BrainProteinExtractor:
    """Extract brain-related protein data from UniProt database."""

    def __init__(self, output_file="brain_proteins.csv"):
        # API endpoints
        self.base_url = "https://rest.uniprot.org/uniprotkb/search"
        self.entry_url = "https://rest.uniprot.org/uniprotkb/"

        # Request parameters
        self.batch_size = 50  # Reduced batch size to avoid API issues
        self.retry_attempts = 3
        self.retry_delay = 3

        # Output settings
        self.output_file = output_file

    def get_queries(self):
        """Create a list of queries for brain-related proteins."""
        # Human proteins (reviewed entries only)
        base = 'organism_id:9606 AND reviewed:true'

        # Use simpler, more API-friendly queries
        queries = [
            f"{base} AND (gene:APP OR gene:PSEN1 OR gene:PSEN2 OR gene:APOE)",  # Alzheimer's
            f"{base} AND (gene:SNCA OR gene:PARK2 OR gene:LRRK2)",  # Parkinson's
            f"{base} AND (gene:HTT)",  # Huntington's
            f"{base} AND (gene:SOD1 OR gene:C9orf72 OR gene:FUS OR gene:TARDBP)",  # ALS
            f"{base} AND alzheimer",  # Alzheimer's disease
            f"{base} AND parkinson",  # Parkinson's disease
            f"{base} AND huntington",  # Huntington's disease
            f"{base} AND epilepsy",  # Epilepsy
            f"{base} AND schizophrenia",  # Schizophrenia
            f"{base} AND \"multiple sclerosis\"",  # MS
            f"{base} AND autism",  # Autism
            f"{base} AND cerebral",  # Other cerebral conditions
            # Specific protein types
            f"{base} AND keyword:\"Ion channel\"",
            f"{base} AND keyword:\"Receptor\"",
            f"{base} AND keyword:\"Neurotransmitter\"",
            f"{base} AND keyword:\"Transporter\"",
            f"{base} AND keyword:\"Neurogenesis\"",
            f"{base} AND gaba",
            f"{base} AND glutamate",
            f"{base} AND dopamine",
            f"{base} AND serotonin",
            f"{base} AND acetylcholine",
            f"{base} AND annotation:(type:\"tissue specificity\" \"brain\")",
            f"{base} AND annotation:(type:\"developmental stage\" \"brain\")",
            f"{base} AND annotation:(type:\"subcellular location\" \"synapse\")",
        ]

        return queries

    def process_protein(self, entry):
        """Extract and format protein information from UniProt entry."""
        # Basic protein info
        protein = {
            'accession': entry.get('primaryAccession', ''),
            'entry_name': entry.get('id', ''),
            'protein_name': 'Unknown',
            'gene_name': '',
            'gene_synonyms': '',
            'organism': 'Homo sapiens',
            'sequence_length': 0,
            'sequence_mass': 0,
            'sequence': '',
            'review_status': entry.get('entryType', '') == 'UniProtKB/Swiss-Prot',
            'annotation_score': entry.get('annotationScore', 0)
        }

        # Extract protein name
        if 'proteinDescription' in entry and 'recommendedName' in entry['proteinDescription']:
            name_obj = entry['proteinDescription']['recommendedName']
            if 'fullName' in name_obj:
                protein['protein_name'] = name_obj['fullName'].get('value', 'Unknown')

        # Extract gene info
        if 'genes' in entry and entry['genes']:
            # Primary gene name
            gene = entry['genes'][0]
            if 'geneName' in gene:
                protein['gene_name'] = gene['geneName'].get('value', '')

            # Synonyms
            synonyms = []
            for gene in entry['genes']:
                for syn in gene.get('synonyms', []):
                    if 'value' in syn:
                        synonyms.append(syn['value'])
            protein['gene_synonyms'] = '|'.join(synonyms)

        # Extract sequence info
        if 'sequence' in entry:
            protein['sequence_length'] = entry['sequence'].get('length', 0)
            protein['sequence'] = entry['sequence'].get('value', '')
            protein['sequence_mass'] = entry['sequence'].get('mass', 0)

        # Process GO terms
        go_terms = {
            'biological_process': [],
            'molecular_function': [],
            'cellular_component': []
        }

        for ref in entry.get('dbReferences', []):
            if ref.get('type') == 'GO':
                go_id = ref.get('id', '')
                props = ref.get('properties', {})
                term = props.get('term', '')
                category = props.get('category', '')

                if category in go_terms:
                    go_terms[category].append(f"{go_id}:{term}")

        # Add GO terms to protein
        for category in go_terms:
            protein[f"go_{category}"] = '|'.join(go_terms[category])

        # Extract domains and structural features
        domains = []
        for feature in entry.get('features', []):
            if feature.get('type') in ['DOMAIN', 'MOTIF', 'REGION', 'TRANSMEM']:
                desc = feature.get('description', '')
                start = feature.get('location', {}).get('start', {}).get('value', '')
                end = feature.get('location', {}).get('end', {}).get('value', '')
                domains.append(f"{desc}({start}-{end})")

        protein['domains_motifs'] = '|'.join(domains)

        # Extract post-translational modifications
        ptms = []
        for feature in entry.get('features', []):
            if feature.get('type') in ['MOD_RES', 'LIPID', 'CARBOHYD', 'DISULFID']:
                desc = feature.get('description', '')
                pos = feature.get('location', {}).get('position', {}).get('value', '')
                if pos:
                    ptms.append(f"{desc}({pos})")
                else:
                    start = feature.get('location', {}).get('start', {}).get('value', '')
                    end = feature.get('location', {}).get('end', {}).get('value', '')
                    ptms.append(f"{desc}({start}-{end})")

        protein['ptms'] = '|'.join(ptms)

        # Extract disease information
        diseases = []
        disease_types = []

        for comment in entry.get('comments', []):
            if comment.get('commentType') == 'DISEASE':
                disease = comment.get('disease', {})
                disease_id = disease.get('diseaseId', '')
                disease_desc = disease.get('description', '')
                diseases.append(f"{disease_id}:{disease_desc}")

                # Extract just the disease name
                if disease_id:
                    # Clean up disease name (remove accession numbers, etc.)
                    disease_name = re.sub(r'\[.*?\]', '', disease_id).strip()
                    disease_types.append(disease_name)

        protein['diseases'] = '|'.join(diseases)
        protein['disease_types'] = '|'.join(disease_types)

        # Extract protein-protein interactions
        ppis = []
        for comment in entry.get('comments', []):
            if comment.get('commentType') == 'INTERACTION':
                for interaction in comment.get('interactions', []):
                    partner = interaction.get('interactantTwo', {})
                    partner_id = partner.get('uniProtKBAccession', '')
                    partner_name = partner.get('geneName', '')
                    if partner_id:
                        ppis.append(f"{partner_id}({partner_name})")

        protein['ppi_data'] = '|'.join(ppis)

        # Extract pathway information
        pathways = []
        for comment in entry.get('comments', []):
            if comment.get('commentType') == 'PATHWAY':
                pathway = comment.get('pathway', '')
                if pathway:
                    pathways.append(pathway)

        protein['pathways'] = '|'.join(pathways)

        # Extract isoform information
        isoforms = []
        for comment in entry.get('comments', []):
            if comment.get('commentType') == 'ALTERNATIVE PRODUCTS':
                for isoform in comment.get('isoforms', []):
                    isoform_name = isoform.get('name', {}).get('value', '')
                    isoform_ids = isoform.get('isoformIds', [])
                    isoform_note = isoform.get('note', {}).get('value', '') if 'note' in isoform else ''
                    isoforms.append(f"{isoform_name}:{','.join(isoform_ids)}:{isoform_note}")

        protein['isoforms'] = '|'.join(isoforms)

        # Process cross-references
        db_refs = defaultdict(list)
        for ref in entry.get('dbReferences', []):
            ref_type = ref.get('type', '')
            ref_id = ref.get('id', '')

            if ref_type and ref_id and ref_type not in ['GO']:
                db_refs[ref_type].append(ref_id)

        # Add important databases as separate columns
        key_dbs = ['PDB', 'AlphaFoldDB', 'Pfam', 'InterPro', 'KEGG', 'Reactome', 'GeneID', 'OMIM']
        for db in key_dbs:
            protein[f"xref_{db}"] = '|'.join(db_refs.get(db, []))

        # Add remaining references
        other_refs = []
        for db, ids in db_refs.items():
            if db not in key_dbs:
                other_refs.append(f"{db}:{','.join(ids)}")

        protein['other_xrefs'] = '|'.join(other_refs)

        return protein

    def get_protein_details(self, accession):
        """Retrieve detailed protein data from UniProt."""
        if not accession:
            print(f"  Error: Missing accession number")
            return None

        url = f"{self.entry_url}{accession}"
        params = {"format": "json"}

        print(f"  Fetching data for {accession}")
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                print(f"  Successfully retrieved {accession}")
                return response.json()
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    print(f"  Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"  Failed to retrieve {accession}")
                    return None

    def save_progress(self, proteins):
        """Save current progress to CSV file."""
        if not proteins:
            return

        df = pd.DataFrame(proteins)
        # Explicitly ensure column names are included
        df.to_csv(self.output_file, index=False, header=True)
        print(f"  Progress saved: {len(proteins)} proteins in {self.output_file}")

    def run(self, limit=None):
        """Execute the full extraction process."""
        # Get search queries
        queries = self.get_queries()

        # Track proteins we've processed
        protein_ids = set()
        proteins = []

        # Show extraction mode
        if limit:
            print(f"Extracting up to {limit} brain-related proteins...")
        else:
            print("Extracting ALL brain-related proteins (unlimited)...")

        # Process each query
        print(f"Running {len(queries)} search queries...")
        for i, query in enumerate(queries):
            # Check if we've reached the limit
            if limit and len(proteins) >= limit:
                break

            print(f"\nQuery {i + 1}/{len(queries)}")
            print(f"Terms: {query}")

            # Prepare search parameters with minimal fields
            params = {
                "query": query,
                "format": "json",
                "size": self.batch_size
            }

            # Paginate through results
            cursor = "*"
            while cursor:
                # Check if we've reached the limit
                if limit and len(proteins) >= limit:
                    break

                # Update cursor for pagination
                if cursor != "*":
                    params["cursor"] = cursor

                # Try the request
                success = False
                for attempt in range(self.retry_attempts):
                    try:
                        print(f"  Requesting page: {cursor if cursor != '*' else 'initial'}")

                        # Make the request
                        response = requests.get(self.base_url, params=params)
                        response.raise_for_status()
                        data = response.json()

                        # Get results
                        results = data.get("results", [])
                        processed = 0

                        # Process each new protein
                        for entry in results:
                            # Check limit again
                            if limit and len(proteins) >= limit:
                                break

                            # Get protein ID - handle both field names
                            acc = entry.get("primaryAccession") or entry.get("accession")
                            if acc and acc not in protein_ids:
                                # Add to tracking set
                                protein_ids.add(acc)

                                # Get full details and process
                                details = self.get_protein_details(acc)
                                if details:
                                    protein_data = self.process_protein(details)
                                    proteins.append(protein_data)
                                    processed += 1

                                    # Save progress more frequently for large datasets
                                    if len(proteins) % 10 == 0:
                                        self.save_progress(proteins)

                        print(f"  Processed {processed} new proteins. Total: {len(proteins)}")

                        # Check for more pages
                        cursor = data.get("nextCursor")
                        if not cursor or not results:
                            cursor = None

                        success = True
                        break

                    except Exception as e:
                        print(f"  Error: {str(e)}")
                        if attempt < self.retry_attempts - 1:
                            print(f"  Retrying in {self.retry_delay}s...")
                            time.sleep(self.retry_delay)
                        else:
                            print("  Moving to next query.")
                            cursor = None
                            break

                # Wait between requests
                time.sleep(0.5)

        # Final save
        self.save_progress(proteins)

        # Show completion message
        if limit and len(proteins) >= limit:
            print(f"\nReached limit of {limit} proteins.")
        else:
            print("\nExtracted all matching proteins.")

        print(f"Total proteins extracted: {len(proteins)}")
        print(f"Data saved to {self.output_file}")

        return len(proteins)


# Main execution
if __name__ == "__main__":
    try:
        # Show banner
        print("=" * 80)
        print("Brain Protein Extractor")
        print("Extract human proteins related to brain disorders from UniProt")
        print("=" * 80)

        # Parse command line arguments
        limit = None  # Default is unlimited
        output_file = "brain_proteins.csv"

        # Process limit argument
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            if arg in ['none', 'unlimited', 'all']:
                limit = None
                print("Mode: Unlimited")
            else:
                try:
                    limit = int(sys.argv[1])
                    print(f"Mode: Limited to {limit} proteins")
                except ValueError:
                    print(f"Invalid argument: '{sys.argv[1]}'. Using unlimited mode.")

        # Process filename argument
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
            print(f"Output: {output_file}")

        # Run extraction
        extractor = BrainProteinExtractor(output_file=output_file)
        count = extractor.run(limit=limit)

        print("\nExtraction completed successfully!")
        print(f"Extracted {count} proteins to {output_file}")

    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)