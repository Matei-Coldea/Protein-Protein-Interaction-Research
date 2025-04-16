import anthropic
import json
import numpy as np
from supabase import create_client
import csv
import time

# Config
CLAUDE_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"

# Settings
MODEL_NAME = "claude-3-opus-20240229"  # Or use claude-3-sonnet or claude-3-haiku
PROTEIN_TABLE = "proteins"
EMBEDDING_SIZE = 256
OUTPUT_FILE = "protein_embeddings.csv"

# Init Claude client
claude = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


# Helper Functions

def build_protein_description(protein_data):
    # Format protein data as readable text
    return (
        f"Accession: {protein_data.get('accession', '')}. "
        f"Protein: {protein_data.get('protein_name', '')} (Gene: {protein_data.get('gene_name', '')}). "
        f"Organism: {protein_data.get('organism', '')}. "
        f"Domains: {protein_data.get('domains', '')}. "
        f"PTMs: {protein_data.get('ptms', '')}. "
        f"Function (GO): {protein_data.get('go_terms', '')}. "
        f"PPI: {protein_data.get('ppi', '')}. "
        f"Pathways: {protein_data.get('pathways', '')}. "
        f"Disease: {protein_data.get('disease_info', '')}. "
        f"Evidence: {protein_data.get('evidence', '')}."
    )


def create_embedding_prompt(protein_text, dimensions=EMBEDDING_SIZE):
    # Create prompt for Claude
    return f"""
You are given information about a protein from UniProt. The fields include:
- Accession Number
- Protein and Gene Names
- Organism and Taxonomic Data
- Protein Domains and Motifs
- Post-Translational Modifications (PTMs)
- Function and Biological Process (GO terms)
- PPI Annotations
- Pathways and Disease Information
- Evidence Level and External Cross-references

Your task is to generate a {dimensions}-dimensional numerical vector that captures the key biological characteristics of this protein. Proteins with similar functions, sequences, and regulatory features should be mapped to similar vectors.

Please output the vector as a JSON object with the key "embedding" containing an array of {dimensions} decimal numbers. Make sure your response includes ONLY the JSON object and nothing else.

Protein Data: {protein_text}

Output Format:
{{ "embedding": [number1, number2, ..., number{dimensions}] }}
    """.strip()


def get_protein_embedding(protein_data, dimensions=EMBEDDING_SIZE, model=MODEL_NAME):
    # Get embedding from Claude for a protein
    protein_text = build_protein_description(protein_data)
    prompt = create_embedding_prompt(protein_text, dimensions)
    
    protein_id = protein_data.get("accession", "")
    print(f"Getting embedding for {protein_id}...")

    try:
        # Call Claude API
        response = claude.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system="You generate numerical embeddings of proteins based on their biological characteristics.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.content[0].text
        
        # Parse JSON - Claude may return cleaner JSON than OpenAI
        # but we keep the JSON extraction logic for safety
        json_start = answer.find("{")
        json_end = answer.rfind("}") + 1
        
        if json_start == -1 or json_end == 0:
            # If no JSON brackets found, try to use the whole response
            json_str = answer.strip()
        else:
            json_str = answer[json_start:json_end]
        
        result = json.loads(json_str)
        embedding = result.get("embedding")
        
        if not embedding or len(embedding) != dimensions:
            raise ValueError(f"Invalid embedding format: expected {dimensions} dimensions")
            
        return embedding
        
    except Exception as e:
        print(f"Error generating embedding for {protein_id}: {str(e)}")
        print(f"API response: {answer if 'answer' in locals() else 'No response'}")
        return None


# Database Functions

def fetch_proteins_from_database():
    # Get all proteins from Supabase
    print("Connecting to database and fetching proteins...")
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase.table(PROTEIN_TABLE).select("*").execute()
        proteins = response.data
        
        print(f"Retrieved {len(proteins)} proteins successfully")
        return proteins
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []


def save_results_to_csv(embedding_results, filename=OUTPUT_FILE):
    # Save results to CSV
    if not embedding_results:
        print("No embeddings to save")
        return False
        
    print(f"Saving {len(embedding_results)} embeddings to {filename}...")
    
    try:
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["accession", "embedding"])
            writer.writeheader()
            
            for row in embedding_results:
                writer.writerow(row)
                
        print(f"Data saved successfully to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False


# Main Process

def generate_protein_embeddings():
    # Main function to process proteins
    proteins = fetch_proteins_from_database()
    
    if not proteins:
        print("No proteins found. Check your database connection.")
        return
    
    results = []
    total = len(proteins)
    
    for i, protein in enumerate(proteins, 1):
        protein_id = protein.get("accession", "")
        print(f"\n[{i}/{total}] Processing: {protein_id}")
        
        embedding = get_protein_embedding(protein)
        
        if embedding:
            results.append({
                "accession": protein_id,
                "embedding": json.dumps(embedding)
            })
            print(f"Successfully embedded protein {protein_id}")
        else:
            print(f"Failed to generate embedding for {protein_id}")
        
        if i < total:
            print("Pausing briefly...")
            time.sleep(1)
    
    if results:
        save_results_to_csv(results)
        print(f"\nComplete! Generated embeddings for {len(results)}/{total} proteins")
    else:
        print("\nNo embeddings were successfully generated")


# Entry Point

if __name__ == "__main__":
    print("Protein Embedding Generator (Claude)")
    print("===================================")
    generate_protein_embeddings()