from blast.run_blast import run_blast
from blast.parse_blast import parse_blast_results
# from kmer.run_kmer import run_kmer

# paths
QUERY = "data/queries.fasta"
CORPUS = "data/corpus.fasta"
BLAST_DB = "data/blast_db"
BLAST_OUT = "data/blast_results.txt"

# ---- METHOD 1: BLAST ----
print("Running BLAST...")
run_blast(QUERY, BLAST_DB, BLAST_OUT)

blast_results = parse_blast_results(BLAST_OUT)



# preview
print("\nBLAST sample:", list(blast_results.items())[:2])