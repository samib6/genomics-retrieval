import subprocess

def run_blast(query_fasta, db_path, output_file):
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db_path,
        "-outfmt", "6 qseqid sseqid evalue bitscore",
        "-max_target_seqs", "5",
        "-out", output_file
    ]

    subprocess.run(cmd, check=True)