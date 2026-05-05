import os
import subprocess
import pandas as pd
import numpy as np
import pytrec_eval
from collections import defaultdict
from dgeb.tasks import ArchRetrieval, EukRetrieval

# =============================================================================
# DATA LOADING
# =============================================================================
def load_dgeb_task(task_class, task_name):
    print(f"Loading {task_name} ...")
    task = task_class()
    meta = task.metadata

    data_ds = meta.datasets[0].load()
    qrels_ds = meta.datasets[1].load()

    corpus = {row["Entry"]: row["Sequence"] for row in data_ds["train"]}
    queries = {row["Entry"]: row["Sequence"] for row in data_ds["test"]}

    qrels = defaultdict(set)
    splits = list(qrels_ds.values()) if hasattr(qrels_ds, "values") else [qrels_ds]

    for split in splits:
        for row in split:
            qid = str(row["query_id"])
            cid = str(row["corpus_id"])
            if float(row["fuzz_ratio"]) > 0:
                qrels[qid].add(cid)

    print(f"  Corpus  : {len(corpus):,}")
    print(f"  Queries : {len(queries):,}")
    print(f"  Queries with >=1 relevant doc: {len(qrels):,}")
    print(f"  Query/qrel overlap: {len(set(queries.keys()) & set(qrels.keys())):,}")

    return queries, corpus, dict(qrels)

# =============================================================================
# FASTA WRITING
# =============================================================================
def write_fasta(seq_dict, output_path):
    with open(output_path, "w") as f:
        for protein_id, seq in seq_dict.items():
            seq = str(seq).replace("*", "").replace(" ", "").replace("\n", "")
            f.write(f">{protein_id}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

# =============================================================================
# BLAST PARSING
# =============================================================================
BLAST_COLS = [
    "query_id", "corpus_id", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore"
]

def parse_blast_results(path, task_name, top_k):
    if not os.path.exists(path):
        print(f"No BLAST results found at {path}")
        return pd.DataFrame(columns=BLAST_COLS + ["task"])

    df = pd.read_csv(path, sep="\t", names=BLAST_COLS)
    df["task"] = task_name

    df = df.sort_values(
        ["query_id", "evalue", "bitscore"],
        ascending=[True, True, False]
    )

    return df.groupby("query_id").head(top_k).copy()

def blast_to_retrieved_list(blast_df, query_ids, max_k):
    grouped = (
        blast_df
        .groupby("query_id")[["corpus_id", "bitscore"]]
        .apply(lambda x: list(zip(x["corpus_id"], x["bitscore"]))[:max_k])
        .to_dict()
    )

    retrieved_lists = []
    similarity_lists = []

    for qid in query_ids:
        hits = grouped.get(str(qid), [])
        retrieved_lists.append([h[0] for h in hits])
        similarity_lists.append([h[1] for h in hits])

    return retrieved_lists, similarity_lists

# =============================================================================
# EVALUATION
# =============================================================================
def qrels_to_pytrec(qrels):
    return {
        str(qid): {str(cid): 1 for cid in rel_docs}
        for qid, rel_docs in qrels.items()
    }

def evaluate_with_pytrec_eval(query_ids, retrieved_list, similarity_list, qrels, k_values=(5, 10, 50)):
    pytrec_qrels = qrels_to_pytrec(qrels)

    results = {}
    for i, (qid, ranked_ids) in enumerate(zip(query_ids, retrieved_list)):
        sim_scores = similarity_list[i] if similarity_list else [
            float(len(ranked_ids) - rank) for rank in range(len(ranked_ids))
        ]
        results[str(qid)] = {
            str(cid): float(score)
            for cid, score in zip(ranked_ids, sim_scores)
        }

    metric_set = {"recip_rank"}
    for k in k_values:
        metric_set.update({
            f"map_cut.{k}",
            f"ndcg_cut.{k}",
            f"P.{k}",
            f"recall.{k}"
        })

    evaluator = pytrec_eval.RelevanceEvaluator(pytrec_qrels, metric_set)
    scores = evaluator.evaluate(results)

    valid_qids = [str(qid) for qid in query_ids if str(qid) in pytrec_qrels]

    aggregated = {}
    for k in k_values:
        map_scores = []
        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for qid in valid_qids:
            q = scores.get(qid, {})
            map_scores.append(q.get(f"map_cut_{k}", 0.0))
            ndcg_scores.append(q.get(f"ndcg_cut_{k}", 0.0))
            recall_scores.append(q.get(f"recall_{k}", 0.0))
            precision_scores.append(q.get(f"P_{k}", 0.0))

        aggregated[f"MAP@{k}"] = round(np.mean(map_scores), 5)
        aggregated[f"nDCG@{k}"] = round(np.mean(ndcg_scores), 5)
        aggregated[f"Precision@{k}"] = round(np.mean(precision_scores), 5)
        aggregated[f"Recall@{k}"] = round(np.mean(recall_scores), 5)

    mrr_scores = [scores.get(qid, {}).get("recip_rank", 0.0) for qid in valid_qids]
    aggregated["MRR"] = round(np.mean(mrr_scores), 5)
    aggregated["n_queries"] = len(valid_qids)

    return aggregated

# =============================================================================
# MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    BLAST_DIR = "blast_method1"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(BLAST_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # File paths
    arch_query_fasta = os.path.join(BLAST_DIR, "arch_queries.fasta")
    arch_corpus_fasta = os.path.join(BLAST_DIR, "arch_corpus.fasta")
    arch_db = os.path.join(BLAST_DIR, "arch_db")
    arch_out = os.path.join(BLAST_DIR, "blast_arch.tsv")

    euk_query_fasta = os.path.join(BLAST_DIR, "euk_queries.fasta")
    euk_corpus_fasta = os.path.join(BLAST_DIR, "euk_corpus.fasta")
    euk_db = os.path.join(BLAST_DIR, "euk_db")
    euk_out = os.path.join(BLAST_DIR, "blast_euk.tsv")

    # Load data
    arch_queries, arch_corpus, arch_qrels = load_dgeb_task(ArchRetrieval, "Arch Retrieval")
    euk_queries, euk_corpus, euk_qrels = load_dgeb_task(EukRetrieval, "Euk Retrieval")

    print("\nWriting FASTA...")
    write_fasta(arch_queries, arch_query_fasta)
    write_fasta(arch_corpus, arch_corpus_fasta)
    write_fasta(euk_queries, euk_query_fasta)
    write_fasta(euk_corpus, euk_corpus_fasta)

    # Build BLAST DB
    print("\nBuilding BLAST DB...")
    subprocess.run(["makeblastdb", "-in", arch_corpus_fasta, "-dbtype", "prot", "-out", arch_db], check=True)
    subprocess.run(["makeblastdb", "-in", euk_corpus_fasta, "-dbtype", "prot", "-out", euk_db], check=True)

    # Run BLAST
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"

    print("\nRunning BLAST...")
    subprocess.run(["blastp", "-query", arch_query_fasta, "-db", arch_db,
                    "-out", arch_out, "-outfmt", outfmt,
                    "-evalue", "10", "-max_target_seqs", "100"], check=True)

    subprocess.run(["blastp", "-query", euk_query_fasta, "-db", euk_db,
                    "-out", euk_out, "-outfmt", outfmt,
                    "-evalue", "10", "-max_target_seqs", "100"], check=True)

    # Parse
    KS = [5, 10, 50]
    top_k_retrieval = max(KS)

    arch_df = parse_blast_results(arch_out, "arch", top_k_retrieval)
    euk_df = parse_blast_results(euk_out, "euk", top_k_retrieval)

    # Convert to ranking lists
    arch_ret, arch_scores = blast_to_retrieved_list(arch_df, list(arch_queries.keys()), top_k_retrieval)
    euk_ret, euk_scores = blast_to_retrieved_list(euk_df, list(euk_queries.keys()), top_k_retrieval)

    # Evaluate
    print("\nEvaluating...")
    arch_metrics = evaluate_with_pytrec_eval(list(arch_queries.keys()), arch_ret, arch_scores, arch_qrels, KS)
    euk_metrics = evaluate_with_pytrec_eval(list(euk_queries.keys()), euk_ret, euk_scores, euk_qrels, KS)

    results = pd.DataFrame([
        {"Method": "BLASTP", "Task": "Arch Retrieval", **arch_metrics},
        {"Method": "BLASTP", "Task": "Euk Retrieval", **euk_metrics}
    ])

    print("\n=== FINAL RESULTS ===")
    print(results.to_string(index=False))

    results.to_csv(os.path.join(CHECKPOINT_DIR, "blast_results.csv"), index=False)