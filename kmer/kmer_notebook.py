# =============================================================================
# METHOD 2: k-mer Cosine Similarity Retrieval
# Project: Cross-Domain Protein Retrieval — DGEB Benchmark
# =============================================================================
#
# Verified dataset structure (from our earlier data loading work):
#   tattabio/arch_retrieval  and  tattabio/euk_retrieval
#     train split = bacterial corpus  (columns: Entry, Sequence)
#     test  split = query proteins    (columns: Entry, Sequence)
#   tattabio/arch_retrieval_qrels  and  tattabio/euk_retrieval_qrels
#     only relevant pairs stored (fuzz_ratio = 1.0 for all stored pairs)
#     columns: query_id, corpus_id, fuzz_ratio
# =============================================================================

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# =============================================================================
# DATA LOADING
# Uses same verified loading logic from our earlier work
# =============================================================================

# Pinned revision hashes from dgeb v0.2.0 source
DATASET_CONFIG = {
    'arch': {
        'seqs_path'  : 'tattabio/arch_retrieval',
        'qrels_path' : 'tattabio/arch_retrieval_qrels',
        'label'      : 'Arch Retrieval',
    },
    'euk': {
        'seqs_path'  : 'tattabio/euk_retrieval',
        'qrels_path' : 'tattabio/euk_retrieval_qrels',
        'label'      : 'Euk Retrieval',
    },
}


def load_task(task_key):
    """
    Load queries, corpus and qrels for a retrieval task.

    Returns
    -------
    queries     : dict[str, str]              {Entry: Sequence}
    corpus      : dict[str, str]              {Entry: Sequence}
    qrels       : dict[str, dict[str, int]]   {query_id: {corpus_id: 1}}
    query_ids   : list[str]   ordered list of query Entry IDs
    corpus_ids  : list[str]   ordered list of corpus Entry IDs
    """
    cfg = DATASET_CONFIG[task_key]
    print(f"\nLoading {cfg['label']} ...")

    seqs_ds  = load_dataset(cfg['seqs_path'])
    qrels_ds = load_dataset(cfg['qrels_path'])

    # Corpus = train split, Queries = test split
    corpus  = {row['Entry']: row['Sequence'] for row in seqs_ds['train']}
    queries = {row['Entry']: row['Sequence'] for row in seqs_ds['test']}

    # Ordered ID lists — order must be consistent with matrix rows/cols
    corpus_ids = list(corpus.keys())
    query_ids  = list(queries.keys())

    # Build qrels: all stored pairs are relevant (fuzz_ratio=1.0 verified)
    qrels = defaultdict(dict)

    for split in qrels_ds:
        for row in qrels_ds[split]:
            qrels[str(row['query_id'])][str(row['corpus_id'])] = 1
            
    avg_relevant = sum(len(v) for v in qrels.values()) / len(qrels) if qrels else 0.0

    print(f"  Corpus  : {len(corpus):,}")
    print(f"  Queries : {len(queries):,}")
    print(f"  Relevant pairs: {sum(len(v) for v in qrels.values()):,}")
    print(f"  Avg relevant per query: {avg_relevant:.1f}")

    return queries, corpus, dict(qrels), query_ids, corpus_ids


# =============================================================================
# K-MER ENCODING
# =============================================================================

def get_kmers(sequence, k):
    """
    Convert a protein sequence to a space-separated string of k-mers.
    Used as input to CountVectorizer.

    Example (k=3): 'MKTII' -> 'MKT KTI TII'

    Sequences shorter than k return an empty string
    (will produce a zero vector after vectorization).
    """
    if len(sequence) < k:
        return ""
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1))


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def average_precision_at_k(ranked_ids, relevant_set, k=5):
    """
    Average Precision@k for a single query.

    Parameters
    ----------
    ranked_ids   : list of corpus Entry IDs ranked best-first
    relevant_set : set of corpus Entry IDs that are truly relevant
    k            : cutoff rank

    Returns
    -------
    float
    """
    if not relevant_set:
        return 0.0
    hits, score = 0, 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_set:
            hits  += 1
            score += hits / rank
    return score / min(len(relevant_set), k)


def ndcg_at_k(ranked_ids, relevant_set, k=5):
    """nDCG@k for a single query (binary relevance)."""
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevant_set, k=5):
    """Recall@k for a single query."""
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def evaluate_rankings(query_ids, ranked_id_lists, qrels, k=5):
    """
    Compute MAP@k, nDCG@k, Recall@k macro-averaged over all queries.

    Parameters
    ----------
    query_ids      : list[str]         ordered query Entry IDs
    ranked_id_lists: list[list[str]]   top-k corpus IDs per query (same order)
    qrels          : dict[str, dict[str, int]]
    k              : cutoff

    Returns
    -------
    dict with MAP@k, nDCG@k, Recall@k, n_queries
    """
    maps, ndcgs, recalls = [], [], []
    for qid, ranked_ids in zip(query_ids, ranked_id_lists):
        # Relevant set = all corpus IDs stored in qrels for this query
        relevant_set = set(qrels.get(qid, {}).keys())
        if not relevant_set:
            continue
        maps.append(average_precision_at_k(ranked_ids, relevant_set, k))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant_set, k))
        recalls.append(recall_at_k(ranked_ids, relevant_set, k))

    return {
        f'MAP@{k}'    : float(np.mean(maps))    if maps else 0.0,
        f'nDCG@{k}'   : float(np.mean(ndcgs))   if ndcgs else 0.0,
        f'Recall@{k}' : float(np.mean(recalls)) if recalls else 0.0,
        'n_queries'   : len(maps),
    }


# =============================================================================
# K-MER RETRIEVAL
# =============================================================================

def run_kmer_retrieval(queries, corpus, query_ids, corpus_ids, k, top_k=5):
    """
    Encode proteins as k-mer frequency vectors and retrieve top-k by cosine similarity.

    Parameters
    ----------
    queries    : dict[str, str]   {Entry: Sequence}
    corpus     : dict[str, str]   {Entry: Sequence}
    query_ids  : list[str]        ordered query IDs (sets row order of query matrix)
    corpus_ids : list[str]        ordered corpus IDs (sets col order of sim matrix)
    k          : int              k-mer length (2, 3, or 4)
    top_k      : int              number of results to return per query

    Returns
    -------
    ranked_id_lists : list[list[str]]  top-k corpus Entry IDs per query
    """
    # Build k-mer strings in the same order as query_ids and corpus_ids
    # so that row i of the similarity matrix corresponds to query_ids[i]
    corpus_kmers = [get_kmers(corpus[cid], k) for cid in corpus_ids]
    query_kmers  = [get_kmers(queries[qid], k) for qid in query_ids]

    # Fit vocabulary on corpus only, then transform both
    # token_pattern matches uppercase amino acid strings exactly
    # (default CountVectorizer regex would filter short k-mers incorrectly)
    vectorizer = CountVectorizer(
        dtype=np.float32,
        lowercase=False,
        token_pattern=r"[A-Z]+",   # match any uppercase string (amino acid k-mers)
    )
    X_corpus = vectorizer.fit_transform(corpus_kmers)  # (n_corpus, vocab_size)
    X_query  = vectorizer.transform(query_kmers)        # (n_queries, vocab_size)

    # L2-normalise rows so dot product == cosine similarity
    X_corpus_norm = normalize(X_corpus)
    X_query_norm  = normalize(X_query)

    print(f"    Vocabulary size (k={k}): {len(vectorizer.vocabulary_):,}")
    print(f"    Query matrix   : {X_query_norm.shape}")
    print(f"    Corpus matrix  : {X_corpus_norm.shape}")

    # Compute full similarity matrix: (n_queries, n_corpus)
    # For Arch: 2,343 × 9,229 — fine on laptop RAM
    sim_matrix = cosine_similarity(X_query_norm, X_corpus_norm)

    # Rank corpus per query by descending similarity, take top_k
    ranked_id_lists = []
    for i in range(len(query_ids)):
        # argsort ascending → reverse for descending
        top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
        ranked_id_lists.append([corpus_ids[j] for j in top_indices])

    return ranked_id_lists


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def evaluate_kmer_task(task_key, k_values=(2, 3, 4), top_k=5):
    """Run k-mer retrieval for multiple k values on one task and return results."""
    queries, corpus, qrels, query_ids, corpus_ids = load_task(task_key)
    label = DATASET_CONFIG[task_key]['label']

    task_results = []

    for k in k_values:
        print(f"\n  --- k={k} ---")
        ranked_id_lists = run_kmer_retrieval(
            queries, corpus, query_ids, corpus_ids, k=k, top_k=top_k
        )
        metrics = evaluate_rankings(query_ids, ranked_id_lists, qrels, k=top_k)

        print(f"    MAP@5    : {metrics['MAP@5']:.4f}")
        print(f"    nDCG@5   : {metrics['nDCG@5']:.4f}")
        print(f"    Recall@5 : {metrics['Recall@5']:.4f}")
        print(f"    n_queries: {metrics['n_queries']}")

        task_results.append({
            'Method'   : f'k-mer (k={k})',
            'Dataset'  : label,
            'MAP@5'    : round(metrics['MAP@5'],    4),
            'nDCG@5'   : round(metrics['nDCG@5'],   4),
            'Recall@5' : round(metrics['Recall@5'], 4),
            'n_queries': metrics['n_queries'],
        })

    return task_results


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':

    all_results = []

    for task_key in ('arch', 'euk'):
        results = evaluate_kmer_task(task_key, k_values=(2, 3, 4), top_k=5)
        all_results.extend(results)

    # Summary table
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 65)
    print("FINAL RESULTS — Method 2: k-mer Cosine Similarity")
    print("=" * 65)
    print(df.to_string(index=False))

    # Save
    df.to_csv('kmer_results.csv', index=False)
    print("\nResults saved to kmer_results.csv")