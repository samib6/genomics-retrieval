# =============================================================================
# METHOD 2: k-mer Cosine Similarity Retrieval (FINAL DGEB VERSION)
# =============================================================================

import numpy as np
import pandas as pd
from collections import defaultdict

import pytrec_eval
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# =============================================================================
# DATA LOADING
# =============================================================================

DATASET_CONFIG = {
    'arch': {
        'seqs_path': 'tattabio/arch_retrieval',
        'qrels_path': 'tattabio/arch_retrieval_qrels',
        'label': 'Arch Retrieval',
    },
    'euk': {
        'seqs_path': 'tattabio/euk_retrieval',
        'qrels_path': 'tattabio/euk_retrieval_qrels',
        'label': 'Euk Retrieval',
    },
}


def load_task(task_key):
    cfg = DATASET_CONFIG[task_key]
    print(f"\nLoading {cfg['label']} ...")

    seqs_ds = load_dataset(cfg['seqs_path'])
    qrels_ds = load_dataset(cfg['qrels_path'])

    corpus = {row['Entry']: row['Sequence'] for row in seqs_ds['train']}
    queries = {row['Entry']: row['Sequence'] for row in seqs_ds['test']}

    corpus_ids = list(corpus.keys())
    query_ids = list(queries.keys())

    qrels = defaultdict(dict)
    for split in qrels_ds:
        for row in qrels_ds[split]:
            qrels[str(row['query_id'])][str(row['corpus_id'])] = 1

    qrels = dict(qrels)

    return queries, corpus, qrels, query_ids, corpus_ids


# =============================================================================
# K-MER ENCODING
# =============================================================================

def get_kmers(sequence, k):
    if len(sequence) < k:
        return ""
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1))


# =============================================================================
# RETRIEVAL
# =============================================================================

def run_kmer_retrieval(queries, corpus, query_ids, corpus_ids, k, top_k):
    corpus_kmers = [get_kmers(corpus[cid], k) for cid in corpus_ids]
    query_kmers = [get_kmers(queries[qid], k) for qid in query_ids]

    vectorizer = CountVectorizer(
        dtype=np.float32,
        lowercase=False,
        token_pattern=r"[A-Z]+",
    )

    X_corpus = vectorizer.fit_transform(corpus_kmers)
    X_query = vectorizer.transform(query_kmers)

    X_corpus = normalize(X_corpus)
    X_query = normalize(X_query)

    print(f"    Vocabulary size (k={k}): {len(vectorizer.vocabulary_):,}")

    sim_matrix = cosine_similarity(X_query, X_corpus)

    ranked_id_lists = []
    similarity_lists = []

    for i in range(len(query_ids)):
        top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]

        ranked_ids = [corpus_ids[j] for j in top_indices]
        sim_scores = [sim_matrix[i][j] for j in top_indices]

        ranked_id_lists.append(ranked_ids)
        similarity_lists.append(sim_scores)

    return ranked_id_lists, similarity_lists


# =============================================================================
# EVALUATION (DGEB CONSISTENT)
# =============================================================================

def evaluate_with_pytrec_eval(
    query_ids,
    ranked_id_lists,
    similarity_lists,
    qrels,
    k_values=(5, 10, 50)
):

    results = {}
    for qid, ranked_ids, sim_scores in zip(query_ids, ranked_id_lists, similarity_lists):
        results[qid] = {
            cid: float(score)
            for cid, score in zip(ranked_ids, sim_scores)
        }

    metrics = {"recip_rank"}

    for k in k_values:
        metrics.update({
            f"map_cut.{k}",
            f"ndcg_cut.{k}",
            f"recall.{k}",
            f"P.{k}",
        })

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    scores = evaluator.evaluate(results)

    aggregated = {}

    for k in k_values:
        map_scores = []
        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for qid in query_ids:
            q = scores.get(qid, {})

            map_scores.append(q.get(f"map_cut_{k}", 0.0))
            ndcg_scores.append(q.get(f"ndcg_cut_{k}", 0.0))
            recall_scores.append(q.get(f"recall_{k}", 0.0))
            precision_scores.append(q.get(f"P_{k}", 0.0))

        aggregated[f"MAP@{k}"] = round(np.mean(map_scores), 5)
        aggregated[f"NDCG@{k}"] = round(np.mean(ndcg_scores), 5)
        aggregated[f"Recall@{k}"] = round(np.mean(recall_scores), 5)
        aggregated[f"P@{k}"] = round(np.mean(precision_scores), 5)

    # MRR
    mrr_scores = [scores.get(qid, {}).get("recip_rank", 0.0) for qid in query_ids]
    aggregated["MRR"] = round(np.mean(mrr_scores), 5)

    aggregated["n_queries"] = len(query_ids)

    return aggregated


# =============================================================================
# MAIN LOOP
# =============================================================================

def evaluate_kmer_task(task_key, k_values=(2, 3, 4), eval_at=(5, 10, 50)):
    queries, corpus, qrels, query_ids, corpus_ids = load_task(task_key)
    label = DATASET_CONFIG[task_key]['label']

    top_k = max(eval_at)

    results = []

    for k in k_values:
        print(f"\n--- k = {k} ---")

        ranked_id_lists, similarity_lists = run_kmer_retrieval(
            queries, corpus, query_ids, corpus_ids,
            k=k, top_k=top_k
        )

        metrics = evaluate_with_pytrec_eval(
            query_ids,
            ranked_id_lists,
            similarity_lists,
            qrels,
            k_values=eval_at
        )

        print(metrics)

        results.append({
            'Method': f'k-mer (k={k})',
            'Dataset': label,
            **metrics
        })

    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    all_results = []

    for task in ("arch", "euk"):
        all_results.extend(
            evaluate_kmer_task(task, k_values=(2, 3, 4), eval_at=(5, 10, 50))
        )

    df = pd.DataFrame(all_results)

    print("\nFINAL RESULTS")
    print(df.to_string(index=False))

    df.to_csv("kmer_results.csv", index=False)