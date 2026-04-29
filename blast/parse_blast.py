from collections import defaultdict

def parse_blast_results(file_path):
    results = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            qid, sid, evalue, bitscore = line.strip().split()
            results[qid].append((sid, float(evalue)))

    # sort by e-value
    final = {}
    for qid in results:
        sorted_hits = sorted(results[qid], key=lambda x: x[1])
        final[qid] = [sid for sid, _ in sorted_hits[:5]]

    return final