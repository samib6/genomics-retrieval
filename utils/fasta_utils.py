from typing import Dict

def read_fasta(file_path: str) -> Dict[str, str]:
    sequences = {}
    with open(file_path, "r") as f:
        seq_id = None
        seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = "".join(seq)
                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)

        if seq_id:
            sequences[seq_id] = "".join(seq)

    return sequences