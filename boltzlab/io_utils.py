from pathlib import Path

def split_fasta(fasta_path: str, output_dir: str = None) -> list[str]:
    """
    Split a multi-entry FASTA into separate files (one per entry).

    Parameters
    ----------
    fasta_path : str
        Path to multi-entry FASTA file.
    output_dir : str
        Where to save split FASTA files. Defaults to input dir.

    Returns
    -------
    List of paths to split FASTA files.
    """
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir) if output_dir else fasta_path.parent
    output_dir.mkdir(exist_ok=True)

    entries = []
    with open(fasta_path) as f:
        lines = f.readlines()

    current_header = None
    current_seq = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if current_header:
                entry_path = write_split_entry(current_header, current_seq, output_dir)
                entries.append(str(entry_path))
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)

    if current_header:
        entry_path = write_split_entry(current_header, current_seq, output_dir)
        entries.append(str(entry_path))

    return entries


def write_split_entry(header, seq_lines, output_dir):
    chain_id = header.split("|")[0][1:]  # >A|protein|...
    filename = f"{chain_id}.fasta"
    entry_path = output_dir / filename
    with open(entry_path, "w") as out_f:
        out_f.write(f"{header}\n")
        out_f.write("\n".join(seq_lines) + "\n")
    return entry_path

def write_fasta_entry(
    output_path: str,
    chain_id: str,
    entity_type: str,
    msa_path: str,
    sequence: str,
):
    """
    Write a single-entry FASTA file for Boltz2 input.

    Parameters
    ----------
    output_path : str
        Path to save the FASTA file.
    chain_id : str
        Unique chain ID (e.g., 'A').
    entity_type : str
        'protein', 'dna', 'rna', 'smiles', or 'ccd'.
    msa_path : str
        MSA path or 'empty' for single-sequence mode.
    sequence : str
        The amino acid or nucleotide sequence.
    """
    with open(output_path, "w") as f:
        f.write(f">{chain_id}|{entity_type}|{msa_path}\n")
        f.write(sequence.strip() + "\n")