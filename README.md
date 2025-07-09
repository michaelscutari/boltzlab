# BoltzLab CLI

Command-line interface for running structure and affinity predictions using Boltz2.

**Note:** This tool is a lightweight wrapper. If you need support for additional Boltz2 features (e.g., YAML inputs, templates, MSAs), ask Michael.

## Setup

1. Activate your micromamba environment:
   ```bash
   micromamba activate /hpc/group/singhlab/lab_envs/boltzlab
   ```

2. Install the CLI tool (in case it was updated):
   ```bash
   v
   pip install -e .
   ```

## Commands

### Predict Structure

```bash
boltzlab predict-structure --fasta input.fasta --outdir results/
```

Or use a raw sequence:
```bash
boltzlab predict-structure --sequence MKT... --outdir results/
```

### Predict Affinity

```bash
boltzlab predict-affinity --csv inputs.csv --outdir results/
```

CSV must contain two columns: `sequence` and `smiles`.

### Split FASTA

```bash
boltzlab split-fasta --fasta input.fasta
```

Splits each FASTA entry into a separate `.fasta` file.

## Python API

You can also import and use these functions directly in Python:

```python
from boltzlab.predict import predict_structure, predict_affinity

# Predict structure from sequence
result = predict_structure(
    raw_sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    output_dir="results/",
    num_samples=1
)

# Predict affinity from CSV
predict_affinity(
    csv="protein_ligand_pairs.csv",
    output_dir="affinity_results/",
    num_samples=5
)
```

The Python functions provide the same functionality as the CLI commands with additional control over parameters like `num_recycling`, `sampling_steps`, and `use_msa_server`.

## Notes

- Structure prediction supports FASTA input with or without MSA.
- Affinity mode assumes protein–ligand pairs using SMILES strings.
- Results are saved in the specified output directory.

## File Layout

```
boltzlab/
├── cli.py
├── predict.py
├── io_utils.py
├── model_loader.py
└── ...
```