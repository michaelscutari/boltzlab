import os
import json
import subprocess
import tempfile
import csv

from pathlib import Path
from typing import Optional

from boltzlab.io_utils import write_fasta_entry

CACHE_DIR = Path(__file__).resolve().parent.parent / ".boltz_cache"


def predict_structure(
    fasta_path: Optional[str] = None,
    raw_sequence: Optional[str] = None,
    output_dir: str = None,
    use_msa_server: bool = True,
    checkpoint: Optional[str] = None,
    num_recycling: int = 3,
    num_samples: int = 1,
    sampling_steps: int = 200,
    output_format: str = "mmcif",
    simple_output: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    if raw_sequence:
        input_fasta = Path(tempfile.mkdtemp()) / "input.fasta"
        write_fasta_entry(
            output_path=input_fasta,
            chain_id="A",
            entity_type="protein",
            msa_path="empty",
            sequence=raw_sequence,
        )
        model_name = "input"
    else:
        input_fasta = Path(fasta_path)
        model_name = Path(fasta_path).stem

    boltz_cmd = [
        "boltz", "predict", str(input_fasta),
        "--out_dir", output_dir,
        "--cache", str(CACHE_DIR),
        "--recycling_steps", str(num_recycling),
        "--diffusion_samples", str(num_samples),
        "--sampling_steps", str(sampling_steps),
        "--output_format", output_format,
    ]

    if use_msa_server:
        boltz_cmd.append("--use_msa_server")
    if checkpoint:
        boltz_cmd += ["--checkpoint", checkpoint]

    subprocess.run(boltz_cmd, check=True)

    pred_dir = Path(output_dir) / f"boltz_results_{model_name}/predictions"
    model_dir = pred_dir / model_name
    confidence_json = model_dir / f"confidence_{model_name}_model_0.json"

    print(f"json: {confidence_json}")

    summary = {}
    if confidence_json.exists():
        with open(confidence_json) as f:
            summary = json.load(f)

    if simple_output:
        from boltzlab.slurm_utils import _reorganize_simple_output
        _reorganize_simple_output(output_dir, model_name)

    return {
        "success": True,
        "output_dir": str(output_dir),
        "confidence": summary,
    }


def predict_affinity(
    csv_file: str,
    output_dir: str,
    use_msa_server: bool = True,
    checkpoint: Optional[str] = None,
    num_samples: int = 5,
    sampling_steps: int = 200,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            seq = row["sequence"]
            smiles = row["smiles"]

            prefix = f"sample_{i}"
            yaml_path = Path(output_dir) / f"{prefix}.yaml"
            yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: "{seq}"
  - ligand:
      id: L
      smiles: "{smiles}"
properties:
  - affinity:
      binder: L
"""
            with open(yaml_path, "w") as yaml_f:
                yaml_f.write(yaml_content)

            boltz_cmd = [
                "boltz", "predict", str(yaml_path),
                "--out_dir", output_dir,
                "--cache", str(CACHE_DIR),
                "--diffusion_samples_affinity", str(num_samples),
                "--sampling_steps_affinity", str(sampling_steps),
                "--affinity_mw_correction",
            ]

            if use_msa_server:
                boltz_cmd.append("--use_msa_server")
            if checkpoint:
                boltz_cmd += ["--affinity_checkpoint", checkpoint]

            subprocess.run(boltz_cmd, check=True)