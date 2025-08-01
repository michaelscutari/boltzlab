"""
cli.py
-------
command-line interface for structure and affinity prediction with boltz2
"""

import sys
import click
from pathlib import Path

# core predictions
from boltzlab.predict import predict_structure, predict_affinity
from boltzlab.parallel import ParallelPredictor

# utilities
from boltzlab.io_utils import split_fasta
from boltzlab.slurm_utils import check_slurm_available, validate_slurm_partition

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    help="Path to input FASTA file with one or more sequences."
)
@click.option(
    "--sequence",
    type=str,
    help="Raw amino acid sequence. Overrides --fasta if both are provided."
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Output directory where prediction results will be saved."
)
@click.option(
    "--simple-output",
    is_flag=True,
    help="Reorganize output into simple structure: structure.pdb and confidence.json."
)
def predict_structure(fasta, sequence, outdir, simple_output):
    """
    Predict 3D structure from input sequence.
    """
    if not fasta and not sequence:
        raise click.UsageError("Must provide either --fasta or --sequence.")

    click.echo("Running structure prediction...")
    from boltzlab.predict import predict_structure as predict_structure_fn

    try:
        result = predict_structure_fn(
            fasta_path=fasta,
            raw_sequence=sequence,
            output_dir=outdir,
            simple_output=simple_output
        )
        click.echo(f"Prediction complete. Results saved to {result['output_dir']}")

        confidence = result.get("confidence")
        if confidence:
            click.echo("Confidence summary:")
            for k, v in confidence.items():
                if isinstance(v, float):
                    click.echo(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    click.echo(f"  {k}: [nested dictionary]")
                else:
                    click.echo(f"  {k}: {v}")
        else:
            print("you have no confidence :()")
                    
    except RuntimeError as err:
        click.echo(f"Prediction failed: {err}", err=True)
        sys.exit(1)

@cli.command()
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to multi-entry FASTA file with sequences to predict."
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Output directory where prediction results will be saved."
)
@click.option(
    "--partition",
    default="gpu",
    help="SLURM partition name (default: gpu)."
)
@click.option(
    "--max-jobs",
    default=50,
    help="Maximum number of concurrent SLURM jobs (default: 50)."
)
@click.option(
    "--num-samples",
    default=1,
    help="Number of diffusion samples per prediction (default: 1)."
)
@click.option(
    "--sampling-steps", 
    default=200,
    help="Number of sampling steps for diffusion (default: 200)."
)
@click.option(
    "--num-recycling",
    default=3,
    help="Number of recycling steps (default: 3)."
)
@click.option(
    "--output-format",
    default="mmcif",
    type=click.Choice(['mmcif', 'pdb']),
    help="Output format for structures (default: mmcif)."
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume previous parallel run (skip completed jobs)."
)
@click.option(
    "--use-msa-server/--no-msa-server",
    default=True,
    help="Use MSA server for alignments (default: enabled)."
)
@click.option(
    "--checkpoint",
    type=str,
    help="Path to custom Boltz checkpoint file."
)
@click.option(
    "--simple-output",
    is_flag=True,
    help="Reorganize output into simple structure: structure.pdb and confidence.json per protein."
)
def predict_parallel(
    fasta, outdir, partition, max_jobs, num_samples, 
    sampling_steps, num_recycling, output_format, resume, 
    use_msa_server, checkpoint, simple_output
):
    """
    Predict structures for multiple sequences in parallel using SLURM.
    
    This command splits a multi-entry FASTA file and submits parallel
    structure prediction jobs to a SLURM cluster. Each sequence gets
    its own job with the same prediction parameters.
    
    Example:
        boltzlab predict-parallel --fasta sequences.fasta --outdir results/ --partition gpu
    """
    
    # Validate SLURM availability
    if not check_slurm_available():
        click.echo("Error: SLURM is not available on this system.", err=True)
        click.echo("Make sure you're running on a SLURM cluster with sinfo/squeue commands available.")
        sys.exit(1)
    
    # Validate partition
    if not validate_slurm_partition(partition):
        click.echo(f"Warning: Could not validate SLURM partition '{partition}'.")
        click.echo("Proceeding anyway - check your partition name if jobs fail to submit.")
    
    # Check if FASTA file has multiple sequences
    fasta_path = Path(fasta)
    with open(fasta_path, 'r') as f:
        content = f.read()
        seq_count = content.count('>')
    
    if seq_count == 0:
        click.echo("Error: No sequences found in FASTA file.", err=True)
        sys.exit(1)
    elif seq_count == 1:
        click.echo("Warning: Only one sequence found. Consider using 'predict-structure' for single sequences.")
    
    click.echo(f"Found {seq_count} sequences in FASTA file.")
    click.echo(f"Output directory: {outdir}")
    click.echo(f"SLURM partition: {partition}")
    click.echo(f"Max concurrent jobs: {max_jobs}")
    
    if resume:
        click.echo("Resume mode: Will skip completed jobs and retry failed ones.")
    
    # Prepare Boltz parameters
    boltz_params = {
        'use_msa_server': use_msa_server,
        'num_recycling': num_recycling,
        'num_samples': num_samples,
        'sampling_steps': sampling_steps,
        'output_format': output_format,
        'simple_output': simple_output,
    }
    
    if checkpoint:
        boltz_params['checkpoint'] = checkpoint
    
    click.echo("\nBoltz parameters:")
    for key, value in boltz_params.items():
        click.echo(f"  {key}: {value}")
    
    # Confirm before submitting
    if not resume:
        click.echo(f"\nAbout to submit {seq_count} jobs to SLURM partition '{partition}'.")
        if not click.confirm("Continue?"):
            click.echo("Cancelled.")
            sys.exit(0)
    
    try:
        # Create parallel predictor and run
        predictor = ParallelPredictor(
            output_dir=outdir,
            slurm_partition=partition,
            max_jobs=max_jobs
        )
        
        summary = predictor.predict_structures_parallel(
            fasta_path=str(fasta_path),
            resume=resume,
            **boltz_params
        )
        
        # Display final results
        click.echo("\n" + "="*50)
        click.echo("PARALLEL PREDICTION COMPLETE")
        click.echo("="*50)
        click.echo(f"Total sequences: {summary['total_jobs']}")
        click.echo(f"Completed: {summary['completed']}")
        click.echo(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            click.echo("\nFailed jobs can be retried by running the same command again.")
            click.echo(f"Check {outdir}/summary.txt for details.")
        
        if summary['completed'] > 0:
            click.echo(f"\nResults available in: {outdir}/results/")
            
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user. Jobs may still be running on SLURM.")
        click.echo("Use 'squeue' to check job status or 'scancel' to cancel jobs.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during parallel prediction: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option(
    "--csv",
    type=click.Path(exists=True),
    required=True,
    help="CSV file with sequence and SMILES columns."
)
@click.option(
    "--outdir",
    required=True,
    type=click.Path(),
    help="Directory to save prediction results."
)
def predict_affinity(csv, outdir):
    """Predict ligand binding affinity from sequence–SMILES pairs."""
    from boltzlab.predict import predict_affinity as predict_affinity_fn

    predict_affinity_fn(
        csv_file=csv,
        output_dir=outdir
    )

    click.echo(f"Affinity prediction complete. Results written to: {outdir}")

@cli.command()
@click.option(
    "--fasta",
    type=click.Path(exists=True),
    required=True,
    help="Path to a multi-entry FASTA file."
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Optional output directory (defaults to same dir as input)."
)
def split_fasta(fasta, outdir):
    """Split a multi-entry FASTA file into separate single-entry FASTA files."""
    from boltzlab.io_utils import split_fasta as split_fasta_fn

    files = split_fasta_fn(fasta_path=fasta, output_dir=outdir)
    click.echo(f"Split {len(files)} entries into: {Path(outdir or fasta).parent.resolve()}")

@cli.command()
@click.option(
    "--partition",
    default=None,
    help="Check specific SLURM partition (optional)."
)
def status(partition):
    """Check SLURM queue status and system availability."""
    
    # Check SLURM availability
    if not check_slurm_available():
        click.echo("❌ SLURM is not available on this system.")
        return
    
    click.echo("✅ SLURM is available.")
    
    # Check partition if specified
    if partition:
        if validate_slurm_partition(partition):
            click.echo(f"✅ Partition '{partition}' is available.")
        else:
            click.echo(f"❌ Partition '{partition}' is not available or accessible.")
    
    # Get queue status
    from boltzlab.slurm_utils import get_queue_status
    
    queue_info = get_queue_status(partition)
    
    if 'error' in queue_info:
        click.echo(f"❌ Error getting queue status: {queue_info['error']}")
    else:
        click.echo(f"\nQueue Status{f' (partition: {partition})' if partition else ''}:")
        click.echo(f"  Total jobs: {queue_info['total_jobs']}")
        click.echo(f"  Running: {queue_info['running']}")
        click.echo(f"  Pending: {queue_info['pending']}")
        click.echo(f"  Boltz jobs: {queue_info['boltz_jobs']}")

if __name__ == "__main__":
    cli()