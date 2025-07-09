"""
slurm_utils.py
--------------
SLURM job management utilities for parallel prediction
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple

import submitit

from boltzlab.predict import predict_structure


def create_slurm_executor(partition: str, log_folder: str) -> submitit.AutoExecutor:
    """
    Create and configure a submitit executor for SLURM.
    
    Parameters
    ----------
    partition : str
        SLURM partition name
    log_folder : str
        Directory for SLURM log files
        
    Returns
    -------
    Configured submitit executor
    """
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Configure SLURM parameters optimized for Boltz2
    executor.update_parameters(
        # Basic SLURM settings
        slurm_partition=partition,
        slurm_job_name="boltz_predict",
        
        # Resource requirements for Boltz2
        slurm_time="02:00:00",         # 2 hours 
        slurm_mem="16G",               # 16GB RAM
        slurm_gres="gpu:1",            # 1 GPU required
        slurm_cpus_per_task=4,         # 4 CPU cores
        
        # Job management
        slurm_array_parallelism=50,    # Max 50 concurrent array jobs
        
        # Output settings (submitit will handle log file naming automatically)
        stderr_to_stdout=True,         # Combine stderr and stdout
    )
    
    return executor


def boltz_job_function(job_args: Tuple[str, str, Dict]) -> Dict[str, Any]:
    """
    Function that runs on each SLURM node to execute Boltz prediction.
    
    Parameters
    ----------
    job_args : Tuple[str, str, Dict]
        Tuple containing (fasta_path, output_path, boltz_params)
        
    Returns
    -------
    Dict containing job results and metadata
    """
    fasta_path, output_path, boltz_params = job_args
    
    try:
        # Convert to absolute paths to avoid working directory issues on compute nodes
        fasta_path = str(Path(fasta_path).resolve())
        output_path = str(Path(output_path).resolve())
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Run Boltz prediction with absolute paths
        result = predict_structure(
            fasta_path=fasta_path,
            output_dir=output_path,
            **boltz_params
        )
        
        # Add job metadata
        result.update({
            'job_status': 'completed',
            'input_fasta': fasta_path,
            'output_dir': output_path,
            'node_name': os.environ.get('SLURMD_NODENAME', 'unknown'),
            'job_id': os.environ.get('SLURM_JOB_ID', 'unknown')
        })
        
        return result
        
    except Exception as e:
        # Return error information
        error_result = {
            'job_status': 'failed',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'input_fasta': fasta_path,
            'output_dir': output_path,
            'node_name': os.environ.get('SLURMD_NODENAME', 'unknown'),
            'job_id': os.environ.get('SLURM_JOB_ID', 'unknown')
        }
        
        # Clean up partial outputs on failure
        cleanup_failed_jobs(output_path)
        
        # Re-raise the exception so submitit knows the job failed
        raise Exception(f"Boltz prediction failed: {str(e)}")


def get_job_status_summary(jobs: List[submitit.Job]) -> Dict[str, int]:
    """
    Get summary of job statuses.
    
    Parameters
    ----------
    jobs : List[submitit.Job]
        List of submitit Job objects
        
    Returns
    -------
    Dict with counts of job statuses
    """
    status_counts = {
        'pending': 0,
        'running': 0,
        'completed': 0,
        'failed': 0,
        'cancelled': 0,
        'unknown': 0
    }
    
    for job in jobs:
        try:
            if job.done():
                try:
                    job.result()  # This will raise exception if failed
                    status_counts['completed'] += 1
                except:
                    status_counts['failed'] += 1
            else:
                # Job is still running or pending
                try:
                    state = job.state
                    if 'RUNNING' in state:
                        status_counts['running'] += 1
                    elif 'PENDING' in state:
                        status_counts['pending'] += 1
                    elif 'CANCELLED' in state:
                        status_counts['cancelled'] += 1
                    else:
                        status_counts['unknown'] += 1
                except:
                    status_counts['unknown'] += 1
                    
        except Exception:
            status_counts['unknown'] += 1
            
    return status_counts


def cleanup_failed_jobs(output_path: str):
    """
    Clean up partial outputs from failed jobs.
    
    Parameters
    ----------
    output_path : str
        Path to job output directory
    """
    output_dir = Path(output_path)
    
    if output_dir.exists():
        try:
            # Remove the entire output directory and its contents
            shutil.rmtree(output_dir)
            print(f"Cleaned up failed job output: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")


def check_slurm_available() -> bool:
    """
    Check if SLURM is available on the system.
    
    Returns
    -------
    bool
        True if SLURM commands are available
    """
    try:
        import subprocess
        result = subprocess.run(['sinfo', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def validate_slurm_partition(partition: str) -> bool:
    """
    Validate that the specified SLURM partition exists and is available.
    
    Parameters
    ----------
    partition : str
        SLURM partition name
        
    Returns
    -------
    bool
        True if partition is valid and available
    """
    try:
        import subprocess
        result = subprocess.run(['sinfo', '-p', partition, '--noheader'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def estimate_job_resources(num_sequences: int, avg_sequence_length: int = 500) -> Dict[str, str]:
    """
    Estimate resource requirements based on number and size of sequences.
    
    Parameters
    ----------
    num_sequences : int
        Number of sequences to predict
    avg_sequence_length : int
        Average sequence length
        
    Returns
    -------
    Dict with recommended resource settings
    """
    # Basic resource estimation (can be refined based on experience)
    base_time_minutes = 30  # Base time per sequence
    
    # Adjust time based on sequence length
    if avg_sequence_length > 1000:
        time_multiplier = 2.0
    elif avg_sequence_length > 500:
        time_multiplier = 1.5
    else:
        time_multiplier = 1.0
        
    estimated_time_per_job = int(base_time_minutes * time_multiplier)
    
    # Format time as HH:MM:SS
    hours = estimated_time_per_job // 60
    minutes = estimated_time_per_job % 60
    time_str = f"{hours:02d}:{minutes:02d}:00"
    
    return {
        'time': time_str,
        'memory': '16G',  # Standard for Boltz2
        'gpus': '1',
        'cpus': '4'
    }


def get_queue_status(partition: str = None) -> Dict[str, Any]:
    """
    Get current SLURM queue status for monitoring.
    
    Parameters
    ----------
    partition : str, optional
        Specific partition to check
        
    Returns
    -------
    Dict with queue information
    """
    try:
        import subprocess
        
        cmd = ['squeue', '--format=%i,%j,%t,%u', '--noheader']
        if partition:
            cmd.extend(['-p', partition])
            
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return {'error': 'Failed to get queue status'}
            
        lines = result.stdout.strip().split('\n')
        
        status = {
            'total_jobs': len(lines) if lines[0] else 0,
            'running': 0,
            'pending': 0,
            'boltz_jobs': 0
        }
        
        for line in lines:
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) >= 3:
                job_name = parts[1] if len(parts) > 1 else ''
                state = parts[2] if len(parts) > 2 else ''
                
                if 'boltz' in job_name.lower():
                    status['boltz_jobs'] += 1
                    
                if state == 'R':
                    status['running'] += 1
                elif state in ['PD', 'CF']:
                    status['pending'] += 1
                    
        return status
        
    except Exception as e:
        return {'error': f'Error checking queue: {str(e)}'}