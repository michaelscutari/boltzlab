"""
parallel.py
-----------
Parallel structure prediction using SLURM job arrays
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import submitit
from tqdm import tqdm

from boltzlab.io_utils import split_fasta
from boltzlab.slurm_utils import (
    create_slurm_executor, 
    boltz_job_function, 
    cleanup_failed_jobs
)


class ParallelPredictor:
    """Manages parallel structure prediction across SLURM cluster."""
    
    def __init__(
        self, 
        output_dir: str, 
        slurm_partition: str = "gpu", 
        max_jobs: int = 50
    ):
        self.output_dir = Path(output_dir)
        self.slurm_partition = slurm_partition
        self.max_jobs = max_jobs
        
        # Working directories
        self.work_dir = self.output_dir / ".parallel_work"
        self.splits_dir = self.work_dir / "splits"
        self.logs_dir = self.work_dir / "logs"
        self.results_dir = self.output_dir / "results"
        self.status_file = self.work_dir / "status.json"
        
    def predict_structures_parallel(
        self,
        fasta_path: str,
        resume: bool = False,
        **boltz_params
    ) -> Dict[str, Any]:
        """
        Run parallel structure prediction on multi-FASTA file.
        
        Parameters
        ----------
        fasta_path : str
            Path to multi-entry FASTA file
        resume : bool
            Resume previous run if True
        **boltz_params
            Parameters passed to boltz predict_structure
            
        Returns
        -------
        Dict containing summary of results
        """
        print("Starting parallel structure prediction...")
        print(f"Output directory: {self.output_dir}")
        print(f"SLURM partition: {self.slurm_partition}")
        
        # Setup working environment
        self._setup_directories()
        
        # Load or create job list
        if resume and self.status_file.exists():
            job_info = self._load_job_status()
            print(f"Resuming previous run with {len(job_info['jobs'])} jobs")
        else:
            job_info = self._create_new_job_list(fasta_path, boltz_params)
            print(f"Created {len(job_info['jobs'])} jobs from FASTA file")
        
        # Submit jobs that need to run
        pending_jobs = self._get_pending_jobs(job_info)
        if pending_jobs:
            print(f"Submitting {len(pending_jobs)} jobs to SLURM...")
            submitted_jobs = self._submit_job_array(pending_jobs, boltz_params)
            self._update_job_status(job_info, submitted_jobs)
        else:
            print("No new jobs to submit.")
            
        # Monitor progress
        if any(job.get('slurm_job_id') for job in job_info['jobs']):
            print("Monitoring job progress...")
            self._monitor_jobs(job_info)
        
        # Collect and summarize results
        summary = self._collect_results(job_info)
        self._write_summary(summary)
        
        print("Parallel prediction complete!")
        print(f"Results summary: {summary['completed']} completed, {summary['failed']} failed")
        
        return summary
    
    def _setup_directories(self):
        """Create necessary working directories."""
        for directory in [self.work_dir, self.splits_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _create_new_job_list(self, fasta_path: str, boltz_params: Dict) -> Dict:
        """Split FASTA and create job list."""
        # Split FASTA into individual files
        split_files = split_fasta(fasta_path, str(self.splits_dir))
        
        # Create job metadata with absolute paths
        jobs = []
        for i, fasta_file in enumerate(split_files):
            seq_name = Path(fasta_file).stem
            # Convert to absolute paths to avoid working directory issues
            abs_fasta_path = str(Path(fasta_file).resolve())
            abs_output_path = str((self.results_dir / seq_name).resolve())
            
            jobs.append({
                'job_id': i,
                'sequence_name': seq_name,
                'fasta_path': abs_fasta_path,
                'output_path': abs_output_path,
                'status': 'pending',
                'slurm_job_id': None,
                'error_message': None
            })
        
        job_info = {
            'total_jobs': len(jobs),
            'boltz_params': boltz_params,
            'jobs': jobs,
            'created_at': time.time()
        }
        
        # Save initial status
        self._save_job_status(job_info)
        return job_info
    
    def _load_job_status(self) -> Dict:
        """Load existing job status from file."""
        with open(self.status_file, 'r') as f:
            return json.load(f)
    
    def _save_job_status(self, job_info: Dict):
        """Save job status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(job_info, f, indent=2)
    
    def _get_pending_jobs(self, job_info: Dict) -> List[Dict]:
        """Get jobs that need to be submitted or resubmitted."""
        pending = []
        for job in job_info['jobs']:
            # Check if job already completed successfully
            if job['status'] == 'completed':
                output_path = Path(job['output_path'])
                if output_path.exists() and any(output_path.iterdir()):
                    continue  # Skip completed jobs with outputs
            
            # Clean up any partial outputs from failed jobs
            if job['status'] == 'failed':
                cleanup_failed_jobs(job['output_path'])
                
            job['status'] = 'pending'
            job['slurm_job_id'] = None
            job['error_message'] = None
            pending.append(job)
            
        return pending
    
    def _submit_job_array(self, jobs: List[Dict], boltz_params: Dict) -> List[submitit.Job]:
        """Submit job array to SLURM."""
        # Create executor
        executor = create_slurm_executor(
            partition=self.slurm_partition,
            log_folder=str(self.logs_dir)
        )
        
        # Limit concurrent jobs
        batch_size = min(len(jobs), self.max_jobs)
        
        # Prepare job arguments
        job_args = []
        for job in jobs[:batch_size]:  # Submit in batches if needed
            args = (
                job['fasta_path'],
                job['output_path'],
                boltz_params
            )
            job_args.append(args)
        
        # Submit job array
        submitted_jobs = executor.map_array(boltz_job_function, job_args)
        
        # Update job metadata with SLURM job references
        for job, slurm_job in zip(jobs[:batch_size], submitted_jobs):
            job['slurm_job_id'] = slurm_job.job_id
            job['status'] = 'submitted'
            
        return submitted_jobs
    
    def _update_job_status(self, job_info: Dict, submitted_jobs: List[submitit.Job]):
        """Update and save job status after submission."""
        self._save_job_status(job_info)
        print(f"Submitted {len(submitted_jobs)} jobs to SLURM queue")
    
    def _monitor_jobs(self, job_info: Dict):
        """Monitor job progress using SLURM commands directly."""
        jobs_with_slurm = [job for job in job_info['jobs'] if job.get('slurm_job_id')]
        
        if not jobs_with_slurm:
            return
            
        print(f"Monitoring {len(jobs_with_slurm)} jobs...")
        
        with tqdm(total=len(jobs_with_slurm), desc="Structure Prediction Progress") as pbar:
            consecutive_checks_without_running_jobs = 0
            max_checks_after_queue_empty = 5  # Wait 5 cycles after queue is empty
            
            while True:
                # Get current SLURM job status
                running_jobs = self._get_slurm_job_status()
                
                # Check if any of our jobs are still in the queue
                our_jobs_in_queue = False
                for job in jobs_with_slurm:
                    job_id = str(job.get('slurm_job_id', ''))
                    if job_id in running_jobs or any(job_id in jid for jid in running_jobs.keys()):
                        our_jobs_in_queue = True
                        break
                
                for job in jobs_with_slurm:
                    if job['status'] in ['completed', 'failed']:
                        continue
                    
                    job_id = str(job.get('slurm_job_id', ''))
                    output_path = Path(job['output_path'])
                    
                    # First priority: Check if output exists (job completed successfully)
                    if output_path.exists() and any(output_path.iterdir()):
                        if job['status'] != 'completed':
                            job['status'] = 'completed'
                        continue
                    
                    # Check if job is in SLURM queue
                    job_found_in_queue = False
                    current_job_status = None
                    
                    # Look for exact match or array job match
                    if job_id in running_jobs:
                        job_found_in_queue = True
                        current_job_status = running_jobs[job_id]
                    else:
                        # Check for array job format (e.g., "12345_1" might appear as "12345_1" or just be part of "12345")
                        for queue_job_id, status in running_jobs.items():
                            if job_id in queue_job_id or queue_job_id in job_id:
                                job_found_in_queue = True
                                current_job_status = status
                                break
                    
                    if job_found_in_queue:
                        # Job is in queue, update status based on SLURM status
                        if current_job_status in ['COMPLETED', 'COMPLETING']:
                            # Job finished, wait a bit for file system sync
                            time.sleep(10)
                            if output_path.exists() and any(output_path.iterdir()):
                                job['status'] = 'completed'
                            # Don't mark as failed yet, might need more time for files to appear
                        elif current_job_status in ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY']:
                            job['status'] = 'failed'
                            job['error_message'] = f'SLURM job {current_job_status.lower()}'
                        else:
                            # Job is PENDING, RUNNING, etc. - keep waiting
                            job['status'] = 'running'
                    else:
                        # Job not found in queue
                        if our_jobs_in_queue:
                            # Other jobs are still running, this one might have finished
                            # Don't mark as failed immediately, check output first
                            if output_path.exists() and any(output_path.iterdir()):
                                job['status'] = 'completed'
                            # Otherwise, keep waiting - don't mark as failed yet
                        else:
                            # No jobs in queue anymore, final check
                            if consecutive_checks_without_running_jobs >= max_checks_after_queue_empty:
                                if output_path.exists() and any(output_path.iterdir()):
                                    job['status'] = 'completed'
                                else:
                                    job['status'] = 'failed'
                                    job['error_message'] = 'Job completed but no output found'
                
                # Update progress bar
                completed = sum(1 for job in jobs_with_slurm if job['status'] == 'completed')
                failed = sum(1 for job in jobs_with_slurm if job['status'] == 'failed')
                running = sum(1 for job in jobs_with_slurm if job['status'] in ['running', 'pending', 'submitted'])
                
                current_done = completed + failed
                pbar.n = current_done
                pbar.refresh()
                
                # Update status display
                if current_done > 0 or running > 0:
                    pbar.set_postfix_str(f"✓{completed} ✗{failed} ⏳{running}")
                
                # Check if all jobs are done
                if current_done >= len(jobs_with_slurm):
                    break
                
                # Track consecutive checks without running jobs
                if not our_jobs_in_queue:
                    consecutive_checks_without_running_jobs += 1
                else:
                    consecutive_checks_without_running_jobs = 0
                    
                # Save status and wait
                self._save_job_status(job_info)
                time.sleep(30)  # Check every 30 seconds
    
    def _get_slurm_job_status(self) -> Dict[str, str]:
        """Get status of current SLURM jobs using squeue."""
        try:
            import subprocess
            result = subprocess.run(
                ['squeue', '--format=%i,%t', '--noheader', '-u', os.environ.get('USER', '')],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return {}
            
            job_status = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        job_id = parts[0].strip()
                        status = parts[1].strip()
                        job_status[job_id] = status
            
            return job_status
            
        except Exception as e:
            print(f"Warning: Could not check SLURM status: {e}")
            return {}
    
    def _collect_results(self, job_info: Dict) -> Dict[str, Any]:
        """Collect and organize final results."""
        summary = {
            'total_jobs': len(job_info['jobs']),
            'completed': 0,
            'failed': 0,
            'completed_jobs': [],
            'failed_jobs': []
        }
        
        for job in job_info['jobs']:
            output_path = Path(job['output_path'])
            
            # Final check for completion
            if output_path.exists() and any(output_path.iterdir()):
                job['status'] = 'completed'
                summary['completed'] += 1
                summary['completed_jobs'].append({
                    'sequence_name': job['sequence_name'],
                    'output_path': job['output_path']
                })
            else:
                job['status'] = 'failed'
                summary['failed'] += 1
                summary['failed_jobs'].append({
                    'sequence_name': job['sequence_name'],
                    'error': job.get('error_message', 'Unknown error')
                })
        
        # Save final status
        self._save_job_status(job_info)
        return summary
    
    def _write_summary(self, summary: Dict[str, Any]):
        """Write human-readable summary to file."""
        summary_file = self.output_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Parallel Structure Prediction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total jobs: {summary['total_jobs']}\n")
            f.write(f"Completed: {summary['completed']}\n")
            f.write(f"Failed: {summary['failed']}\n\n")
            
            if summary['completed_jobs']:
                f.write("Completed Jobs:\n")
                f.write("-" * 20 + "\n")
                for job in summary['completed_jobs']:
                    f.write(f"  {job['sequence_name']}: {job['output_path']}\n")
                f.write("\n")
            
            if summary['failed_jobs']:
                f.write("Failed Jobs:\n")
                f.write("-" * 20 + "\n")
                for job in summary['failed_jobs']:
                    f.write(f"  {job['sequence_name']}: {job['error']}\n")
                f.write("\n")
        
        print(f"Summary written to: {summary_file}")