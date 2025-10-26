"""
GPU Scheduler for Parallel FL Training

This module manages parallel execution of FL experiments across multiple GPUs,
with task queuing and load balancing.
"""

import os
import sys
import json
import time
import subprocess
import threading
import queue
import signal
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # FedAvg parameters
    local_epochs: int = 5
    global_rounds: int = 20
    
    # Solo/Combined parameters
    epochs: int = 100
    
    # Common parameters
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Script paths
    scripts: Dict[str, str] = field(default_factory=lambda: {
        'fedavg': 'src/autorun/fedavg.py',
        'fedprox': 'src/autorun/fedprox.py',
        'scaffold': 'src/autorun/scaffold.py',
        'fedov': 'src/autorun/fedov.py',
        'fedtree': 'src/autorun/fedtree.py',
        'solo': 'src/autorun/solo.py',
        'combined': 'src/autorun/solo.py'  # Use solo.py for combined training
    })


@dataclass
class GPUTask:
    """Represents a single FL training task."""
    pair_id: str
    data_dir: str
    config_file: str
    output_file: str
    gpu_id: int
    task_type: str = 'fedavg'  # 'fedavg', 'solo', 'combined'
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    retry_count: int = 0
    max_retries: int = 2


class GPUScheduler:
    """
    GPU scheduler for parallel FL training experiments.
    
    Manages task distribution across multiple GPUs with load balancing
    and concurrent execution.
    """
    
    def __init__(self, 
                 num_gpus: int = 4,
                 max_concurrent_per_gpu: int = 2,
                 base_output_dir: str = "out/autorun/results",
                 log_dir: str = "out/autorun/logs",
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialize GPU scheduler.
        
        Args:
            num_gpus: Number of available GPUs (default: 4)
            max_concurrent_per_gpu: Max concurrent tasks per GPU (default: 2)
            base_output_dir: Base directory for output files
            log_dir: Directory for log files
            training_config: Training configuration object
        """
        self.num_gpus = num_gpus
        self.max_concurrent_per_gpu = max_concurrent_per_gpu
        self.base_output_dir = Path(base_output_dir)
        self.log_dir = Path(log_dir)
        self.training_config = training_config or TrainingConfig()
        
        # Validate configuration
        self._validate_configuration()
        
        # Create directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Task management
        self.task_queue = queue.Queue()
        self.running_tasks = {gpu_id: [] for gpu_id in range(num_gpus)}
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Threading
        self.scheduler_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        logger.info(f"Initialized GPU scheduler with {num_gpus} GPUs, "
                   f"max {max_concurrent_per_gpu} concurrent tasks per GPU")
    
    def _validate_configuration(self) -> None:
        """Validate the scheduler configuration."""
        # Check if training scripts exist
        for task_type, script_path in self.training_config.scripts.items():
            if not Path(script_path).exists():
                logger.warning(f"Training script not found: {script_path} for task type {task_type}")
        
        # Validate GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                if self.num_gpus > available_gpus:
                    logger.warning(f"Requested {self.num_gpus} GPUs but only {available_gpus} available")
                    self.num_gpus = min(self.num_gpus, available_gpus)
            else:
                logger.warning("CUDA not available, GPU scheduling may fail")
        except ImportError:
            logger.warning("PyTorch not available for GPU validation")
    
    def add_task(self, pair_id: str, data_dir: str, config_file: str, 
                task_type: str = 'fedavg') -> bool:
        """
        Add a training task to the queue.
        
        Args:
            pair_id: Unique identifier for the database pair
            data_dir: Directory containing processed data
            config_file: Path to configuration file
            task_type: Type of training task ('fedavg', 'solo', 'combined')
            
        Returns:
            True if task added successfully, False otherwise
        """
        # Validate inputs
        if not self._validate_task_inputs(pair_id, data_dir, config_file, task_type):
            return False
        
        # Create output file path
        output_file = self.base_output_dir / f"{pair_id}_{task_type}_results.json"
        
        task = GPUTask(
            pair_id=pair_id,
            data_dir=data_dir,
            config_file=config_file,
            output_file=str(output_file),
            gpu_id=-1,  # Will be assigned when scheduled
            task_type=task_type
        )
        
        self.task_queue.put(task)
        logger.info(f"Added task: {pair_id} ({task_type})")
        return True
    
    def _validate_task_inputs(self, pair_id: str, data_dir: str, 
                             config_file: str, task_type: str) -> bool:
        """Validate task inputs before adding to queue."""
        # Check task type
        if task_type not in self.training_config.scripts:
            logger.error(f"Unknown task type: {task_type}")
            return False
        
        # Check data directory
        if not Path(data_dir).exists():
            logger.error(f"Data directory not found: {data_dir}")
            return False
        
        # Check config file
        if not Path(config_file).exists():
            logger.error(f"Config file not found: {config_file}")
            return False
        
        # Check if training script exists
        script_path = self.training_config.scripts[task_type]
        if not Path(script_path).exists():
            logger.error(f"Training script not found: {script_path}")
            return False
        
        return True
    
    def get_available_gpu(self) -> Optional[int]:
        """
        Find an available GPU with capacity for more tasks.
        
        Returns:
            GPU ID if available, None otherwise
        """
        for gpu_id in range(self.num_gpus):
            if len(self.running_tasks[gpu_id]) < self.max_concurrent_per_gpu:
                return gpu_id
        return None
    
    def create_training_command(self, task: GPUTask) -> List[str]:
        """
        Create command line arguments for training script.
        
        Args:
            task: Training task
            
        Returns:
            List of command arguments
        """
        script_path = self.training_config.scripts.get(task.task_type)
        if not script_path:
            raise ValueError(f"No script configured for task type: {task.task_type}")
        
        cmd = [
            sys.executable, 
            script_path,
            "--data-dir", task.data_dir,
            "--config-file", task.config_file,
            "--output-file", task.output_file,
            "--device", "cuda:0",  # Always use cuda:0 since CUDA_VISIBLE_DEVICES makes only one GPU visible
            "--hidden-dims"
        ]
        
        # Add hidden dimensions
        cmd.extend([str(dim) for dim in self.training_config.hidden_dims])
        
        # Add common parameters
        cmd.extend([
            "--learning-rate", str(self.training_config.learning_rate),
            "--batch-size", str(self.training_config.batch_size)
        ])
        
        # Add task-specific parameters
        if task.task_type in ['fedavg', 'fedprox', 'scaffold']:
            cmd.extend([
                "--local-epochs", str(self.training_config.local_epochs), 
                "--global-rounds", str(self.training_config.global_rounds)
            ])
            # FedProx specific
            if task.task_type == 'fedprox':
                cmd.extend(["--mu", "0.001"])
        elif task.task_type == 'fedov':
            # FedOV uses more local epochs for one-shot learning
            cmd.extend([
                "--local-epochs", "10",
                "--augmentation-rate", "0.3",
                "--outlier-threshold", "0.5"
            ])
        elif task.task_type == 'fedtree':
            # FedTree uses XGBoost parameters
            cmd.extend([
                "--num-trees", "100",
                "--max-depth", "6"
            ])
        elif task.task_type in ['solo', 'combined']:
            cmd.extend([
                "--epochs", str(self.training_config.epochs),
                "--algorithm", task.task_type
            ])
        
        return cmd
    
    def start_task(self, task: GPUTask) -> bool:
        """
        Start a training task on an assigned GPU.
        
        Args:
            task: Training task to start
            
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Create log file
            log_file = self.log_dir / f"{task.pair_id}_{task.task_type}_gpu{task.gpu_id}.log"
            
            # Create command
            cmd = self.create_training_command(task)
            
            # Start process with GPU isolation
            # CUDA_VISIBLE_DEVICES=N makes only GPU N visible to the subprocess,
            # and it becomes cuda:0 within that process. This provides clean GPU isolation.
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(task.gpu_id))
                )
            
            task.process = process
            task.status = 'running'
            task.start_time = datetime.now()
            
            logger.info(f"Started task {task.pair_id} ({task.task_type}) on GPU {task.gpu_id}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Log file: {log_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start task {task.pair_id}: {e}")
            task.status = 'failed'
            task.error_message = str(e)
            return False
    
    def check_running_tasks(self) -> None:
        """Check status of running tasks and handle completion."""
        with self.lock:
            for gpu_id in range(self.num_gpus):
                completed_tasks = []
                
                for task in self.running_tasks[gpu_id]:
                    if task.process is None:
                        continue
                    
                    # Check if process has finished
                    return_code = task.process.poll()
                    
                    if return_code is not None:
                        # Process finished
                        task.end_time = datetime.now()
                        duration = task.end_time - task.start_time
                        
                        if return_code == 0:
                            task.status = 'completed'
                            self.completed_tasks.append(task)
                            logger.info(f"Task {task.pair_id} ({task.task_type}) completed "
                                      f"on GPU {gpu_id} in {duration}")
                        else:
                            task.status = 'failed'
                            task.error_message = f"Process exited with code {return_code}"
                            
                            # Check if we should retry
                            if task.retry_count < task.max_retries:
                                task.retry_count += 1
                                task.status = 'pending'
                                task.gpu_id = -1
                                task.process = None
                                task.start_time = None
                                task.end_time = None
                                self.task_queue.put(task)
                                logger.warning(f"Task {task.pair_id} ({task.task_type}) failed "
                                             f"on GPU {gpu_id}, retrying ({task.retry_count}/{task.max_retries})")
                            else:
                                self.failed_tasks.append(task)
                                logger.error(f"Task {task.pair_id} ({task.task_type}) failed "
                                           f"on GPU {gpu_id} after {duration} (max retries reached)")
                        
                        completed_tasks.append(task)
                
                # Remove completed tasks from running list
                for task in completed_tasks:
                    self.running_tasks[gpu_id].remove(task)
    
    def scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Starting scheduler loop")
        
        while self.running:
            try:
                # Check running tasks
                self.check_running_tasks()
                
                # Try to start new tasks
                while not self.task_queue.empty():
                    gpu_id = self.get_available_gpu()
                    
                    if gpu_id is None:
                        # No available GPUs, wait a bit
                        break
                    
                    try:
                        task = self.task_queue.get_nowait()
                        task.gpu_id = gpu_id
                        
                        if self.start_task(task):
                            with self.lock:
                                self.running_tasks[gpu_id].append(task)
                        else:
                            self.failed_tasks.append(task)
                        
                    except queue.Empty:
                        break
                
                # Wait before next iteration
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)
        
        logger.info("Scheduler loop ended")
    
    def start_scheduling(self) -> None:
        """Start the scheduler in a separate thread."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self.scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Scheduler started")
    
    def stop_scheduling(self) -> None:
        """Stop the scheduler and wait for completion."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping scheduler...")
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        # Wait for running tasks to complete or kill them
        self._cleanup_running_tasks()
        
        logger.info("Scheduler stopped")
    
    def _cleanup_running_tasks(self) -> None:
        """Clean up any remaining running tasks."""
        logger.info("Cleaning up running tasks...")
        
        for gpu_id in range(self.num_gpus):
            for task in self.running_tasks[gpu_id]:
                if task.process and task.process.poll() is None:
                    logger.info(f"Terminating task {task.pair_id} on GPU {gpu_id}")
                    task.process.terminate()
                    
                    # Wait a bit for graceful termination
                    try:
                        task.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing task {task.pair_id}")
                        task.process.kill()
                    
                    task.status = 'terminated'
                    task.end_time = datetime.now()
    
    def wait_for_completion(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for all tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all tasks completed, False if timeout
        """
        start_time = time.time()
        last_status_time = start_time
        
        while True:
            # Check if all tasks are done
            total_running = sum(len(tasks) for tasks in self.running_tasks.values())
            
            if self.task_queue.empty() and total_running == 0:
                logger.info("All tasks completed")
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout reached after {timeout} seconds")
                return False
            
            # Status update every 30 seconds
            current_time = time.time()
            if current_time - last_status_time >= 30:
                self._log_progress_status()
                last_status_time = current_time
            
            time.sleep(5)  # Check every 5 seconds
    
    def _log_progress_status(self) -> None:
        """Log detailed progress status."""
        total_running = sum(len(tasks) for tasks in self.running_tasks.values())
        total_completed = len(self.completed_tasks)
        total_failed = len(self.failed_tasks)
        queue_size = self.task_queue.qsize()
        
        # Calculate estimated completion time
        if total_completed > 0:
            avg_duration = sum(
                (task.end_time - task.start_time).total_seconds() 
                for task in self.completed_tasks
            ) / total_completed
            remaining_tasks = queue_size + total_running
            eta_seconds = remaining_tasks * avg_duration
            eta_minutes = eta_seconds / 60
            eta_str = f"ETA: {eta_minutes:.1f} minutes"
        else:
            eta_str = "ETA: calculating..."
        
        logger.info(f"PROGRESS - Queue: {queue_size}, Running: {total_running}, "
                   f"Completed: {total_completed}, Failed: {total_failed}, {eta_str}")
        
        # Log GPU utilization
        gpu_status = []
        for gpu_id in range(self.num_gpus):
            running_count = len(self.running_tasks[gpu_id])
            if running_count > 0:
                task_names = [f"{task.pair_id}_{task.task_type}" for task in self.running_tasks[gpu_id]]
                gpu_status.append(f"GPU{gpu_id}: {running_count}/{self.max_concurrent_per_gpu} ({', '.join(task_names)})")
            else:
                gpu_status.append(f"GPU{gpu_id}: idle")
        
        logger.info(f"GPU STATUS - {' | '.join(gpu_status)}")
    
    def get_running_task_details(self) -> List[Dict]:
        """Get detailed information about currently running tasks."""
        running_details = []
        
        for gpu_id in range(self.num_gpus):
            for task in self.running_tasks[gpu_id]:
                if task.start_time:
                    runtime = datetime.now() - task.start_time
                    running_details.append({
                        'pair_id': task.pair_id,
                        'task_type': task.task_type,
                        'gpu_id': gpu_id,
                        'runtime_seconds': runtime.total_seconds(),
                        'retry_count': task.retry_count
                    })
        
        return running_details
    
    def kill_task(self, pair_id: str, task_type: str) -> bool:
        """Kill a specific running task."""
        with self.lock:
            for gpu_id in range(self.num_gpus):
                for task in self.running_tasks[gpu_id]:
                    if task.pair_id == pair_id and task.task_type == task_type:
                        if task.process and task.process.poll() is None:
                            logger.info(f"Killing task {pair_id} ({task_type}) on GPU {gpu_id}")
                            task.process.terminate()
                            task.status = 'killed'
                            task.end_time = datetime.now()
                            return True
        return False
    
    def get_status_summary(self) -> Dict:
        """Get current status summary."""
        total_running = sum(len(tasks) for tasks in self.running_tasks.values())
        
        summary = {
            'queue_size': self.task_queue.qsize(),
            'running_tasks': total_running,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'gpu_utilization': {
                gpu_id: len(tasks) for gpu_id, tasks in self.running_tasks.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def save_final_report(self, output_file: str) -> None:
        """Save final execution report."""
        report = {
            'summary': self.get_status_summary(),
            'completed_tasks': [
                {
                    'pair_id': task.pair_id,
                    'task_type': task.task_type,
                    'gpu_id': task.gpu_id,
                    'start_time': task.start_time.isoformat() if task.start_time else None,
                    'end_time': task.end_time.isoformat() if task.end_time else None,
                    'duration': str(task.end_time - task.start_time) if task.start_time and task.end_time else None,
                    'output_file': task.output_file
                }
                for task in self.completed_tasks
            ],
            'failed_tasks': [
                {
                    'pair_id': task.pair_id,
                    'task_type': task.task_type,
                    'gpu_id': task.gpu_id,
                    'error_message': task.error_message,
                    'start_time': task.start_time.isoformat() if task.start_time else None,
                    'end_time': task.end_time.isoformat() if task.end_time else None
                }
                for task in self.failed_tasks
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to: {output_file}")


def load_processed_pairs(preprocessing_summary_file: str) -> List[Dict]:
    """Load successfully processed pairs from preprocessing summary."""
    with open(preprocessing_summary_file, 'r') as f:
        summary = json.load(f)
    
    successful_pairs = []
    for result in summary['results']:
        if 'error' not in result:
            successful_pairs.append(result)
    
    logger.info(f"Loaded {len(successful_pairs)} successfully processed pairs")
    return successful_pairs


def create_signal_handler(scheduler: GPUScheduler):
    """Create signal handler for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        scheduler.stop_scheduling()
        sys.exit(0)
    
    return signal_handler


def main():
    """Main function for GPU scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU scheduler for FL training")
    parser.add_argument("--preprocessing-summary", type=str, required=True,
                       help="Preprocessing summary file")
    parser.add_argument("--data-dir", type=str, default="data/auto",
                       help="Base data directory (default: data/auto)")
    parser.add_argument("--num-gpus", type=int, default=4,
                       help="Number of GPUs (default: 4)")
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=2,
                       help="Max concurrent tasks per GPU (default: 2)")
    parser.add_argument("--output-dir", type=str, default="out/autorun/results",
                       help="Output directory (default: out/autorun/results)")
    parser.add_argument("--log-dir", type=str, default="out/autorun/logs",
                       help="Log directory (default: out/autorun/logs)")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Timeout in seconds (default: None)")
    
    # Training configuration arguments
    parser.add_argument("--local-epochs", type=int, default=5,
                       help="Local epochs for FedAvg (default: 5)")
    parser.add_argument("--global-rounds", type=int, default=20,
                       help="Global rounds for FedAvg (default: 20)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Epochs for solo/combined training (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[64, 32],
                       help="Hidden dimensions (default: 64 32)")
    
    # Task selection arguments
    parser.add_argument("--task-types", nargs='+', default=['fedavg', 'solo', 'combined'],
                       choices=['fedavg', 'fedprox', 'scaffold', 'fedov', 'fedtree', 'solo', 'combined'],
                       help="Task types to run (default: fedavg, solo, combined)")
    parser.add_argument("--max-pairs", type=int, default=None,
                       help="Maximum number of pairs to process (default: all)")
    parser.add_argument("--skip-validation", action='store_true',
                       help="Skip input validation (faster but riskier)")
    
    # Script path arguments
    parser.add_argument("--fedavg-script", type=str, default="src/autorun/fedavg.py",
                       help="Path to FedAvg script")
    parser.add_argument("--solo-script", type=str, default="src/autorun/solo.py",
                       help="Path to solo training script")
    parser.add_argument("--combined-script", type=str, default="src/autorun/solo.py",
                       help="Path to combined training script")
    
    # Monitoring arguments
    parser.add_argument("--progress-interval", type=int, default=30,
                       help="Progress update interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Create training configuration
    training_config = TrainingConfig(
        local_epochs=args.local_epochs,
        global_rounds=args.global_rounds,
        epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        scripts={
            'fedavg': args.fedavg_script,
            'fedprox': 'src/autorun/fedprox.py',
            'scaffold': 'src/autorun/scaffold.py',
            'fedov': 'src/autorun/fedov.py',
            'fedtree': 'src/autorun/fedtree.py',
            'solo': args.solo_script,
            'combined': args.combined_script
        }
    )
    
    # Load processed pairs
    pairs = load_processed_pairs(args.preprocessing_summary)
    
    if not pairs:
        logger.error("No successfully processed pairs found")
        return 1
    
    # Limit number of pairs if specified
    if args.max_pairs and args.max_pairs < len(pairs):
        pairs = pairs[:args.max_pairs]
        logger.info(f"Limited to first {args.max_pairs} pairs")
    
    # Create scheduler
    scheduler = GPUScheduler(
        num_gpus=args.num_gpus,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        base_output_dir=args.output_dir,
        log_dir=args.log_dir,
        training_config=training_config
    )
    
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, create_signal_handler(scheduler))
    signal.signal(signal.SIGTERM, create_signal_handler(scheduler))
    
    # Add all tasks
    logger.info(f"Adding training tasks for {len(pairs)} pairs...")
    logger.info(f"Task types: {args.task_types}")
    logger.info(f"Training config: local_epochs={args.local_epochs}, "
               f"global_rounds={args.global_rounds}, epochs={args.epochs}, "
               f"lr={args.learning_rate}, batch_size={args.batch_size}")
    
    total_tasks = 0
    failed_to_add = 0
    
    for pair in pairs:
        pair_id = pair['pair_id']
        data_dir = os.path.join(args.data_dir, pair_id)
        config_file = os.path.join(data_dir, 'config.json')
        
        # Add specified task types for each pair
        for task_type in args.task_types:
            if args.skip_validation:
                # Skip validation for speed
                output_file = Path(args.output_dir) / f"{pair_id}_{task_type}_results.json"
                task = GPUTask(
                    pair_id=pair_id,
                    data_dir=data_dir,
                    config_file=config_file,
                    output_file=str(output_file),
                    gpu_id=-1,
                    task_type=task_type
                )
                scheduler.task_queue.put(task)
                total_tasks += 1
            else:
                # Use validation
                if scheduler.add_task(pair_id, data_dir, config_file, task_type):
                    total_tasks += 1
                else:
                    failed_to_add += 1
                    logger.warning(f"Failed to add task: {pair_id} ({task_type})")
    
    if failed_to_add > 0:
        logger.warning(f"Failed to add {failed_to_add} tasks due to validation errors")
    
    logger.info(f"Added {total_tasks} tasks ({len(pairs)} pairs × {len(args.task_types)} algorithms) to scheduler")
    
    if total_tasks == 0:
        logger.error("No tasks were added to the scheduler")
        return 1
    
    # Start scheduling
    try:
        scheduler.start_scheduling()
        
        # Initial status
        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        
        # Wait for completion
        completed = scheduler.wait_for_completion(timeout=args.timeout)
        
        if not completed:
            logger.warning("Not all tasks completed within timeout")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    
    finally:
        # Stop scheduler
        logger.info("Stopping scheduler and cleaning up...")
        scheduler.stop_scheduling()
        
        # Save final report
        report_file = Path(args.output_dir) / "execution_report.json"
        scheduler.save_final_report(str(report_file))
        
        # Print final summary
        summary = scheduler.get_status_summary()
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"  Completed: {summary['completed_tasks']}")
        logger.info(f"  Failed: {summary['failed_tasks']}")
        logger.info(f"  Queue remaining: {summary['queue_size']}")
        logger.info(f"  Still running: {summary['running_tasks']}")
        
        # Calculate success rate
        total_attempted = summary['completed_tasks'] + summary['failed_tasks']
        if total_attempted > 0:
            success_rate = (summary['completed_tasks'] / total_attempted) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")
        
        logger.info(f"  Report saved: {report_file}")
        logger.info("=" * 60)
    
    return 0 if summary['failed_tasks'] == 0 else 1


if __name__ == "__main__":
    exit(main())