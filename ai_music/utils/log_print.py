import os
from datetime import datetime
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback

def print_dataset_statistics(train_loader, val_loader):
    # Print dataset statistics
    
    # Count real and fake songs in train set
    train_dataset = train_loader.dataset
    train_real = (train_dataset.tracks['source'] == 'real').sum()
    train_fake = (train_dataset.tracks['source'] == 'fake').sum()

    
    # Count real and fake songs in val set
    val_dataset = val_loader.dataset
    val_real = (val_dataset.tracks['source'] == 'real').sum()
    val_fake = (val_dataset.tracks['source'] == 'fake').sum()

    
    train_filenames = set(train_dataset.tracks['filename'].values)
    val_filenames = set(val_dataset.tracks['filename'].values)
    overlap = train_filenames.intersection(val_filenames)


    

    # Setup logging
    csv_logger = CSVLogger("lightning_logs", name="")
    progress_log_file = os.path.join(csv_logger.log_dir, "training_progress.txt")
    progress_logger = ProgressLogger(progress_log_file)
    
    # print(f"\nLogging to:")
    # print(f"  CSV: {os.path.join(csv_logger.log_dir, 'metrics.csv')}")
    # print(f"  Progress: {progress_log_file}\n")

    return csv_logger, progress_logger


class ProgressLogger(Callback):
    """Custom callback to log training progress to a text file."""
    
    def __init__(self, log_file, accum_batches=4, log_every=10):
        super().__init__()
        self.log_file = log_file
        self.accum_batches = accum_batches
        self.log_every = log_every
        self.accum_count = 0
        self.accum_loss = 0.0
        self.accum_acc = 0.0
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Create the log file and write header
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def on_train_epoch_start(self, trainer, pl_module):
        with open(self.log_file, 'a') as f:
            f.write(f"\n[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] Training started\n")
            f.flush()
    
    def _write_log(self, line: str):
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")
            f.flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        batch_loss = metrics.get('train_loss', None)
        batch_acc = metrics.get('train_acc', None)

        # Accumulate stats over 4 batches
        if batch_loss is not None:
            self.accum_loss += batch_loss
        if batch_acc is not None:
            self.accum_acc += batch_acc
        self.accum_count += 1

        # When we've accumulated 4 batches, compute the average
        if self.accum_count == self.accum_batches:
            avg_loss = self.accum_loss / self.accum_batches
            avg_acc = self.accum_acc / self.accum_batches

            # Reset accumulators
            self.accum_loss = 0.0
            self.accum_acc = 0.0
            self.accum_count = 0

            # Log every 10th batch group
            if (batch_idx + 1) % self.log_every == 0:
                log_line = f"  Batch {batch_idx + 1:5d}: loss={avg_loss:.4f} acc={avg_acc:.4f}"
                self._write_log(log_line)
    
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        log_line = f"[Epoch {trainer.current_epoch + 1}] Training completed - "
        if 'train_loss' in metrics:
            log_line += f"loss={metrics['train_loss']:.4f} "
        if 'train_acc' in metrics:
            log_line += f"acc={metrics['train_acc']:.4f}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
            f.flush()
    
    def on_validation_epoch_start(self, trainer, pl_module):
        with open(self.log_file, 'a') as f:
            f.write(f"[Epoch {trainer.current_epoch + 1}] Validation started\n")
            f.flush()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        log_line = f"[Epoch {trainer.current_epoch + 1}] Validation completed - "
        if 'val_loss' in metrics:
            log_line += f"loss={metrics['val_loss']:.4f} "
        if 'val_acc' in metrics:
            log_line += f"acc={metrics['val_acc']:.4f}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
            f.write("-"*80 + "\n")
            f.flush()
    
    def on_sanity_check_start(self, trainer, pl_module):
        with open(self.log_file, 'a') as f:
            f.write(f"\n[Sanity Check] Starting validation sanity check...\n")
            f.flush()
    
    def on_sanity_check_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        log_line = f"[Sanity Check] Completed - "
        if 'val_loss' in metrics:
            log_line += f"loss={metrics['val_loss']:.4f} "
        if 'val_acc' in metrics:
            log_line += f"acc={metrics['val_acc']:.4f}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
            f.write("-"*80 + "\n")
            f.flush()
        
        # Also print to console so it's immediately visible
        print(f"\n{log_line}\n")