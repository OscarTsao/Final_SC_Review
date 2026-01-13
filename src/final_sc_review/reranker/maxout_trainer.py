"""Maximum GPU utilization trainer for reranker fine-tuning.

Implements all GPU optimizations:
- Mixed precision training (BF16/FP16 via AMP)
- Gradient accumulation for larger effective batch sizes
- Multi-worker DataLoader with prefetching and pin_memory
- torch.compile for kernel fusion (PyTorch 2.0+)
- Flash Attention 2 (if available)
- Gradient checkpointing for memory efficiency
- Learning rate scheduling with warmup
- Parallel HPO with Optuna

Training Regimes (from research plan Section 8):
- R0: Off-the-shelf inference (no training)
- R1: Supervised binary (BCE)
- R2: Pairwise ranking (RankNet)
- R3: Listwise ranking (ListNet, ListMLE, PListMLE, LambdaLoss)
- R4: Curriculum (BCE warmup → LambdaLoss)
- R5: Distillation (teacher → student)

Loss Functions (from research plan Section 7):
- Pointwise: BCE, Focal
- Pairwise: RankNet, MarginRanking, Softplus
- Listwise: ListNet, ListMLE, PListMLE, LambdaLoss, ApproxNDCG
- Contrastive: InfoNCE
- Distillation: MSE, MarginMSE
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from final_sc_review.reranker.losses import (
    LossConfig,
    compute_loss,
    hybrid_loss,  # Legacy compatibility
)
from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class MaxoutConfig:
    """Configuration for maximum GPU utilization training."""

    # Model
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Training Regime (R0-R5 from research plan)
    training_regime: str = "R3"  # R1=BCE, R2=RankNet, R3=Listwise, R4=Curriculum

    # Loss Type Selection (Section 7 of research plan)
    pointwise_type: str = "bce"  # bce, focal
    pairwise_type: str = "ranknet"  # ranknet, margin_ranking, pairwise_softplus
    listwise_type: str = "listnet"  # listnet, listmle, plistmle, lambda, approx_ndcg

    # Loss weights
    w_list: float = 1.0
    w_pair: float = 0.5
    w_point: float = 0.1
    temperature: float = 1.0
    max_pairs_per_group: Optional[int] = 50

    # Loss-specific hyperparameters
    sigma: float = 1.0  # For RankNet/Lambda
    margin: float = 1.0  # For margin ranking
    focal_alpha: float = 0.25  # For focal loss
    focal_gamma: float = 2.0  # For focal loss
    ndcg_k: int = 10  # For Lambda/ApproxNDCG

    # Curriculum training (R4)
    curriculum_enabled: bool = False
    curriculum_warmup_epochs: int = 1  # BCE warmup before switching
    curriculum_final_loss: str = "lambda"  # Final loss to switch to

    # GPU Optimizations
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # or "float16"
    use_compile: bool = False  # torch.compile (PyTorch 2.0+)
    compile_mode: str = "reduce-overhead"  # max-autotune, reduce-overhead, default
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Output
    output_dir: str = "outputs/training"
    seed: int = 42
    early_stopping_patience: int = 3

    # Logging
    log_interval: int = 10
    eval_interval: int = 100

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def get_loss_config(self, epoch: int = 0) -> LossConfig:
        """Get loss configuration, handling curriculum training."""
        if self.curriculum_enabled and epoch < self.curriculum_warmup_epochs:
            # Curriculum warmup: BCE only
            return LossConfig(
                pointwise_type="bce",
                pairwise_type="pairwise_softplus",
                listwise_type="listnet",
                w_list=0.0,
                w_pair=0.0,
                w_point=1.0,
                temperature=self.temperature,
                max_pairs_per_group=self.max_pairs_per_group,
            )
        elif self.curriculum_enabled:
            # Curriculum main phase: switch to target loss
            return LossConfig(
                pointwise_type=self.pointwise_type,
                pairwise_type=self.pairwise_type,
                listwise_type=self.curriculum_final_loss,
                w_list=self.w_list,
                w_pair=self.w_pair,
                w_point=self.w_point,
                temperature=self.temperature,
                sigma=self.sigma,
                margin=self.margin,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                ndcg_k=self.ndcg_k,
                max_pairs_per_group=self.max_pairs_per_group,
            )
        else:
            # Standard training
            return LossConfig(
                pointwise_type=self.pointwise_type,
                pairwise_type=self.pairwise_type,
                listwise_type=self.listwise_type,
                w_list=self.w_list,
                w_pair=self.w_pair,
                w_point=self.w_point,
                temperature=self.temperature,
                sigma=self.sigma,
                margin=self.margin,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                ndcg_k=self.ndcg_k,
                max_pairs_per_group=self.max_pairs_per_group,
            )


@dataclass
class MaxoutHPOConfig:
    """Configuration for parallel HPO.

    Searches over all dimensions from the research plan:
    - Training hyperparameters (LR, batch size, warmup)
    - Loss weights (w_list, w_pair, w_point)
    - Loss function types (listwise, pairwise, pointwise)
    - Loss-specific hyperparameters (temperature, sigma, margin)
    """

    # HPO settings
    n_trials: int = 100
    n_jobs: int = 1  # Parallel trials (set to -1 for CPU count)
    timeout: Optional[int] = None  # Seconds

    # Search space: training hyperparameters
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 2e-5, 5e-5, 1e-4])
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    warmup_ratios: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])

    # Search space: loss weights
    w_list_range: Tuple[float, float] = (0.5, 2.0)
    w_pair_range: Tuple[float, float] = (0.1, 1.0)
    w_point_range: Tuple[float, float] = (0.0, 0.5)

    # Search space: loss function types (Section 7 of research plan)
    listwise_types: List[str] = field(default_factory=lambda: [
        "listnet", "listmle", "plistmle", "lambda", "approx_ndcg"
    ])
    pairwise_types: List[str] = field(default_factory=lambda: [
        "ranknet", "margin_ranking", "pairwise_softplus"
    ])
    pointwise_types: List[str] = field(default_factory=lambda: ["bce", "focal"])

    # Search space: loss-specific hyperparameters
    temperature_range: Tuple[float, float] = (0.5, 2.0)
    sigma_range: Tuple[float, float] = (0.5, 2.0)  # For RankNet/Lambda
    margin_range: Tuple[float, float] = (0.5, 2.0)  # For margin ranking
    ndcg_k_choices: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Search space: curriculum training
    search_curriculum: bool = True  # Whether to search over curriculum vs non-curriculum

    # Optuna settings
    study_name: str = "reranker_training_hpo"
    storage: Optional[str] = None  # SQLite for parallel
    pruner: str = "median"  # median, hyperband, none


def get_amp_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    else:
        return torch.float32


class OptimizedDataLoader:
    """Factory for creating optimized DataLoaders."""

    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        collate_fn: Callable,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> DataLoader:
        """Create an optimized DataLoader with prefetching."""
        # Disable persistent_workers if num_workers is 0
        if num_workers == 0:
            persistent_workers = False
            prefetch_factor = None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers,
            drop_last=shuffle,  # Drop last for training only
        )


class MaxoutRerankerTrainer:
    """Trainer with maximum GPU utilization for reranker fine-tuning."""

    def __init__(self, config: MaxoutConfig):
        set_seed(config.seed)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model with optimizations
        self._init_model()

        # Mixed precision setup
        self.amp_dtype = get_amp_dtype(config.amp_dtype)
        self.scaler = GradScaler("cuda", enabled=config.use_amp and config.amp_dtype == "float16")

        # Random generator for loss sampling
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.seed)

        logger.info(f"MaxoutTrainer initialized on {self.device}")
        logger.info(f"  Effective batch size: {config.effective_batch_size()}")
        logger.info(f"  AMP: {config.use_amp} ({config.amp_dtype})")
        logger.info(f"  Compile: {config.use_compile}")
        logger.info(f"  Flash Attention: {config.use_flash_attention}")
        logger.info(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")

    def _init_model(self) -> None:
        """Initialize model with all optimizations."""
        config = self.config

        # Model kwargs for Flash Attention
        model_kwargs = {
            "num_labels": 1,
            "trust_remote_code": True,
        }

        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = get_amp_dtype(config.amp_dtype)

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                **model_kwargs,
            )
        except Exception as e:
            logger.warning(f"Flash Attention failed, falling back: {e}")
            # Fallback without flash attention
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=1,
                trust_remote_code=True,
            )

        # Set pad token id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Gradient checkpointing
        if config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)

        # torch.compile (PyTorch 2.0+)
        if config.use_compile and hasattr(torch, "compile"):
            logger.info(f"Compiling model with mode={config.compile_mode}")
            self.model = torch.compile(self.model, mode=config.compile_mode)

    def _collate_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate grouped examples into a flattened batch with group sizes."""
        group_sizes: List[int] = []
        pairs: List[List[str]] = []
        labels: List[float] = []

        for ex in batch:
            query = ex["query"]
            sentences = ex["sentences"]
            ex_labels = ex["labels"]
            group_sizes.append(len(sentences))
            for sent, label in zip(sentences, ex_labels):
                pairs.append([query, sent])
                labels.append(float(label))

        enc = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float),
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
        }

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create optimized DataLoader."""
        return OptimizedDataLoader.create(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_batch,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: List[int],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with configurable loss types and optional AMP.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Target labels
            group_sizes: Size of each group in the batch
            epoch: Current epoch (for curriculum training)

        Returns:
            Tuple of (loss tensor, loss component dict for logging)
        """
        cfg = self.config

        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=cfg.use_amp):
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.squeeze(-1)

            # Get loss config (handles curriculum training)
            loss_config = cfg.get_loss_config(epoch)

            loss, loss_components = compute_loss(
                logits=logits.float(),  # Cast back for loss stability
                labels=labels,
                group_sizes=group_sizes,
                config=loss_config,
                rng=self.rng,
            )

        return loss, loss_components

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, Any]:
        """Train with all GPU optimizations."""
        cfg = self.config

        # Create data loaders
        train_loader = self._make_loader(train_dataset, shuffle=True)

        # Calculate total steps
        steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
        total_steps = steps_per_epoch * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            fused=torch.cuda.is_available(),  # Fused optimizer for CUDA
        )

        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Training for {cfg.num_epochs} epochs ({total_steps} steps)")
        logger.info(f"Warmup steps: {warmup_steps}")

        # Training loop
        training_stats = []
        start_time = time.time()

        for epoch in range(cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            # Log curriculum phase change
            if cfg.curriculum_enabled:
                if epoch < cfg.curriculum_warmup_epochs:
                    logger.info(f"Epoch {epoch + 1}: Curriculum warmup (BCE only)")
                elif epoch == cfg.curriculum_warmup_epochs:
                    logger.info(f"Epoch {epoch + 1}: Switching to {cfg.curriculum_final_loss} loss")

            for step, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                group_sizes = batch["group_sizes"].tolist()

                # Forward pass with AMP (pass epoch for curriculum training)
                loss, loss_components = self._compute_loss(
                    input_ids, attention_mask, labels, group_sizes, epoch
                )
                loss = loss / cfg.gradient_accumulation_steps

                # Backward pass with gradient scaling
                if cfg.use_amp and cfg.amp_dtype == "float16":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * cfg.gradient_accumulation_steps

                # Gradient accumulation step
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.use_amp and cfg.amp_dtype == "float16":
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.max_grad_norm
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.max_grad_norm
                        )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # Logging
                    if self.global_step % cfg.log_interval == 0:
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {loss.item() * cfg.gradient_accumulation_steps:.4f} | "
                            f"LR: {lr:.2e}"
                        )

            # End of epoch
            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = self.evaluate(val_dataset, epoch)

            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - start_time,
            }
            training_stats.append(epoch_stats)

            logger.info(
                f"Epoch {epoch + 1}/{cfg.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Callback
            if callback:
                callback(epoch_stats)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save(cfg.output_dir)
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Final stats
        total_time = time.time() - start_time
        final_stats = {
            "total_epochs": len(training_stats),
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": total_time,
            "steps_per_second": self.global_step / total_time,
            "training_history": training_stats,
        }

        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return final_stats

    def evaluate(self, dataset: Dataset, epoch: int = 0) -> float:
        """Evaluate with optimized inference."""
        self.model.eval()
        loader = self._make_loader(dataset, shuffle=False)
        total_loss = 0.0

        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                group_sizes = batch["group_sizes"].tolist()

                loss, _ = self._compute_loss(input_ids, attention_mask, labels, group_sizes, epoch)
                total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def save(self, output_dir: str) -> None:
        """Save model and tokenizer."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Unwrap compiled model if necessary
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod

        model_to_save.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        logger.info(f"Model saved to {out}")

    def cleanup(self) -> None:
        """Free GPU memory aggressively."""
        # Clear model and tokenizer
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'scaler'):
            del self.scaler
        if hasattr(self, 'rng'):
            del self.rng

        # Force garbage collection
        gc.collect()
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()


class MaxoutHPO:
    """Parallel HPO for reranker training with Optuna."""

    def __init__(
        self,
        base_config: MaxoutConfig,
        hpo_config: MaxoutHPOConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.base_config = base_config
        self.hpo_config = hpo_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _create_trial_config(self, trial) -> MaxoutConfig:
        """Create config from Optuna trial.

        Samples over all dimensions from the research plan:
        - Training hyperparameters (LR, batch size, warmup)
        - Loss weights (w_list, w_pair, w_point)
        - Loss function types (listwise, pairwise, pointwise)
        - Loss-specific hyperparameters (temperature, sigma, margin)
        - Curriculum training (optional)
        """
        import copy

        hpo = self.hpo_config
        config = copy.deepcopy(self.base_config)

        # === Training hyperparameters ===
        config.learning_rate = trial.suggest_categorical("learning_rate", hpo.learning_rates)
        config.batch_size = trial.suggest_categorical("batch_size", hpo.batch_sizes)
        config.warmup_ratio = trial.suggest_categorical("warmup_ratio", hpo.warmup_ratios)

        # === Loss function types (Section 7) ===
        config.listwise_type = trial.suggest_categorical("listwise_type", hpo.listwise_types)
        config.pairwise_type = trial.suggest_categorical("pairwise_type", hpo.pairwise_types)
        config.pointwise_type = trial.suggest_categorical("pointwise_type", hpo.pointwise_types)

        # === Loss weights ===
        config.w_list = trial.suggest_float("w_list", *hpo.w_list_range)
        config.w_pair = trial.suggest_float("w_pair", *hpo.w_pair_range)
        config.w_point = trial.suggest_float("w_point", *hpo.w_point_range)

        # === Loss-specific hyperparameters ===
        config.temperature = trial.suggest_float("temperature", *hpo.temperature_range)

        # Only sample sigma if using RankNet or Lambda loss
        if config.listwise_type == "lambda" or config.pairwise_type == "ranknet":
            config.sigma = trial.suggest_float("sigma", *hpo.sigma_range)

        # Only sample margin if using margin ranking loss
        if config.pairwise_type == "margin_ranking":
            config.margin = trial.suggest_float("margin", *hpo.margin_range)

        # Only sample ndcg_k if using Lambda or ApproxNDCG
        if config.listwise_type in ["lambda", "approx_ndcg"]:
            config.ndcg_k = trial.suggest_categorical("ndcg_k", hpo.ndcg_k_choices)

        # === Curriculum training (R4) ===
        if hpo.search_curriculum:
            config.curriculum_enabled = trial.suggest_categorical(
                "curriculum_enabled", [True, False]
            )
            if config.curriculum_enabled:
                config.curriculum_final_loss = trial.suggest_categorical(
                    "curriculum_final_loss", ["lambda", "listmle", "plistmle"]
                )

        # Set trial-specific output dir
        config.output_dir = f"{self.base_config.output_dir}/trial_{trial.number}"

        return config

    def _objective(self, trial) -> float:
        """Optuna objective function."""
        import optuna

        config = self._create_trial_config(trial)

        try:
            trainer = MaxoutRerankerTrainer(config)

            # Callback for pruning
            def pruning_callback(stats: Dict) -> None:
                trial.report(stats["val_loss"], stats["epoch"])
                if trial.should_prune():
                    raise optuna.TrialPruned()

            stats = trainer.train(
                self.train_dataset,
                self.val_dataset,
                callback=pruning_callback,
            )

            trainer.cleanup()
            return stats["best_val_loss"]

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    def run(self) -> Dict[str, Any]:
        """Run parallel HPO."""
        import optuna

        hpo = self.hpo_config

        # Create pruner
        if hpo.pruner == "median":
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
        elif hpo.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        # Create or load study
        study = optuna.create_study(
            study_name=hpo.study_name,
            storage=hpo.storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=hpo.n_trials,
            n_jobs=hpo.n_jobs,
            timeout=hpo.timeout,
            show_progress_bar=True,
        )

        # Results
        best_trial = study.best_trial
        results = {
            "best_trial_number": best_trial.number,
            "best_val_loss": best_trial.value,
            "best_params": best_trial.params,
            "n_trials": len(study.trials),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }

        logger.info(f"HPO complete. Best val loss: {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")

        return results


def benchmark_gpu_utilization(
    config: MaxoutConfig,
    dataset: Dataset,
    n_batches: int = 50,
) -> Dict[str, Any]:
    """Benchmark GPU utilization for a config."""
    import subprocess

    trainer = MaxoutRerankerTrainer(config)
    loader = trainer._make_loader(dataset, shuffle=True)

    # Warmup (no gradients)
    warmup_batches = min(3, len(loader))
    trainer.model.zero_grad(set_to_none=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup_batches:
                break
            input_ids = batch["input_ids"].to(trainer.device)
            attention_mask = batch["attention_mask"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            _, _ = trainer._compute_loss(input_ids, attention_mask, labels, batch["group_sizes"].tolist())

    # Benchmark (with gradients, but clear after each step)
    torch.cuda.synchronize()
    start = time.time()

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Clear gradients before each step
        trainer.model.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(trainer.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(trainer.device, non_blocking=True)
        labels = batch["labels"].to(trainer.device, non_blocking=True)
        loss, _ = trainer._compute_loss(input_ids, attention_mask, labels, batch["group_sizes"].tolist())
        loss.backward()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Get GPU stats
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
    except Exception:
        gpu_util, mem_used, mem_total = "N/A", "N/A", "N/A"

    trainer.cleanup()

    return {
        "batches_per_second": n_batches / elapsed,
        "seconds_per_batch": elapsed / n_batches,
        "gpu_utilization": gpu_util,
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
        "config": {
            "batch_size": config.batch_size,
            "use_amp": config.use_amp,
            "amp_dtype": config.amp_dtype,
            "use_compile": config.use_compile,
            "num_workers": config.num_workers,
        },
    }
