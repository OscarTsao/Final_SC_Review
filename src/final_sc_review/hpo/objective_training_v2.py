"""Training-stage HPO objective with LoRA/PEFT support and memory optimizations."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import GroupedRerankerDataset, build_grouped_examples
from final_sc_review.reranker.losses import hybrid_loss
from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.seed import set_seed

logger = get_logger(__name__)

# Try to import PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available, LoRA training disabled")


@dataclass
class TrainingConfigV2:
    """Enhanced training config with LoRA and memory options."""
    model_name: str
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    w_list: float
    w_pair: float
    w_point: float
    temperature: float
    max_pairs_per_group: Optional[int]
    seed: int
    # LoRA/PEFT options
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    use_qlora: bool = False  # 4-bit quantization
    # Memory options
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    # Training mode
    training_mode: str = "pointwise"  # "pointwise", "pairwise", "listwise"


def log_memory_stats(step: str, device: str = "cuda"):
    """Log GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        logger.info(f"[Memory@{step}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return {"allocated_gb": allocated, "reserved_gb": reserved}
    return {}


def create_model_with_peft(
    model_name: str,
    cfg: TrainingConfigV2,
    device: str,
) -> tuple:
    """Create model with optional LoRA/QLoRA configuration."""

    # Quantization config for QLoRA
    quantization_config = None
    if cfg.use_qlora and cfg.use_lora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Determine dtype
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if cfg.use_qlora else None,
    )

    # Enable gradient checkpointing
    if cfg.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Apply LoRA if requested
    if cfg.use_lora and PEFT_AVAILABLE:
        if cfg.use_qlora:
            model = prepare_model_for_kbit_training(model)

        # Target modules for Qwen3-based models
        target_modules = cfg.lora_target_modules
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )

        model = get_peft_model(model, lora_config)

        # Log trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    if not cfg.use_qlora:
        model.to(device)

    return model


def run_training_hpo_v2(trial: optuna.Trial, cfg: Dict) -> float:
    """Run one training trial with enhanced config."""
    seed = cfg["split"]["seed"] + trial.number
    set_seed(seed)

    train_cfg = _sample_training_params_v2(trial, cfg)
    train_cfg.seed = seed

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["criteria"]))

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )

    # Build examples
    train_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["train"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=seed,
    )
    val_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["val"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=seed,
    )

    train_dataset = GroupedRerankerDataset(train_examples)
    val_dataset = GroupedRerankerDataset(val_examples)

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model with PEFT
    model = create_model_with_peft(train_cfg.model_name, train_cfg, device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    log_memory_stats("after_model_load", device)

    # Optimizer - use 8-bit if available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        logger.info("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

    # Collate function based on training mode
    if train_cfg.training_mode == "pointwise":
        collate_fn = lambda b: _collate_pointwise(b, tokenizer, train_cfg.max_length)
    else:
        collate_fn = lambda b: _collate_batch(b, tokenizer, train_cfg.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Training loop with gradient accumulation
    best_val = float("inf")
    accum_steps = train_cfg.gradient_accumulation_steps

    for epoch in range(train_cfg.num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            loss = _compute_loss_v2(batch, model, train_cfg, device)
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Log memory periodically
            if step == 0 and epoch == 0:
                log_memory_stats(f"train_step_{step}", device)

        # Validation
        val_loss = _evaluate_v2(val_loader, model, train_cfg, device)
        best_val = min(best_val, val_loss)

        trial.report(-val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        logger.info(f"Epoch {epoch+1}/{train_cfg.num_epochs}: val_loss={val_loss:.4f}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return -best_val


def _compute_loss_v2(batch: Dict, model, cfg: TrainingConfigV2, device: str) -> torch.Tensor:
    """Compute loss with support for different training modes."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

    if cfg.training_mode == "pointwise":
        # Simple BCE loss for pointwise training
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    else:
        # Grouped hybrid loss for listwise/pairwise
        group_sizes = batch["group_sizes"].tolist()
        loss = hybrid_loss(
            logits=logits,
            labels=labels,
            group_sizes=group_sizes,
            w_list=cfg.w_list,
            w_pair=cfg.w_pair,
            w_point=cfg.w_point,
            temperature=cfg.temperature,
            max_pairs_per_group=cfg.max_pairs_per_group,
        )

    return loss


def _evaluate_v2(loader: DataLoader, model, cfg: TrainingConfigV2, device: str) -> float:
    """Evaluate with inference mode."""
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for batch in loader:
            loss = _compute_loss_v2(batch, model, cfg, device)
            total += loss.item()
    return total / max(len(loader), 1)


def _collate_pointwise(batch: List[Dict], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """Collate for pointwise training - one (query, sentence) pair at a time."""
    pairs: List[List[str]] = []
    labels: List[float] = []

    for ex in batch:
        query = ex["query"]
        sentences = ex["sentences"]
        ex_labels = ex["labels"]
        for sent, label in zip(sentences, ex_labels):
            pairs.append([query, sent])
            labels.append(float(label))

    enc = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float),
    }


def _collate_batch(batch: List[Dict], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """Collate for listwise/pairwise training with group tracking."""
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

    enc = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float),
        "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
    }


def _sample_training_params_v2(trial: optuna.Trial, cfg: Dict) -> TrainingConfigV2:
    """Sample training parameters with LoRA options."""
    space = cfg.get("search_space", {})
    peft_cfg = cfg.get("peft", {})
    memory_cfg = cfg.get("memory", {})

    # Basic params
    max_length = trial.suggest_categorical("max_length", space.get("max_length", [512]))
    batch_size = trial.suggest_categorical("batch_size", space.get("batch_size", [1]))
    num_epochs_range = space.get("num_epochs", [1, 3])
    num_epochs = trial.suggest_int("num_epochs", low=num_epochs_range[0], high=num_epochs_range[1])
    learning_rate = trial.suggest_float("learning_rate", *space.get("learning_rate", [1e-5, 1e-4]), log=True)
    weight_decay = trial.suggest_float("weight_decay", *space.get("weight_decay", [1e-5, 1e-2]), log=True)

    # Loss weights
    w_list = trial.suggest_float("w_list", 0.0, 1.0)
    w_pair = trial.suggest_float("w_pair", 0.0, 1.0)
    w_point = trial.suggest_float("w_point", 0.0, 1.0)
    temperature = trial.suggest_float("temperature", 0.5, 2.0)

    # LoRA params (if enabled)
    use_lora = peft_cfg.get("enable", False)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    if use_lora and "r" in peft_cfg:
        lora_r = trial.suggest_categorical("lora_r", peft_cfg["r"])
    if use_lora and "alpha" in peft_cfg:
        lora_alpha = trial.suggest_categorical("lora_alpha", peft_cfg["alpha"])
    if use_lora and "dropout" in peft_cfg:
        lora_dropout = trial.suggest_categorical("lora_dropout", peft_cfg["dropout"])

    # Training mode
    training_mode = trial.suggest_categorical("training_mode", space.get("training_mode", ["pointwise"]))

    return TrainingConfigV2(
        model_name=cfg["models"]["jina_v3"],
        max_length=max_length,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        w_list=w_list,
        w_pair=w_pair,
        w_point=w_point,
        temperature=temperature,
        max_pairs_per_group=space.get("max_pairs_per_group", 16),
        seed=cfg["split"]["seed"],
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=peft_cfg.get("target_modules"),
        use_qlora=peft_cfg.get("method") == "qlora",
        use_bf16=memory_cfg.get("use_bf16", True),
        use_gradient_checkpointing=memory_cfg.get("gradient_checkpointing", True),
        gradient_accumulation_steps=memory_cfg.get("gradient_accumulation_steps", 1),
        training_mode=training_mode,
    )
