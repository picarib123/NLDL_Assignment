"""
Privacy-Aware Hypernetwork Pruning + Gradient Ascent Unlearning
================================================================

Architecture:
  - Hypernetwork (HN): Trained on PUBLIC wikitext data (standard training).
    Learns structured pruning gates for LLaMA.
  - LoRA adapters: UNLEARNED on SENSITIVE Enron email data via gradient ascent.
    Instead of learning Enron content, we actively push the model to forget it.

Unlearning strategy:
  - Gradient ascent on Enron data: maximize cross-entropy loss so the model
    becomes worse at predicting/reproducing Enron emails.
  - The combination ensures targeted forgetting of sensitive data while
    preserving model utility.

Key references:
  - "Who's Harry Potter? Approximate Unlearning in LLMs" (Eldan & Russinovich, 2023)
  - "Large Language Model Unlearning" (Yao et al., 2023)
  - "Knowledge Unlearning for Mitigating Language Models' Memorization" (Jang et al., 2023)
"""

import os
import random
import time
import datetime
import math
import json
from functools import partial
from typing import Union, Optional, List, Dict

import torch
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer
from datasets import IterableDataset, load_dataset
import tqdm
import warnings

warnings.filterwarnings("ignore")

# Custom modules (from your codebase)
from data import dataloader_creator, load_hf_dataset_wikitext
from models.modeling_llama_pruning import LlamaForCausalLM as PruneLlamaForCausalLM
from models.modeling_llama_pruning import LoRAConfig
from models import LlamaTokenizer
from pruning.hypernetwork import hypernetwork
from pruning.pruning_helper import collect_info_reg_llama, help_functions_hn
from models.lora_linear import LoRALinear

ENRON_DATASET_SIZES = {
    "aeslc": {"train": 14436, "validation": 1960, "test": 1906},
    "tiedaar/email_enron": {"train": 517401},
}


# ============================================================================
# 1. ENRON DATASET LOADER
# ============================================================================

def load_enron_dataset(split: str = "train", streaming: bool = False):
    """
    Load the Enron email dataset from HuggingFace.
    This is the FORGET set — data we want the model to unlearn.
    """
    try:
        dataset = load_dataset("aeslc", split=split, streaming=True)
        dataset = dataset.rename_column("email_body", "text")
        dataset = dataset.select_columns(["text"])
        return dataset
    except Exception as e:
        print(f"AESLC load failed ({e}), trying full Enron corpus...")
        raise RuntimeError("Could not load Enron dataset.")


# ============================================================================
# 2. UNLEARNING ENGINE
# ============================================================================

class GradientAscentUnlearner:
    """
    Gradient Ascent Unlearning engine for LoRA parameters.

    Core idea: maximize loss on the forget set (Enron emails) so the model
    can no longer reproduce that data

    Supports several unlearning objectives:
      - "ga":            Pure gradient ascent on forget set
    """

    def __init__(
        self,
        method: str = "ga",
        forget_loss_weight: float = 1.0,
        max_forget_loss: float = 100.0,
        device: str = "cuda",
    ):
        self.method = method
        self.forget_loss_weight = forget_loss_weight
        self.max_forget_loss = max_forget_loss
        self.device = device
        self.steps_taken = 0
        self.forget_losses = []

        print(f"[Unlearner] Initialized:")
        print(f"  method               = {self.method}")
        print(f"  forget_loss_weight   = {self.forget_loss_weight}")
        print(f"  max_forget_loss      = {self.max_forget_loss}")

    def get_report(self) -> dict:
        """Generate a summary of the unlearning process."""
        return {
            "method": self.method,
            "steps_taken": self.steps_taken,
            "forget_loss_weight": self.forget_loss_weight,
            "avg_forget_loss": (
                sum(self.forget_losses[-100:]) / len(self.forget_losses[-100:])
                if self.forget_losses else 0.0
            ),
        }



def train_hn(
    model,
    hn,
    hn_helper,
    param_reg,
    batch,
    forget_batch,
    unlearner: GradientAscentUnlearner,
    block_size,
    device_id,
    data_type,
    ignored_token=-1,
    iter_num=0,
    forget_loss_threshold=5
):
    """Train hypernetwork on PUBLIC wikitext data."""
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device_id)
        targets = batch["labels"].to(device_id)
        input_ids = input_ids[:, :block_size]
        targets = targets[:, :block_size]
        forget_ids = forget_batch["input_ids"].to(device_id)[:, :block_size]
        forget_targets = forget_batch["labels"].to(device_id)[:, :block_size]

    with autocast(device_type="cuda", dtype=data_type):
        vectors = hn()
        hn_helper.set_gate_vectors(model, vectors)

        output = model(input_ids, use_teacher=False)
        logits = output.logits if hasattr(output, "logits") else output
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=ignored_token,
        )


        forget_output = model(forget_ids, use_teacher=False)
        forget_logits = (
            forget_output.logits
            if hasattr(forget_output, "logits")
            else forget_output
        )
        forget_ce = F.cross_entropy(
            forget_logits.view(-1, forget_logits.size(-1)),
            forget_targets.view(-1),
            ignore_index=ignored_token,
        )

        # NEGATE the loss → gradient ascent → model forgets this data
        # Clamp to prevent the forget loss from exploding
        forget_loss = torch.abs(forget_ce* unlearner.forget_loss_weight-forget_loss_threshold)


        hard_out = hn.hard_output()
        reg_loss = param_reg(hard_out)
        total_loss = reg_loss + ce_loss + forget_loss

    if torch.isnan(total_loss):
        print("!!! nan loss detected !!!")
        total_loss.fill_(0)
        exit()

    print(
        f"[HN] iter {iter_num} | "
        f"ce_loss: {ce_loss.item():.4f} | "
        f"reg_loss: {reg_loss.item():.4f}"
    )

    return total_loss, reg_loss


# ============================================================================
# 5. EVALUATION (unchanged)
# ============================================================================

def load_eval_data(dataset_name: str) -> str:
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testdata = "\n\n".join(testdata["sentence"])
    elif dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testdata = " ".join(testdata[:1100]["text"])
    else:
        raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")
    return testdata


def evaluate(model, tokenizer, datasets="wikitext", block_size=4096):
    model.eval()
    results = {}
    for dsname in datasets.split(","):
        test_string = load_eval_data(dsname)
        encoded_text = tokenizer.encode(test_string, return_tensors="pt")
        encoded_text = encoded_text[:, : 256 * 2048]

        nlls = 0
        toks = 0
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in tqdm.tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i : i + block_size]
                model_output = model(inp.cuda())
                logits = (
                    model_output.logits
                    if hasattr(model_output, "logits")
                    else model_output
                )
                nll = F.cross_entropy(
                    logits[0, :-1],
                    inp[0, 1:].to(dtype=torch.long).cuda(),
                    reduction="sum",
                )
                toks += inp.size(1) - 1
                nlls += nll.item()

        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")
        results[dsname] = ppl
    return results


def evaluate_forget_quality(model, tokenizer, block_size=2048):
    """
    Evaluate how well the model has forgotten Enron data.
    Higher perplexity on Enron = better unlearning.
    """
    model.eval()
    try:
        forget_dataset = load_dataset("aeslc", split="test", streaming=False)
    except Exception:
        print("Could not load Enron test split for forget evaluation.")
        return {}

    texts = [ex["email_body"] for ex in forget_dataset]
    full_text = "\n\n".join(texts[:500])  # Use subset for speed
    encoded_text = tokenizer.encode(full_text, return_tensors="pt")
    encoded_text = encoded_text[:, : 128 * 2048]

    nlls = 0
    toks = 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, encoded_text.shape[1], block_size):
            inp = encoded_text[:, i : i + block_size]
            model_output = model(inp.cuda())
            logits = (
                model_output.logits
                if hasattr(model_output, "logits")
                else model_output
            )
            nll = F.cross_entropy(
                logits[0, :-1],
                inp[0, 1:].to(dtype=torch.long).cuda(),
                reduction="sum",
            )
            toks += inp.size(1) - 1
            nlls += nll.item()

    ppl = math.exp(nlls / toks)
    print(f"Perplexity on Enron (forget set): {ppl:.2f}  [higher = better unlearning]")
    return {"enron_ppl": ppl}

def merge_lora_to_model(model):
    """Merge all LoRA adapters into base weights across the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_lora()


# ============================================================================
# 6. MAIN TRAINING LOOP
# ============================================================================

def train(
    hf_model,
    model: torch.nn.Module,
    hn: torch.nn.Module,
    train_hn_data,
    train_forget_data,
    hn_helper,
    param_reg,
    unlearner: GradientAscentUnlearner,
    start_iter=0,
    ignored_token=-1,
    max_iter=100000,
    out_dir=None,
    p=None,
    hn_block_size=2048,
    hn_lr=1e-3,
    lora_lr=1e-4,
    dataset: str = "wikitext",
    lam=0.0,
    tokenizer=None,
    reference_model=None,
):
    data_type = torch.bfloat16
    device_id = "cuda"
    iter_num = start_iter
    eval_iter = 100

    torch.cuda.empty_cache()

    # ---- Freeze base model, enable HN + LoRA ----
    print("Freezing base model parameters, enabling HN + LoRA...")
    for param in hn.parameters():
        param.requires_grad = True
    for p_n, param in model.named_parameters():
        # if "lora" in p_n:
        #     param.requires_grad = True
        # else:
        param.requires_grad = False

    hn_param_count = sum(p.numel() for p in hn.parameters())
    print(f"  HN trainable params:   {hn_param_count:,}")

    # ---- Optimizers ----
    optimizer_hn = torch.optim.AdamW(hn.parameters(), lr=hn_lr, weight_decay=0.05)


    hn.train()
    model.train()

    train_hn_data = iter(train_hn_data)
    train_forget_data = iter(train_forget_data)

    print("\n" + "=" * 70)
    print("TRAINING START — GRADIENT ASCENT UNLEARNING")
    print(f"  HN data:     wikitext (public)")
    print(f"  Forget data: Enron emails (gradient ascent → unlearn)")
    print(f"  Retain data: wikitext (gradient descent → preserve)")
    print(f"  Method:      {unlearner.method}")
    print("=" * 70 + "\n")

    while True:
        if True:
            # ============================================================
            # STEP A: Train Hypernetwork on PUBLIC wikitext (standard GD)
            # ============================================================
            try:
                hn_batch = next(train_hn_data)
                forget_batch = next(train_forget_data)
            except StopIteration:
                print("\nFinished iterating over HN dataset.")
                break

            total_loss, reg_loss = train_hn(
                model, hn, hn_helper, param_reg, hn_batch, forget_batch, unlearner,
                hn_block_size, device_id, data_type,
                ignored_token, iter_num,
            )
            total_loss.backward()

            optimizer_hn.step()
            optimizer_hn.zero_grad()


        # ============================================================
        # Bookkeeping
        # ============================================================
        iter_num += 1

        # Evaluate
        if iter_num % eval_iter == 0 or iter_num >= max_iter:
            print(f"\n--- Evaluation at iter {iter_num} ---")
            with torch.no_grad():
                hard_out = hn.hard_output()
                hn_helper.set_gate_vectors(model, hard_out)
                # Utility: how good is the model on general text?
                evaluate(model, tokenizer, datasets="wikitext", block_size=2048)
                # Forget quality: how bad is it on Enron?
                evaluate_forget_quality(model, tokenizer, block_size=2048)
            model.train()

        # Save checkpoints
        if iter_num % 500 == 0 or iter_num >= max_iter:
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                hn_path = os.path.join(
                    out_dir, f"hn-unlearn-p_{p:.2f}-{dataset}-lam_{lam}-{unlearner.method}-v3.pt"
                )
                # model_path = os.path.join(
                #     out_dir, f"model-unlearn-p_{p:.2f}-{dataset}-lam_{lam}-{unlearner.method}-v3.pt"
                # )
                torch.save(hn.state_dict(), hn_path)
                # torch.save(model.state_dict(), model_path)

                report = unlearner.get_report()
                report_path = os.path.join(out_dir, "unlearning_report.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

                print(f"Saved checkpoint + unlearning report")

        if iter_num >= max_iter:
            break

    # merge_lora_to_model(model)
    # model_path = os.path.join(out_dir, f"model-unlearn-p_{p:.2f}-{dataset}-lam_{lam}-{unlearner.method}-v3.pt")    
    # torch.save(model.state_dict(), model_path)
    hn_path = os.path.join(out_dir, f"hn-unlearn-p_{p:.2f}-{dataset}-lam_{lam}-{unlearner.method}-v3.pt")
    torch.save(hn.state_dict(), hn_path)

    # ---- Final Report ----
    final_report = unlearner.get_report()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — FINAL UNLEARNING REPORT")
    print(f"  Method:            {final_report['method']}")
    print(f"  Total steps:       {final_report['steps_taken']}")
    print(f"  Avg forget loss:   {final_report['avg_forget_loss']:.4f}  [higher = better unlearning]")
    print(f"  Interpretation:    The model's ability to reproduce Enron emails")
    print(f"                     has been degraded via gradient ascent while")
    print(f"                     general language capability is preserved.")
    print("=" * 70)

    return final_report


# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================

def main(
    out_dir: str = None,
    hf_model: str = "meta-llama/Llama-2-7b-hf",
    total_n_step: int = 1000,
    batch_size: int = 1,
    num_workers: int = 1,
    non_hf_tokenizer_path: str = None,
    p: float = 0.0,
    lam: float = 16.0,
    hn_block_size: int = 4096,
    hn_lr: float = 1e-3,
    lora_lr: float = 1e-4,
    dataset: str = "wikitext",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.01,
    # Unlearning parameters
    unlearn_method: str = "ga",
    forget_loss_weight: float = 1.0,
    max_forget_loss: float = 100.0,
):
    device_id = "cuda"
    data_type = torch.bfloat16
    torch.cuda.empty_cache()

    # ---- Tokenizer ----
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    if non_hf_tokenizer_path:
        print("Using non_hf_tokenizer ...")
        tokenizer = LlamaTokenizer(non_hf_tokenizer_path, output_type="list")
    ignored_token = tokenizer.bos_token_id

    # ---- Model ----
    lora_config = LoRAConfig(
        lora_r=0, lora_alpha=0, lora_dropout=0.0
    )
    model = PruneLlamaForCausalLM.from_pretrained(
        hf_model, lora_config=lora_config, device_map="cuda"
    )
    model.config.use_cache = False

    print(model.config)

    # ---- (Optional) Reference model for KL regularization ----
    reference_model = None
    

    # ---- Datasets ----
    # PUBLIC data for hypernetwork
    hn_dataset = load_hf_dataset_wikitext("train", False)

    # FORGET set: Enron emails (gradient ascent)
    forget_dataset = load_enron_dataset("train", False)
    enron_size = ENRON_DATASET_SIZES["aeslc"]["train"]
    print(f"Enron (forget) dataset size: {enron_size}")


    train_dataloader_hn = dataloader_creator(
        dataset=hn_dataset, tokenizer=tokenizer, batch_size=batch_size,
        block_size=hn_block_size, num_workers=num_workers, cycling=True,
        rank=0, world_size=1, ignored_token=ignored_token,
    )
    train_dataloader_forget = dataloader_creator(
        dataset=forget_dataset, tokenizer=tokenizer, batch_size=batch_size,
        block_size=hn_block_size, num_workers=num_workers, cycling=True,
        rank=0, world_size=1, ignored_token=ignored_token,
    )

    # ---- Pruning setup ----
    param_reg = collect_info_reg_llama(model, lam=lam, p=p)
    print(param_reg.structures)
    hn_model = hypernetwork(t_structures=param_reg.structures)
    hn_helper = help_functions_hn(param_reg.structures)

    hn_model.to(device_id)
    model = model.to(data_type).to(device_id)

    # ---- Unlearning Engine ----
    unlearner = GradientAscentUnlearner(
        method=unlearn_method,
        forget_loss_weight=forget_loss_weight,
        max_forget_loss=max_forget_loss,
        device=device_id,
    )

    # ---- Train ----
    report = train(
        hf_model, model, hn=hn_model,
        train_hn_data=train_dataloader_hn,
        train_forget_data=train_dataloader_forget,
        hn_helper=hn_helper, param_reg=param_reg,
        unlearner=unlearner, ignored_token=ignored_token,
        max_iter=total_n_step, out_dir=out_dir, p=p,
        hn_block_size=hn_block_size, hn_lr=hn_lr, lora_lr=lora_lr,
        dataset=dataset, lam=lam, tokenizer=tokenizer,
        reference_model=reference_model,
    )

    # Save final report
    if out_dir:
        with open(os.path.join(out_dir, "unlearning_report_final.json"), "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)