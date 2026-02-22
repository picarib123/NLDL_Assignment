DEBUG = False

if DEBUG:
    from torchviz import make_dot

import os
import time
import datetime
from functools import partial
from typing import Union

import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer
import torch.nn.functional as F
from models.modeling_llama_pruning import LoRAConfig
# Set the number of threads for intra-op parallelism
num_threads = 4  # Choose the desired number of threads
torch.set_num_threads(num_threads)
# Optional: also set the number of inter-op threads (for running separate operations in parallel)
torch.set_num_interop_threads(num_threads) 
# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
#     MixedPrecision,
#     FullStateDictConfig,
#     StateDictType,
# )

# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from datasets import IterableDataset
import math
import tqdm
import warnings
from datasets import load_dataset
warnings.filterwarnings("ignore")
import gc
# Custom modules and tools
from utils import DistributedEnv
from data import dataloader_creator, load_hf_dataset_wikitext, get_fineweb_edu
from models import LlamaTokenizer, PruneLlamaForCausalLM, PruneLlamaDecoderLayer
from pruning import hypernetwork, collect_info_reg_llama, help_functions_hn


def round_to_block_size(current_rank, block_size=32):
    """Round `current_rank` down to the nearest multiple of `block_size`."""
    round_rank = max(block_size, (current_rank // block_size) * block_size)
    return round_rank


def cleanup_cuda():
    """
    Full CUDA memory cleanup between training loops.
    Call this at the end of each training run before starting the next.
    """
    gc.collect()                        # free Python objects referencing tensors
    torch.cuda.empty_cache()            # release PyTorch's cached allocator memory
    torch.cuda.ipc_collect()            # clean up inter-process CUDA handles (multi-GPU)
    torch.cuda.synchronize()            # wait for all CUDA ops to finish before proceeding

def load_eval_data(dataset_name: str) -> str:
    # this mimics gptq datautils
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testdata = "\n\n".join(testdata["sentence"])
    elif dataset_name == 'pile':
        testdata = load_hf_dataset_pile_dedup('validation')
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "c4":
        testdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        testdata = " ".join(testdata[:1100]["text"])

    else:
        raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")
    return testdata

def evaluate(model, tokenizer, datasets="wiki", block_size=4096):
    model.eval()
    # model.cuda()
    datasets = datasets

    total_toks = 0
    results = {}
    for dsname in datasets.split(","):
        test_string = load_eval_data(dsname)
        encoded_text = tokenizer.encode(test_string, return_tensors='pt')
        encoded_text = encoded_text[:, : 256 *  2048]

        nlls = 0
        toks = 0
        # with torch.no_grad(): # torch.inference_mode():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            block_size = block_size 
            for i in tqdm.tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i : i + block_size]
                model_output = model(inp.cuda())
                if hasattr(model_output, 'logits'):
                    logits = model_output.logits
                else:
                    logits = model_output
                nll = torch.nn.functional.cross_entropy(
                    logits[0,:-1], inp[0, 1:].to(dtype=torch.long).cuda(), reduction="sum")
                toks += inp.size(1) - 1
                nlls += nll.item()

        # print(encoded_text.shape, logits.shape)

        encoded_text = encoded_text[:, : logits.shape[0]]
        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")
        total_toks += toks
        results[dsname] = ppl
    
    model.train()
    return results 


def main(
    exp_name: str = 'displlm',
    out_dir: str = None,
    hf_model: str = 'meta-llama/Llama-2-7b-hf',
    learning_rate: float = None,
    total_n_step: int = 100000,
    start_iter: int = 0,
    batch_size: int = 1,
    use_fsdp: bool = True,
    num_workers: int = 0,
    rand_seed: int = None,
    non_hf_tokenizer_path: str = None,
    compile_flag: bool = True,
    p: float = 0.48,
    lam: float = 16.0,
    hn_block_size: int = 4096,
    hn_lr: float = 1e-3,
    min_hn_lr: float = 1e-3,
    use_sch: bool = False,
    use_bf16: bool = False,
):

    # Initialize the distributed environment
    # env = DistributedEnv()
    # print(env)

    # dist.init_process_group(
    #     backend="nccl",
    #     rank=env.global_rank,
    #     world_size=env.world_size,
    #     timeout=datetime.timedelta(seconds=3600 * 5),
    # )

    # Use bf16 if supported, otherwise fallback to fp16
    data_type =torch.bfloat16 #  torch.bfloat16 if torch.cuda.is_bf16_supported() else 

    # Prepare output directory, random seed, and learning rate
    if out_dir is None:
        user_name = 'zhengao'
        dateTimeObj = datetime.datetime.now()
        out_dir = os.path.join('/output/', user_name, exp_name)

    if rand_seed is None:
        rand_seed = start_iter

    # Automatically calculate learning rate if not provided
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4 * 1024 * 1024)
        learning_rate = min(
            llama_learning_rate_per_sample * batch_size * 4096 , # * env.world_size
            0.0003
        )

    os.makedirs(out_dir, exist_ok=True)

    # Set the current GPU
    device_id = 'cuda' #env.local_rank
   # torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Load the tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    if non_hf_tokenizer_path:
       # env.print_master('Using non_hf_tokenizer ...')
        tokenizer = LlamaTokenizer(non_hf_tokenizer_path, output_type='list')
    ignored_token = tokenizer.bos_token_id

    # Load the prunable LLaMA model and collect pruning information
    lora_config = LoRAConfig(
        lora_r=0, lora_alpha=0, lora_dropout=0
    )
    model = PruneLlamaForCausalLM.from_pretrained(
        hf_model, lora_config=lora_config, device_map="cuda"
    )
    model.config.use_cache = False
  #  env.print_master(model.config)
    print(model)
    
    # Load dataset
    tic = time.time()
    result_dataset = load_hf_dataset_wikitext('train', 1 * num_workers)
    # result_dataset  = get_fineweb_edu(train=True)

    train_dataloader_hn = dataloader_creator(
        dataset=result_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        block_size=hn_block_size,
        num_workers=num_workers,
        cycling=True,
        rank=0,
        world_size=1,
        # num_tokens=4096*2,
        ignored_token=ignored_token,    
    )
    toc = time.time() - tic
    print(f"Initializing training dataset - done. Time: {toc:.2f}s")

    # Use collect_info_reg_llama to compute pruning regularization
    param_reg = collect_info_reg_llama(model, p=p, lam=lam)

    # Create hypernetwork for pruning
    hn = hypernetwork(t_structures=param_reg.structures)
    hn_helper = help_functions_hn(param_reg.structures)
    print(hn)
    # Move the model and hypernetwork to the appropriate device
    hn.to(device_id)
   # hn = DDP(hn)  # Wrap hypernetwork with DDP
    hn = torch.compile(hn)
    model.to(device_id)


    # Wrap with FSDP if enabled
    # if use_fsdp:
    #     my_auto_wrap_policy = partial(
    #         transformer_auto_wrap_policy,
    #         transformer_layer_cls={PruneLlamaDecoderLayer}
    #     )
    #     if use_bf16:
    #         model = model.to(data_type).to(device_id)
    #         model = FSDP(
    #             model, 
    #             auto_wrap_policy=my_auto_wrap_policy,
    #             use_orig_params=True
    #             )
    #     else:
    #         model = FSDP(
    #             model,
    #             auto_wrap_policy=my_auto_wrap_policy,
    #             use_orig_params=True,
    #             mixed_precision=MixedPrecision(
    #                 param_dtype=data_type,
    #                 reduce_dtype=data_type,
    #                 buffer_dtype=data_type
    #             ),
    #         )
    # else:
    if use_bf16:
        model = model.to(data_type).to(device_id)
        # model = DDP(model)

    # Enable torch.compile
    # if compile_flag:
    model = torch.compile(model)

    # Train hypernetwork
    train_hn(
        # env,
        model,
        hn=hn,
        train_hn_data=train_dataloader_hn,
        hn_helper=hn_helper,
        param_reg=param_reg,
        ignored_token=ignored_token,
        max_iter=total_n_step,
        out_dir=out_dir,
        p=p,
        hn_block_size=hn_block_size,
        hn_lr=hn_lr,
        min_hn_lr=min_hn_lr,
        use_sch=use_sch,
        use_fsdp=use_fsdp,
        tokenizer=tokenizer
    )
    
def train_hn(
    # env: DistributedEnv,
    model: torch.nn.Module,
    hn: torch.nn.Module,
    train_hn_data: IterableDataset,
    hn_helper,
    param_reg,
    start_iter=0,
    ignored_token=-1,
    log_interval=50,
    max_iter=100000,
    out_dir=None,
    p=None,
    hn_block_size=2048,
    hn_lr=1e-3,
    min_hn_lr=1e-3,
    use_sch=False,
    use_fsdp=False,
    tokenizer=None
):
    data_type = torch.bfloat16 # torch.bfloat16 if torch.cuda.is_bf16_supported() else 
    # device_id = env.local_rank
    iter_num = start_iter

    # Select the appropriate GradScaler (ShardedGradScaler for FSDP)
    # if use_fsdp:
    #     from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    #     scaler = ShardedGradScaler()
    # else:
    scaler = GradScaler()

    # Enable learning rate scheduling if configured
    optimizer = torch.optim.AdamW(hn.parameters(), lr=hn_lr, weight_decay=0.05)
    if use_sch:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iter,
            eta_min=min_hn_lr,
            last_epoch=iter_num - 1
        )

    tic = time.time()
    torch.cuda.empty_cache()

    # Freeze parameters of the main model
    for param in model.parameters():
        param.requires_grad = False
    # Enable training only for the hypernetwork
    for param in hn.parameters():
        param.requires_grad = True
    hn.train()
    model.train()
    # print("Current Memory before:", torch.cuda.memory_allocated())
    while True:
        for batch in train_hn_data:
            if iter_num >= max_iter:
                break

            with torch.no_grad():
                input_ids = batch['input_ids'].to('cuda')
                targets = batch['labels'].to('cuda')
                input_ids = input_ids[:, :hn_block_size]
                targets = targets[:, :hn_block_size]

            # print("Current Memory A:", torch.cuda.memory_allocated())
            with autocast(device_type='cuda', dtype=data_type):
            # if True:
                # Generate pruning vectors using the hypernetwork
                vectors = hn()
                hn_helper.set_gate_vectors(model, vectors) 

                # Forward pass
                output = model(input_ids, use_teacher=False)
                logits = output.logits if hasattr(output, 'logits') else output
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=ignored_token
                )

                # Calculate pruning regularization
                if hasattr(hn, "module"):
                    hard_out = hn.module.hard_output()
                else:
                    hard_out = hn.hard_output()
                    
                reg_loss = param_reg(hard_out)
                # _ = model(input_ids, use_teacher=True) # , output_attentions=True
                # final_kl_loss = compute_kl_loss(out_teacher.logits.detach(), output.logits)
                #  
                # mbe_loss = hn_helper.get_mbe_loss(model)
                total_loss = ce_loss + reg_loss #+ 2 * mbe_loss  #+ kl_loss * 2e-3 # 

            # Check for NaN loss values
            if torch.isnan(total_loss):
                print("!!! nan loss detected !!!")
                total_loss.fill_(0)
# 
            print("ce loss: %.4f" % ce_loss, "reg_loss: %.4f" % reg_loss)
            # print("kl loss: %.4f" % final_kl_loss, "reg_loss: %.4f" % reg_loss)
            # Backward pass and optimizer step
            total_loss.backward()
            optimizer.step()
            # scaler.scale(total_loss).backward()
            # scaler.unscale_(optimizer)

            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()
            cleanup_cuda()

            if iter_num % 100 == 0:
                
                with torch.no_grad():
                    hn_helper.set_gate_vectors(model, hard_out)  
                    evaluate(model, tokenizer, datasets='wikitext',block_size=4096) # ,ptb
                elapsed = time.time() - tic
                print(
                    f"Iter {iter_num}/{max_iter}, "
                    f"loss={total_loss.item():.4f}, "
                    f"reg={reg_loss.item():.4f}, "
                    f"time={elapsed*1000:.2f}ms"
                )
                tic = time.time()
                
                torch.save(hn.state_dict(), os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}-llama2.pt"))
            iter_num += 1
            if iter_num >= max_iter:
                break
        if iter_num >= max_iter:
            break

    # Save the hypernetwork
    # if env.world_size == 1:
    #     if env.global_rank == 0:
    torch.save(hn.state_dict(), os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}-llama2.pt"))
    # else:
    #     # For multi-GPU, save only on rank=0
    #     if hasattr(hn, "module"):
    #         state_dict_hn = hn.module.state_dict()
    #     else:
    #         state_dict_hn = hn.state_dict()

    #     if env.global_rank == 0:
    #         torch.save(state_dict_hn, os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt"))

    
def compute_kl_loss(student_logits, teacher_logits, T=1.0):
    """
    Compute KL divergence loss for distillation.

    student_logits: [B, ..., C]
    teacher_logits: [B, ..., C]
    """
    # soft targets from teacher
    pred = F.softmax(teacher_logits / T, dim=-1)

    # log probabilities from student
    log_pred = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence (PyTorch does sum over last dim by default)
    # kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # return kl * (T * T)   
    kl = (pred * (pred.add(1e-8).log() - log_pred)).sum(dim=-1)  # sum over classes
    return kl.mean() * (T * T)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)