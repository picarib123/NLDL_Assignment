import torch
from tqdm.auto import tqdm
import time
from torch.cuda.amp import GradScaler
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
import os
import math
from transformers.models.llama.modeling_llama import LlamaRMSNorm

class collect_info_reg_llama(nn.Module):
    def __init__(self, model, p=None, lam=4.0):
        super(collect_info_reg_llama, self).__init__()
        self.sum_ori_params = 0 
        self.p = p  
        self.lam = lam  
        self.in_dim_list = [] 
        self.out_dim_list = []  
        self.num_w_list = []  
        self.structures = []  
        self.gate_type = []  
        
        modules = list(model.modules())  
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_block_basic_operation':
                self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('mlp_block')
            if type(m).__name__ == 'virtual_mlp_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                self.structures.append(m.dim)
                self.gate_type.append('mlp')
            if type(m).__name__ == 'virtual_block_attn_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append((m.ex_dict['dim_2'], m.ex_dict['dim_3']))
                self.num_w_list.append(m.ex_dict['num_weight'])
                self.structures.append(m.dim)
                self.head_dim = m.head_dim
                self.num_heads = m.dim
                self.gate_type.append('attn_block')
            if type(m).__name__ == 'virtual_basic_operation':
                self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('basic_gate')

        print("Number of original parameters: %.3f" % (self.sum_ori_params / 10 ** 6))
            
    def forward(self, vectors):
        block_mlp_dim = None
        sum_params = 0
        i = 0
        while i < len(self.structures):
            # Process attention blocks
            if self.gate_type[i] == 'attn_block':
                attn_in_dim = vectors[i].sum()
                attn_out_dim = vectors[i+1].sum()
                current_params = attn_in_dim * 1 * self.out_dim_list[i][0] + attn_in_dim * 2* self.out_dim_list[i][1]+ attn_out_dim * self.out_dim_list[i][0]
                i += 2
                sum_params += current_params

            # Process MLP blocks
            if self.gate_type[i] == 'mlp_block':
                block_mlp_in_dim = vectors[i].sum()
                block_mlp_middle_dim = vectors[i+1].sum()
                block_mlp_out_dim = vectors[i+2].sum()
                current_params = block_mlp_in_dim * block_mlp_middle_dim * 2 + block_mlp_middle_dim * block_mlp_out_dim
                i += 3
                sum_params += current_params

        # Calculate parameter ratio
        param_ratio = sum_params / self.sum_ori_params
        print('ratio', param_ratio)
        if param_ratio > self.p:
            clamped_p_ratio = torch.clamp(param_ratio, min=self.p)
            loss = torch.log(clamped_p_ratio / self.p)
        else:
            clamped_p_ratio = torch.clamp(param_ratio, max=self.p)
            loss = torch.log(self.p / clamped_p_ratio)

        return self.lam * loss

class help_functions_hn(nn.Module):
    def __init__(self, structures, constrained=None):
        self.structures = structures
        self.constrained = constrained

    # Print the structures and summed values of gate vectors
    def print_info(self, vectors):
        print(self.structures)
        config = []
        for i in range(len(vectors)):
            config.append(vectors[i].sum().item())
        print(config)

    def get_mbe_loss(self, model):
        modules = list(model.modules())
        neg_entropy_loss = 0
        count_loss = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_cache':
                neg_entropy_loss += m.neg_entropy # compute_kl_loss(nq_out, q_out)
                # neg_entropy_loss += compute_mbe_activation(m.cache)
                m.neg_entropy = None
                count_loss += 1 
        return 1/ count_loss * neg_entropy_loss 
    
    # Set gate vectors for different modules in the model
    def set_gate_vectors(self, model, vectors):
        if self.constrained == 'structural':
            modules = list(model.modules())
            ind = 0
            model_dim = vectors[0]
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'virtual_basic_operation':
                    m.set_vector_value(model_dim)
                if type(m).__name__ == 'virtual_att_operation':
                    m.set_vector_value(vectors[ind+1])
                    ind += 1
                if type(m).__name__ == 'virtual_mlp_operation':
                    m.set_vector_value(vectors[ind+1])
                    ind += 1
        elif self.constrained == 'same':
            modules = list(model.modules())
            ind = 0
            model_dim = vectors[0]
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'virtual_basic_operation':
                    m.set_vector_value(model_dim)
                if type(m).__name__ == 'virtual_block_basic_operation':
                    m.set_vector_value(model_dim)
                if type(m).__name__ == 'virtual_mlp_operation':
                    m.set_vector_value(vectors[ind+1])
                    ind += 1
                if type(m).__name__ == 'virtual_block_attn_operation':
                    m.set_vector_value(model_dim)
        else:
            modules = list(model.modules())
            ind = 0
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'virtual_basic_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == 'virtual_block_basic_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == 'virtual_mlp_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == 'virtual_block_attn_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1

    def set_gate_status(self, model, use_gate=False):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if hasattr(m, 'use_gate'):
                m.use_gate = use_gate




def compute_mbe_activation(hidden_states: torch.Tensor, alpha: float = 2.0, eps: float = 1e-9) -> float:
    """
    Paper's MBE: Rényi entropy on Gram matrix of token representations.
    
    Args:
        hidden_states: (seq_len, hidden_dim) activation tensor from one layer
        alpha: Rényi order (paper uses alpha=2 as default)
    """
    hidden_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(hidden_shape[0] * hidden_shape[1], hidden_shape[2])
    R = hidden_states.float()           # (s, d)
    K = R @ R.T                         # Gram matrix (s, s) — cheap when s << d
    # print(hidden_states.shape)
    # Eigenvalues of Gram matrix = squared singular values of R
    eigvals = torch.linalg.eigvalsh(K)  # symmetric, so eigvalsh is stable
    eigvals = eigvals.clamp(min=0)      # numerical safety

    prob = eigvals / (eigvals.sum() + eps)

    if alpha == 1:
        return -(prob * torch.log(prob + eps)).sum().item()
    elif alpha == float('inf'):
        return -torch.log(prob.max() + eps).item()
    else:
        return ((1 / (1 - alpha)) * torch.log((prob ** alpha).sum() + eps)).item()