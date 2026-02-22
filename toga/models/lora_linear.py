import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 

class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        adapter_name: str = '',
        lora_r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        # use_lora :bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.active_adapter = adapter_name
        self.use_lora = True 
        self.r = lora_r 
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x : x
        if lora_r > 0:
            self.lora_A = nn.Parameter(torch.zeros(lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
            self.scaling = self.lora_alpha / self.r
        else:
            self.use_lora=False
        self.reset_lora_parameters()
        
    def reset_lora_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    
    def forward(self, x, use_teacher=False):
        pretrained = F.linear(x, self.weight, bias=self.bias)
        if self.use_lora and not use_teacher:
            lora = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1) * self.scaling
            # lora = x @ (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
        else:
            lora = 0
        return pretrained + lora

    def merge_lora(self):
        """Merge LoRA weights into the base weight matrix. W' = W + B @ A * scaling"""
        if self.use_lora and hasattr(self, 'lora_A') and self.r > 0:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.use_lora = False

        