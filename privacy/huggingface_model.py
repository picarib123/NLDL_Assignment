import torch
from typing import Optional, Dict, List, Any
import warnings
class HuggingFaceModel:
    """
    Wrapper for HuggingFace models with PII evaluation support
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        model_path:str = "",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        api_token: Optional[str] = None
    ):
        """
        Initialize HuggingFace model
        
        Args:
            model_name: Model ID from HuggingFace Hub
                Examples:
                - "gpt2" (small, fast)
                - "meta-llama/Llama-2-7b-chat-hf" (requires auth)
                - "mistralai/Mistral-7B-Instruct-v0.2"
                - "tiiuae/falcon-7b-instruct"
                - "EleutherAI/pythia-6.9b"
            device: "cuda", "cpu", or "auto"
            load_in_8bit: Use 8-bit quantization (reduces memory)
            load_in_4bit: Use 4-bit quantization (reduces memory more)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling vs greedy
            api_token: HuggingFace API token for gated models
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        if True:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            # Set device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            print(f"Using device: {self.device}")
            
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=api_token,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization options
            print("Loading model...")
            model_kwargs = {
                "token": api_token,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            elif load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.device if self.device == "cuda" else None
            
            if model_path != model_name:
                import sys 
                sys.path.append(model_path)
                from modeling_llama_pruned import LlamaForCausalLM 
                self.model = LlamaForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            
            # Move to device if not using quantization
            if not (load_in_8bit or load_in_4bit) and self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✓ Model loaded successfully")
            print(f"  Parameters: ~{self.model.num_parameters() / 1e9:.2f}B")
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        return_full_text: bool = False
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top_p
            do_sample: Override default sampling
            return_full_text: If True, return prompt + generation
        
        Returns:
            Generated text
        """
        # Use instance defaults if not specified
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        if True:
            # Generate
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                return_full_text=return_full_text,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove prompt if not returning full text
            if not return_full_text and generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
        
        Returns:
            List of generated responses
        """
        responses = []
        
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": self.model.num_parameters(),
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.tokenizer.model_max_length,
        }