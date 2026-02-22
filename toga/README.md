Pruning Techniques
---

####  Project Overview 

This folder is based on DISP-LLM github repository https://github.com/ZhengaoLi/DISP-LLM-Dimension-Independent-Structural-Pruning

---

####  Requirements

The code is extensively tested with Pytorch 2.0.1 and transformers 4.44.2, and a complete environment is provided in the environment.yml file.



#### How to Run


1. For standard pruning (DISP-LLM):
```
python train_hypernetwork.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --out_dir path/to/your/out_dir -p 0.5
python prune_model.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --out_dir /path/to/pruned_model
```

2. For CMWP:
```
python train_hypernetwork_unearn_v3.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --out_dir path/to/your/out_dir -p 0.5
python prune_model.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --out_dir /path/to/pruned_model
```

2. For our proposed method:
```
python train_hypernetwork_unearn_v2.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --out_dir path/to/your/out_dir -p 0.5
python prune_model.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --merged_model path/to/your/lora_finetuned/model  --out_dir /path/to/pruned_model
```

#### Citation
```
@inproceedings{gaodisp,
  title={DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models},
  author={Gao, Shangqian and Lin, Chi-Heng and Hua, Ting and Tang, Zheng and Shen, Yilin and Jin, Hongxia and Hsu, Yen-Chang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
