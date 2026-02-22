"""
Fixed version: Replaced Chat model with HuggingFace model

Changes:
1. Removed Chat.from_helm dependency
2. Added HuggingFaceModel integration
3. Updated generation logic
4. Simplified batch processing
"""

import json
import os
os.environ["HF_HOME"] = "/cluster/projects/nn9342k/.cache/huggingface/datasets/"
import yaml
from tempfile import TemporaryDirectory

# Import HuggingFace model instead of Chat
from huggingface_model import HuggingFaceModel

# Keep original imports (you'll need to provide these files)
from sys_prompts import PROMPT_TEMPLATE, PII_INFO, PII_PROMPT_TYPE, PRIVACY_TOPIC, JAILBREAK_PROMPTS
from utils import get_privacy_understanding_messages, load_enron_email_prompts, get_pii_messages_k_shot
from result_agg import calculate_scores


class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc, is_raw_input=False):
        self.sys_demo = sys_demo
        self.task_desc = task_desc
        self.is_raw_input = is_raw_input
    
    def get_message(self, input):
        messages = []
        if len(self.sys_demo) > 0:
            messages.append({"role": "system", "content": self.sys_demo})

        if len(self.task_desc) > 0:
            messages.append({"role": "user", "content": self.task_desc})
        
        if self.is_raw_input:
            messages.append({"role": "user", "content": input})   
        else:    
            messages = messages + input
            
        return messages
    
    def format_messages_to_prompt(self, messages):
        """
        Convert messages to a single prompt string for HuggingFace model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        
        # Add final "Assistant:" to prompt completion
        if not prompt.endswith("Assistant:"):
            prompt += "Assistant:"
        
        return prompt


ScenarioName2ResultDir = {
    "pii": "./results/privacy/generations/pii",
    "pii_jailbreak": "./results/privacy/generations/pii_jailbreak",
    "privacy_understanding": "./results/privacy/generations/understanding",
    "enron_email_extraction": "./results/privacy/generations/enron",
}


def generate_and_save(
    max_tokens, 
    t,
    dataset, 
    gt_list, 
    constructor, 
    out_file, 
    batch_size, 
    model,
    keynames=['gt'], 
    ner=False
):
    """
    Generate responses using HuggingFace model and save to file
    
    Args:
        args: Arguments
        dataset: List of input data
        gt_list: List of ground truth lists
        constructor: MessageConstructor instance
        out_file: Output file path
        batch_size: Batch size for processing
        model: HuggingFaceModel instance
        keynames: Keys for ground truth data
        ner: Whether to use NER
    
    Returns:
        Tuple of (price_cost, prompt_tokens, comple_tokens, num_prompt)
    """
    dataset_size = len(dataset)
    num_batches = int(dataset_size / batch_size)
    
    # Token counting (approximate for HuggingFace models)
    prompt_tokens = 0
    comple_tokens = 0
    num_prompt = 0
    price_cost = 0  # Not applicable for local HuggingFace models
    
    if ner == True: 
        try:
            from transformers import pipeline
            ner_classifier = pipeline(
                "token-classification", 
                model="bigcode/starpii", 
                aggregation_strategy="simple"
            )
        except Exception as err:
            print(f"NER initialization error: {err}")
            ner_classifier = None

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch_data = dataset[batch_start:batch_end]
        
        # Process each item in batch
        cache = []
        for item in batch_data:
            # Get messages
            messages = constructor.get_message(item)
            
            # Convert messages to prompt string
            prompt = constructor.format_messages_to_prompt(messages)
            
            # Generate response using HuggingFace model
            try:
                response = model.generate(
                        prompt,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        # temperature=t ,
                        # do_sample=True if t > 0 else False
                )
                
                # Count tokens (approximate)
                prompt_tokens += len(prompt.split())
                comple_tokens += len(response.split())
                num_prompt += 1
                
                # Format cache entry (mimicking original format)
                # [index, continuation, isbanned, prompt, ..., response]
                cache_entry = [
                    len(cache),           # index
                    response,             # continuation
                    False,                # isbanned (not applicable)
                    prompt,               # prompt
                    response              # response (last element)
                ]
                cache.append(cache_entry)
                
            except Exception as e:
                print(f"Generation error: {e}")
                # Add empty entry on error
                cache.append([len(cache), "", False, prompt, ""])
        
        # Save results
        with open(out_file, "a") as f:
            for idx, x in enumerate(cache):
                res = {}
                
                # Add ground truth data
                for i, key in enumerate(keynames):
                    batch_gt = gt_list[i][batch_start:batch_end] 
                    res[key] = batch_gt[idx]
                
                # Add generation results
                res_gen = {
                    "continuation": x[1], 
                    "isbanned": x[2], 
                    "prompt": x[3], 
                    "response": x[-1]
                }
                res.update(res_gen)
                
                # Add NER if requested
                if ner == True and ner_classifier is not None: 
                    try:
                        ner_results = ner_classifier(x[1])
                        ner_words = [] 
                        if len(ner_results) > 0:
                            ner_words = [entity['word'] for entity in ner_results]
                        res.update({"named_entity_rec_continuation": ner_words})
                    except Exception as err:
                        print(f"NER processing error: {err}")
                
                f.write(json.dumps(res) + "\n")
        
        print(f"Processed batch {batch_idx + 1}/{num_batches}")
    
    print(f"Total prompts: {num_prompt}")
    print(f"Approx prompt tokens: {prompt_tokens}")
    print(f"Approx completion tokens: {comple_tokens}")
    
    return price_cost, prompt_tokens, comple_tokens, num_prompt


def main(out_file, dataset_size, model_name, scenario_name, personal_infos, template, 
         prompt_types, data_file, few_shot_num, batch_size, privacy_topics, question_prompt,
         pruned_model=None,
         max_tokens=16, t=1, seed=123):

    all_cost = {}
    all_prompt_tokens = {}
    all_comple_tokens = {}
    all_num_prompt = {}
    
    # Initialize HuggingFace model (once for all scenarios)
    model_path = pruned_model if pruned_model is not None else model_name
    print(f"Loading HuggingFace model: {model_path}")
    model = HuggingFaceModel(
        model_name=model_name,
        model_path=model_path,
        device="auto",
        max_new_tokens=max_tokens,
        temperature=t,
        do_sample=True
    )
    print(f"✓ Model loaded: {model_name}")

    # =========================================================================
    # SCENARIO 1: PII
    # =========================================================================
    if scenario_name == "pii":
        for pii in personal_infos:
            if pii not in PII_INFO:
                print(f"Skip {pii} as it is not defined in supported PII types {PII_INFO}")
                continue
            
            constructor = MessageConstructor(
                PROMPT_TEMPLATE[template]['sys_prompt'].format(pii),
                PROMPT_TEMPLATE[template]['task_desc']
            )

            for fewshot_type in prompt_types:
                if fewshot_type not in PII_PROMPT_TYPE:
                    print(f"Skip {fewshot_type} as it is not defined in supported prompt types {PII_PROMPT_TYPE}")
                    continue

                dataset, gt_list = get_pii_messages_k_shot(
                    filename=data_file, 
                    k=few_shot_num, 
                    pp=pii,
                    fewshot_type=fewshot_type, 
                    num_cases=dataset_size
                )

                result_dir = ScenarioName2ResultDir[scenario_name]
                os.makedirs(result_dir, exist_ok=True)
                out_file = os.path.join(
                    result_dir,
                    "{}_{}shot{}_s{}_{}_{}.jsonl".format(
                        pii, few_shot_num, fewshot_type, dataset_size, 
                        model_path, seed
                    ).replace("/", "_").replace(" ", "")
                )
                print(f"Saving to {out_file}")

                price_cost, prompt_tokens, comple_tokens, num_prompt = generate_and_save(
                    max_tokens, 
                    t,
                    dataset=dataset, 
                    gt_list=[gt_list],
                    constructor=constructor, 
                    out_file=out_file, 
                    batch_size=batch_size,
                    model=model,
                    keynames=['gt']
                )

                all_cost[pii] = price_cost
                all_prompt_tokens[pii] = prompt_tokens
                all_comple_tokens[pii] = comple_tokens
                all_num_prompt[pii] = num_prompt

    # =========================================================================
    # SCENARIO 2: PII with Jailbreak
    # =========================================================================
    elif scenario_name == "pii_jailbreak":
        for pii in personal_infos:
            if pii not in PII_INFO:
                print(f"Skip {pii} as it is not defined in supported PII types {PII_INFO}")
                continue
            
            for jb_idx, jailbreak_prompt in enumerate(JAILBREAK_PROMPTS):
                constructor = MessageConstructor(
                    jailbreak_prompt, 
                    PROMPT_TEMPLATE[template]['task_desc']
                )

                for fewshot_type in prompt_types:
                    if fewshot_type not in PII_PROMPT_TYPE:
                        print(f"Skip {fewshot_type} as it is not defined in supported prompt types {PII_PROMPT_TYPE}")
                        continue

                    dataset, gt_list = get_pii_messages_k_shot(
                        filename=data_file, 
                        k=few_shot_num, 
                        pp=pii,
                        fewshot_type=fewshot_type, 
                        num_cases=dataset_size
                    )

                    result_dir = ScenarioName2ResultDir[scenario_name]
                    os.makedirs(result_dir, exist_ok=True)
                    base_fname = "{}_{}shot{}_s{}_{}_{}_jb{}.jsonl".format(
                        pii, few_shot_num, fewshot_type, dataset_size, 
                        model_path, seed, jb_idx
                    ).replace("/", "_").replace(" ", "")
                    out_file = os.path.join(result_dir, base_fname)
                    
                    print(f"Saving to {out_file}")

                    price_cost, prompt_tokens, comple_tokens, num_prompt = generate_and_save(
                        max_tokens, 
                        t,
                        dataset=dataset, 
                        gt_list=[gt_list],
                        constructor=constructor, 
                        out_file=out_file, 
                        batch_size=batch_size,
                        model=model,
                        keynames=['gt'],
                        ner=True
                    )
                    
                    all_cost[pii] = price_cost
                    all_prompt_tokens[pii] = prompt_tokens
                    all_comple_tokens[pii] = comple_tokens
                    all_num_prompt[pii] = num_prompt

    # =========================================================================
    # SCENARIO 3: Privacy Understanding
    # =========================================================================
    elif scenario_name == "privacy_understanding":
        constructor = MessageConstructor(
            PROMPT_TEMPLATE[template]['sys_prompt'],
            PROMPT_TEMPLATE[template]['task_desc']
        )
        
        for topic in privacy_topics:
            if topic not in PRIVACY_TOPIC:
                print(f"Skip {topic} as it is not defined in supported topics {PRIVACY_TOPIC}")
                continue
            
            dataset, gt_list = get_privacy_understanding_messages(
                topic, 
                question_prompt
            )

            result_dir = ScenarioName2ResultDir[scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                result_dir,
                "{}_t{}_s{}_{}_ques{}_{}.jsonl".format(
                    topic, template, dataset_size, model_path, 
                    question_prompt, seed
                ).replace("/", "_").replace(" ", "")
            )

            # Expand dataset
            expanded_dataset = []
            expanded_gt_list = []
            for i, data in enumerate(dataset):
                expanded_dataset.extend([data] * dataset_size)
                expanded_gt_list.extend([gt_list[i]] * dataset_size)
    
            print(f"Saving to {out_file}")
            price_cost, prompt_tokens, comple_tokens, num_prompt = generate_and_save(
                max_tokens, 
                t,
                dataset=expanded_dataset, 
                gt_list=[expanded_gt_list], 
                constructor=constructor, 
                out_file=out_file, 
                batch_size=dataset_size,
                model=model,
                keynames=['word']
            )
        
            all_cost[topic] = price_cost
            all_prompt_tokens[topic] = prompt_tokens
            all_comple_tokens[topic] = comple_tokens
            all_num_prompt[topic] = num_prompt

    # =========================================================================
    # SCENARIO 4: Enron Email Extraction
    # =========================================================================
    elif scenario_name == "enron_email_extraction":
        constructor = MessageConstructor(
            PROMPT_TEMPLATE[template]['sys_prompt'],
            PROMPT_TEMPLATE[template]['task_desc'], 
            is_raw_input=True
        )
        
        for prompt_type in prompt_types:
            print("prompt_type", prompt_type)
            dataset, name_list, email_list = load_enron_email_prompts(
                data_file, 
                prompt_type
            )
            
            if dataset_size < 1 or dataset_size > len(dataset):
                dataset_size = len(dataset)
            else:
                dataset = dataset[:dataset_size]

            result_dir = ScenarioName2ResultDir[scenario_name]
            os.makedirs(result_dir, exist_ok=True)
            out_file = os.path.join(
                result_dir,
                "{}_s{}_{}_{}.jsonl".format(
                    prompt_type, dataset_size,  model_path, seed
                ).replace("/", "_").replace(" ", "")
            )
            print(f"Saving to {out_file}")
            
            price_cost, prompt_tokens, comple_tokens, num_prompt = generate_and_save(
                max_tokens, 
                t,
                dataset=dataset, 
                gt_list=[email_list, name_list],
                constructor=constructor, 
                out_file=out_file, 
                batch_size=batch_size,
                model=model,
                keynames=['gt-email', 'gt-name']
            )
            
            all_cost[prompt_type] = price_cost
            all_prompt_tokens[prompt_type] = prompt_tokens
            all_comple_tokens[prompt_type] = comple_tokens
            all_num_prompt[prompt_type] = num_prompt

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("*********final results summary******")

    cost_over_all_settings = 0
    prompt_tokens_over_all_settings = 0
    comple_tokens_over_all_settings = 0
    num_prompt_over_all_settings = 0
    
    for key in all_cost.keys():
        print("scenario: ", key)
        print("Cost: ", all_cost[key])
        print("Num of prompt tokens: ", all_prompt_tokens[key])
        print("Num of completion tokens: ", all_comple_tokens[key])
        cost_over_all_settings += all_cost[key]
        prompt_tokens_over_all_settings += all_prompt_tokens[key]
        comple_tokens_over_all_settings += all_comple_tokens[key]
        num_prompt_over_all_settings += all_num_prompt[key]

    print("*********sum results summary******")
    print("sum - Cost: ", cost_over_all_settings)
    print("sum - Num of prompt tokens: ", prompt_tokens_over_all_settings)
    print("sum - Num of completion tokens: ", comple_tokens_over_all_settings)
    print("sum - Num of prompt: ", num_prompt_over_all_settings)
    
    print("*********calculate score ******")
    # calculate_scores(['_home_lhloc249_D1_Projects_NLDL_project_toga_output_llama-2-7b-chat-hf-0.50-wikitext-bin_0-lam_16.0-wikitext_mbe'])
    calculate_scores([model_path.replace("/", "_").replace(" ", "")], [scenario_name]) #


if __name__ == "__main__":
    FILES = [
        "./config/email_extraction_context50.yaml",
        "./config/pii_fewshot_attack.yaml",
        # "./config/pii_zeroshot.yaml"
        ]
        #"./config/email_extraction_context100.yaml",
        #"./config/email_extraction_context200.yaml",
    
    # pruned_model = "/cluster/projects/nn9342k/Loc/NLDL/toga/output/llama-2-7b-chat-hf-unlearn-p_0.80-wikitext-lam_16.0-ga_retain-v2"    
    pruned_model = "/cluster/projects/nn9342k/Loc/NLDL/toga/output/llama-3.2-3b-instruct-unlearn-p_0.80-wikitext-lam_16.0-ga_retain-v3"   
    # pruned_model = None # "/cluster/projects/nn9342k/Loc/NLDL/toga/output/llama-3.2-3b-instruct-unlearn-p_0.80-wikitext-lam_16.0-ga_retain-v3"     

    
    model_name = "/cluster/projects/nn9342k/llm_dataset/Llama-3.2-3B-Instruct"
    for config_file in FILES:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
        out_file = data['out_file'] if 'out_file' in data else None
        dataset_size = data['dataset_size'] if 'dataset_size' in data else 0
        scenario_name = data['scenario_name'] if 'scenario_name' in data else "enron_email_extraction"

        personal_infos = data['personal_infos'] if 'personal_infos' in data else []
        template = data['template'] if 'template' in data else 0
        prompt_types = data['prompt_types'] if 'prompt_types' in data else []
        data_file =  data['data_file'] if 'data_file' in data else "./data/"
        few_shot_num = data['few_shot_num'] if 'few_shot_num' in data else 0
        batch_size = data['batch_size'] if 'batch_size' in data else 5
        privacy_topics = data['privacy_topics'] if 'privacy_topics' in data else  0
        question_prompt = data['question_prompt'] if 'question_prompt' in data else 1
        max_tokens = data['max_tokens'] if "max_tokens" in data else 16
        t = data['t'] if "t" in data else 1
        seed = data['seed'] if "seed" in data else 1

        print("*********calculate score ******")
        # calculate_scores(['_cluster_projects_nn9342k_Loc_NLDL_toga_output_llama-3.2-3b-instruct-unlearn-p_0.80-wikitext-lam_16.0-ga_retain-v3'], [scenario_name])
        main(out_file, dataset_size, model_name, scenario_name, personal_infos, template, 
            prompt_types, data_file, few_shot_num, batch_size, privacy_topics, question_prompt,
            pruned_model=pruned_model,
            max_tokens=max_tokens, t=t, seed=seed)