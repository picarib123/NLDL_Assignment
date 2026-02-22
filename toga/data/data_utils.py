import os
from typing import Optional
import random
import numpy as np
import torch
import datasets
from datasets import IterableDataset, load_dataset
from itertools import chain
from torch.utils.data import DataLoader

# Automatically downloaded dataset from Hugging Face
def load_hf_dataset_wikitext(split='train', n_shards=None):

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    ds = ds.select_columns("text")

    return ds

def get_fineweb_edu(train: bool = True):
    print("Loading FineWeb-Edu v2")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    # tokens_to_load = num_tokens
    # if train:
    # dataset = dataset.filter(lambda x: x.get('score', 0.0) >= 0.9)
    dataset = dataset.select_columns("text")
    dataset = dataset.take(1000 * 4096)
    dataset = dataset.shuffle(seed=42, buffer_size=100_000)
    # else:
    #     dataset = dataset.select(range(dataset.num_rows//2, dataset.num_rows))
    # dataset = dataset.shuffle(seed=0)
    # data_iter = iter(dataset)
    # data = []
    # while tokens_to_load > 0:
    #     sample = next(data_iter)
    #     tokenized_sample = tokenizer.encode(sample["text"], return_tensors="pt", add_special_tokens=False)
    #     # print(tokenized_sample)
    #     # print(min(tokenized_sample.shape[1], tokens_to_load))
    #     tokenized_sample = tokenized_sample[:, :int(min(tokenized_sample.shape[1], tokens_to_load))]
    #     # Split the sequence into multiple samples if it is too long
    #     # Just throwing away extra tokens would introduce bias to the dataset
    #     while tokenized_sample.shape[1] > sequence_length:
    #         data.append(tokenized_sample[:, :sequence_length])
    #         tokenized_sample = tokenized_sample[:, sequence_length:]
    #         tokens_to_load -= sequence_length
    #     data.append(tokenized_sample)
    #     tokens_to_load -= tokenized_sample.shape[1]
    # print(f"Total tokens loaded: {sum([sample.shape[1] for sample in data])}")

    return dataset


def is_distirbuted_dataset(iterable):
    return False


def dataloader_creator(dataset,
                       tokenizer,
                       batch_size,
                       block_size,
                       rank,
                       world_size,
                       num_workers=1,
                       cycling=False,
                       shuffle_seed=1,
                       shuffle_buffer=0,
                       sample_group_size=50,
                    #    num_tokens=-1,
                       ignored_token=None):

    print(f"--- dataloader_creator: dataset type = {type(dataset)} ---")

    block_size = block_size + 1
    if ignored_token is None:
        ignored_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def pad_list(x):
        if len(x) < block_size:
            x += [ignored_token] * (block_size - len(x))
        return x

    def group_tokens(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [pad_list(t[i: i + block_size]) for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result


    dataset = dataset.map(lambda x: {'input_ids': tokenizer.encode(x["text"])},
                          remove_columns='text',
                          batched=False)  

    dataset = dataset.map(group_tokens, batched=True, batch_size=sample_group_size)

    dataset = dataset.map(lambda x: {'input_ids': torch.LongTensor(x['input_ids'])})
    dataset = dataset.map(lambda x: {
        'labels': x['input_ids'][1:],
        'input_ids': x['input_ids'][:-1]
    })


    def collate_fn(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'labels':    torch.stack([x['labels'] for x in batch])
        }

    def set_worker_sharing_strategy(worker_id: int) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        worker_init_fn=set_worker_sharing_strategy
    )

    if cycling:
        dataloader = cycle_loader(dataloader)

    return dataloader

# def limit_by_tokens(dataset, num_tokens):
#     if num_tokens <= 0:
#         yield from dataset
#         return

#     total = 0
#     for example in dataset:
#         tokens_in_example = example['input_ids'].shape[-1]
#         if total + tokens_in_example > num_tokens:
#             break
#         yield example
#         total += tokens_in_example
#         if total >= num_tokens:
#             break

def cycle_loader(dataloader):
    dataloader_iterator = iter(dataloader)
    while True:
        try:
            yield next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            yield next(dataloader_iterator)
