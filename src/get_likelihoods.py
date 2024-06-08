import argparse
import os
import pickle
import random
import sys

import datasets
import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation-model', type=str, default='huggyllama/llama-7b')
parser.add_argument('--run-name', type=str, default='huggyllama/llama-7b/coqa')
# parser.add_argument('--evaluation-model', type=str, default='facebook/opt-125m')
# parser.add_argument('--run-name', type=str, default='facebook/opt-125m/coqa')
args = parser.parse_args()

device = 'cuda'
import config

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

# Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

run_name = args.run_name

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

with open(f'{config.output_dir}/{run_name}/generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/{run_name}/semantic_clusters.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)

if 'opt-30b' in args.evaluation_model or 'llama-13b' in args.evaluation_model:
    model = AutoModelForCausalLM.from_pretrained(args.evaluation_model, torch_dtype=torch.float16, device_map='auto')
else:
    model = AutoModelForCausalLM.from_pretrained(args.evaluation_model, torch_dtype=torch.float16).cuda()

tokenizer = AutoTokenizer.from_pretrained(args.evaluation_model, use_fast=False)

if 'opt' in args.evaluation_model:
    pad_token_id = tokenizer.pad_token_id
elif 'llama' in args.evaluation_model:
    pad_token_id = 1
else:
    raise NotImplementedError


def get_token_wise_entropies(generation, logits, labels, vocab_size):
    shifted_logits = logits[..., :-1, :].reshape(-1, vocab_size)
    shifted_labels = labels[..., 1:].reshape(-1)
    token_wise_entropy = torch.nn.CrossEntropyLoss(reduction='none')(shifted_logits, shifted_labels)
    token_wise_entropy = token_wise_entropy[shifted_labels != -100].cpu().detach()
    generation = generation[labels != -100]
    assert token_wise_entropy.size(0) == generation.size(0), f'{token_wise_entropy.shape} \t {generation.shape}'

    return token_wise_entropy


def get_neg_loglikelihoods(model, sequences):
    with torch.no_grad():
        result = []
        for sample in tqdm.tqdm(sequences):
            result_dict = {}
            prompt = sample['prompt']
            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            else:
                generations = sample['generations'].to(device)
            id_ = sample['id']

            average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
            pointwise_mutual_information = torch.zeros((generations.shape[0],))
            sequence_embeddings = []

            token_wise_entropy_list = []

            for generation_index in range(generations.shape[0]):
                prompt = prompt[prompt != pad_token_id]
                generation = generations[generation_index][generations[generation_index] != pad_token_id]

                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                target_ids = generation.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                token_wise_entropy = get_token_wise_entropies(generation, model_output.logits, target_ids,
                                                              vocab_size=model.config.vocab_size)
                token_wise_entropy_list.append(token_wise_entropy)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood = model_output['loss']
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index]

            sequence_embeddings = torch.stack(sequence_embeddings)

            result_dict['prompt'] = prompt
            result_dict['generations'] = generations
            result_dict['cleaned_generated_texts'] = sample['cleaned_generated_texts']
            result_dict['question'] = sample['question']
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            result_dict['sequence_embeddings'] = sequence_embeddings
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
            result_dict['id'] = id_
            result_dict['token_wise_entropy'] = token_wise_entropy_list
            result.append(result_dict)

        return result


likelihoods = get_neg_loglikelihoods(model, sequences)

with open(f'{config.output_dir}/{run_name}/generations_likelihoods.pkl',
          'wb') as outfile:
    pickle.dump(likelihoods, outfile)
