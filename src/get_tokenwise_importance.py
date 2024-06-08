
import pickle
import argparse
import os

import tqdm

import config
import torch
import pandas as pd
import sklearn
import sklearn.metrics
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer


def get_tokenwise_importance(args):
    if 'cross-encoder' in args.measurement_model:
        measure_model = CrossEncoder(model_name=args.measurement_model, num_labels=1)
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False)

    with open(f'{config.output_dir}/{args.run_name}/generations.pkl',
              'rb') as infile:
        generations = pickle.load(infile)

    scores = []
    token_importance_list = []
    for sample_idx, gen in enumerate(tqdm.tqdm(generations)):
        # likelihoods = gen['token_wise_entropy']
        gen_scores = []
        for k in range(len(gen['cleaned_generated_texts'])):
            generated_text = gen['cleaned_generated_texts'][k]
            question = gen['question']
            tokenized = torch.tensor(tokenizer.encode(generated_text, add_special_tokens=False))

            # likelihood = likelihoods[k]['original_token_wise_entropy']
            token_importance = []
            # measure cosine similarity by removing each token and compare the similarity
            for token in tokenized:
                similarity_to_original = measure_model.predict([question + generated_text,
                                                                question + generated_text.replace(
                                                                    tokenizer.decode(token, skip_special_tokens=True),
                                                                    '')])
                token_importance.append(1 - torch.tensor(similarity_to_original))

            token_importance = torch.tensor(token_importance).reshape(-1)
            token_importance_list.append(token_importance)

        scores.append(torch.tensor(gen_scores).mean())
    scores = torch.tensor(scores)
    if torch.isnan(scores).sum() > 0:
        scores[torch.isnan(scores).nonzero(as_tuple=True)] = 0

    # auc = sklearn.metrics.roc_auc_score(1 - correctness, scores)
    # print(f'AUC: {auc}')

    measure_model_name = args.measurement_model.replace('/', '-')
    token_wise_importance_path = os.path.join(config.output_dir, args.run_name,
                                              f'tokenwise_importance_promptQuestion_model{measure_model_name}.pkl')
    with open(token_wise_importance_path, 'wb') as f:
        pickle.dump(token_importance_list, f)


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--measurement-model", default='cross-encoder/stsb-roberta-large',
                   choices=['cross-encoder/stsb-roberta-large',
                            'cross-encoder/stsb-distilroberta-base'],
                   help="desc")
    p.add_argument('--tokenizer-model', default='facebook/opt-13b')
    p.add_argument('--run-name', default='')

    return (p.parse_args())


if __name__ == '__main__':
    args = cmdline_args()
    get_tokenwise_importance(args)
