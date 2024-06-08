import pickle
import argparse
import numpy as np
import os
import config
import torch
import pandas as pd
import sklearn
import sklearn.metrics
import config


def load_cached(config, args):
    # load generations
    generation_path = os.path.join(config.output_dir, args.run_name, 'generations.pkl')
    if os.path.exists(generation_path):
        with open(generation_path, 'rb') as f:
            generations = pickle.load(f)
    else:
        raise ValueError

    # load likelihoods
    likelihood_path = os.path.join(config.output_dir, args.run_name, 'generations_likelihoods.pkl')
    if os.path.exists(generation_path):
        with open(likelihood_path, 'rb') as f:
            likelihoods = pickle.load(f)
    else:
        raise ValueError

    # load tokenwise importance
    token_impt_meas_model = args.token_impt_meas_model.replace('/', '-')
    tokenwise_importance_path = os.path.join(config.output_dir, args.run_name,
                                             f'tokenwise_importance_promptQuestion_model{token_impt_meas_model}.pkl')
    if os.path.exists(tokenwise_importance_path):
        with open(tokenwise_importance_path, 'rb') as f:
            token_importance = pickle.load(f)
    else:
        token_importance = None

    # load sentence similarities
    senten_sim_meas_model = args.senten_sim_meas_model.replace('/', '-')
    senten_sim_path = os.path.join(config.output_dir, args.run_name,
                                   f'sentence_similarities_promptQuestion_model{senten_sim_meas_model}.pkl')
    if os.path.exists(senten_sim_path):
        with open(senten_sim_path, 'rb') as f:
            sentence_similarities = pickle.load(f)
    else:
        sentence_similarities = None

    # load semantic clusters
    semantic_cluster_path = os.path.join(config.output_dir, args.run_name, 'semantic_clusters.pkl')
    if os.path.exists(semantic_cluster_path):
        with open(semantic_cluster_path, 'rb') as f:
            semantic_clusters = pickle.load(f)
    else:
        semantic_clusters = None

    return {
        'generations': generations,
        'likelihoods': likelihoods,
        'token_importance': token_importance,
        'sentence_similarities': sentence_similarities,
        'semantic_clusters': semantic_clusters
    }


def sar(cached_data, t=0.001, num_generation=None):
    likelihoods = cached_data['likelihoods']
    token_importance = cached_data['token_importance']
    sentence_similarities = cached_data['sentence_similarities']
    new_likelihoods = []
    # num_of_generated refers to how many sentences are generated for each question during this running
    num_of_generated = len(likelihoods[0]['token_wise_entropy'])
    if num_generation is not None:
        for sample_ids, likeli in enumerate(likelihoods):
            new_likelihoods.append({'token_wise_entropy': likeli['token_wise_entropy'][:num_generation]})
    likelihoods = new_likelihoods

    scores = []
    error_count = 0

    def semantic_weighted_log(similarities, entropies, t, num_generation=None):
        probs = torch.exp(-1 * entropies)
        weighted_entropy = []
        for idx, (prob, ent) in enumerate(zip(probs, entropies)):
            if num_generation is not None:
                if idx + 1 >= num_generation:
                    w_ent = - torch.log(
                        prob + ((torch.tensor(similarities[idx][:num_generation - 1]) / t) * probs[:idx]).sum())
                else:
                    w_ent = - torch.log(
                        prob + ((torch.tensor(similarities[idx][:num_generation - 1]) / t) * torch.cat(
                            [probs[:idx], probs[idx + 1:num_generation]])).sum())
            else:
                w_ent = - torch.log(
                    prob + ((torch.tensor(similarities[idx]) / t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            weighted_entropy.append(w_ent)
        return torch.tensor(weighted_entropy)

    for sample_idx, likeli in enumerate(likelihoods):
        gen_scores = []
        gen_token_wise_entropy = likeli['token_wise_entropy']
        for k in range(len(gen_token_wise_entropy)):
            token_wise_entropy = gen_token_wise_entropy[k].float()
            importance = token_importance[sample_idx * num_of_generated + k]
            if len(importance) == len(token_wise_entropy):
                weighted_score = ((importance / importance.sum()) * token_wise_entropy)
                gen_scores.append(torch.tensor(weighted_score).sum())
            else:
                error_count += 1
                gen_scores.append(0.0)

        similarity = sentence_similarities[sample_idx]
        gen_scores = torch.tensor(gen_scores)
        if num_generation is None or num_generation > 1:
            gen_scores = semantic_weighted_log(similarity, gen_scores, t=t, num_generation=num_generation)

        scores.append(gen_scores.mean())
    print(f'Error count: {error_count}')
    return scores


def sentence_sar(cached_data, t=0.001):
    likelihoods = cached_data['likelihoods']
    sentence_similarities = cached_data['sentence_similarities']
    scores = []
    error_count = 0

    def semantic_weighted_log(similarities, entropies, t):
        probs = torch.exp(-1 * entropies)
        weighted_entropy = []
        for idx, (prob, ent) in enumerate(zip(probs, entropies)):
            w_ent = - torch.log(
                prob + ((torch.tensor(similarities[idx]) / t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            weighted_entropy.append(w_ent)
        return torch.tensor(weighted_entropy)

    for sample_idx, likeli in enumerate(likelihoods):
        gen_scores = []
        gen_token_wise_entropy = likeli['token_wise_entropy']
        for k in range(len(gen_token_wise_entropy)):
            token_wise_entropy = gen_token_wise_entropy[k].float()
            gen_scores.append(torch.tensor(token_wise_entropy).sum())

        similarity = sentence_similarities[sample_idx]
        gen_scores = torch.tensor(gen_scores)
        gen_scores = semantic_weighted_log(similarity, gen_scores, t=t)

        scores.append(gen_scores.mean())
    print(f'Error count: {error_count}')
    return scores


def token_sar(cached_data):
    likelihoods = cached_data['likelihoods']
    token_importance = cached_data['token_importance']
    scores = []
    error_count = 0

    for sample_idx, likeli in enumerate(likelihoods):
        gen_scores = []
        gen_token_wise_entropy = likeli['token_wise_entropy']
        for k in range(len(gen_token_wise_entropy)):
            token_wise_entropy = gen_token_wise_entropy[k].float()
            importance = token_importance[sample_idx * len(gen_token_wise_entropy) + k]
            if len(importance) == len(token_wise_entropy):
                weighted_score = ((importance / importance.sum()) * token_wise_entropy)
                gen_scores.append(torch.tensor(weighted_score).sum())
            else:
                error_count += 1
                gen_scores.append(0.0)

        gen_scores = torch.tensor(gen_scores)

        scores.append(gen_scores.mean())
    print(f'Error count: {error_count}')
    return scores


def semantic_entropy(cached_data, num_generation=None):
    llh_shift = torch.tensor(5.0)
    likelihoods = cached_data['likelihoods']

    new_likelihoods = []
    if num_generation is not None:
        for sample_ids, likeli in enumerate(likelihoods):
            new_likelihoods.append({'token_wise_entropy': likeli['token_wise_entropy'][: num_generation],
                                    'semantic_set_ids': likeli['semantic_set_ids'][: num_generation]})
    likelihoods = new_likelihoods

    scores = []
    for sample_idx, likeli in enumerate(likelihoods):
        token_wise_entropy = likeli['token_wise_entropy']
        gen_entropy = torch.tensor([torch.mean(ent) for ent in token_wise_entropy]).float()
        semantic_set_ids = torch.tensor(likeli['semantic_set_ids']).to(gen_entropy.device)
        semantic_cluster_entropy = []
        for semantic_id in torch.unique(semantic_set_ids):
            semantic_cluster_entropy.append(torch.logsumexp(-1 * gen_entropy[semantic_set_ids == semantic_id], dim=0))
        semantic_cluster_entropy = torch.tensor(semantic_cluster_entropy) - llh_shift
        semantic_cluster_entropy = - torch.sum(semantic_cluster_entropy, dim=0) / torch.tensor(
            semantic_cluster_entropy.shape[0])
        scores.append(torch.mean(semantic_cluster_entropy))
    return scores


def len_normed_predictive_entropy(cached_data, num_generation):
    likelihoods = cached_data['likelihoods']
    new_likelihoods = []
    if num_generation is not None:
        for sample_ids, likeli in enumerate(likelihoods):
            new_likelihoods.append({'token_wise_entropy': likeli['token_wise_entropy'][:num_generation]})
    likelihoods = new_likelihoods
    scores = []
    for sample_idx, likeli in enumerate(likelihoods):
        token_wise_entropy = likeli['token_wise_entropy']
        gen_score = torch.tensor([torch.mean(ent) for ent in token_wise_entropy])
        scores.append(torch.mean(gen_score))
    return scores


def predictive_entropy(cached_data, num_generation):
    likelihoods = cached_data['likelihoods']
    new_likelihoods = []
    if num_generation is not None:
        for sample_ids, likeli in enumerate(likelihoods):
            new_likelihoods.append({'token_wise_entropy': likeli['token_wise_entropy'][:num_generation]})
    likelihoods = new_likelihoods
    scores = []
    for sample_idx, likeli in enumerate(likelihoods):
        token_wise_entropy = likeli['token_wise_entropy']
        gen_score = torch.tensor([torch.sum(ent) for ent in token_wise_entropy])
        scores.append(torch.mean(gen_score))
    return scores


def lexical_similarity(cached_data):
    semantic_clusters = cached_data['semantic_clusters']
    generations = cached_data['generations']
    scores = []
    for gen in generations:
        id_ = gen['id'][0]
        seman_c = semantic_clusters[id_]
        number_generation = len(seman_c['semantic_set_ids'])
        rouge_L_score = seman_c['syntactic_similarities']['rougeL']
        scores.append(torch.tensor(rouge_L_score).sum() / (number_generation * (number_generation - 1) / 2))
    return scores


def get_uncertainty(method, cached_data, args):
    if method == 'sar':
        return sar(cached_data, args.temperature, args.num_generation)
    elif method == 'token-sar':
        return token_sar(cached_data)
    elif method == 'sentence-sar':
        return sentence_sar(cached_data, args.temperature)
    elif method == 'semantic-entropy':
        return semantic_entropy(cached_data, args.num_generation)
    elif method == 'len-normed-predictive-entropy':
        return len_normed_predictive_entropy(cached_data, args.num_generation)
    elif method == 'predictive-entropy':
        return predictive_entropy(cached_data, args.num_generation)
    elif method == 'lexical-similarity':
        return lexical_similarity(cached_data)
    else:
        raise NotImplementedError


def get_correctness(generations, type, threshold, args):
    assert type in ['rougeL_to_target', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'sentsim']
    if 'bertscore' in type:
        correctness = 1 - (pd.DataFrame(generations)[type] > threshold).astype('int')
    elif 'rougeL' in type:
        correctness = (pd.DataFrame(generations)[type] > threshold).astype('int')
    elif 'sentsim' in type:
        with open(os.path.join(config.output_dir, args.run_name, 'sentsim_for_correctness.pkl'), 'rb') as f:
            correctness = pickle.load(f)
            correctness = (np.asarray(correctness) > threshold).astype(int)

    return correctness


def uncertainty_estimation(config, args):
    # load catched data
    cached_data = load_cached(config, args)
    correctness_list = []
    if args.threshold is None:
        range = np.arange(0.1, 1.0, 0.1)
        for type in args.metrics:
            for threshold in range:
                correctness_list.append(
                    {'correctness': get_correctness(cached_data['generations'], type=type, threshold=threshold,
                                                    args=args),
                     'type': type,
                     'threshold': threshold})
    else:
        for type in args.metrics:
            correctness_list.append(
                {'correctness': get_correctness(cached_data['generations'], type=type, threshold=args.threshold,
                                                args=args),
                 'type': type,
                 'threshold': args.threshold})

    with open(os.path.join(config.output_dir, args.run_name, 'uncertainty_estimation_results.txt'), 'w') as f:
        for method in args.methods:
            scores = torch.tensor(get_uncertainty(method, cached_data, args))
            # nan
            if torch.isnan(scores).any():
                scores[torch.isnan(scores)] = 0
            for eval in correctness_list:
                correctness = eval['correctness']
                type = eval['type']
                threshold = eval['threshold']
                if (correctness == 1).all() or (correctness == 0).all():
                    continue
                else:
                    auc = sklearn.metrics.roc_auc_score(1 - correctness, scores)
                    info = f'accuracy: {correctness.sum() / len(correctness)} \t ' \
                           f'method: {method} \t type: {type} \t ' \
                           f'threshold: {threshold:.4f} \t AUC: {auc:.4f}'
                    print(info)
                    f.write(info + '\n')
                    f.flush()


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--senten-sim-meas-model", default='cross-encoder/stsb-roberta-large',
                   choices=['cross-encoder/stsb-roberta-large',
                            'cross-encoder/stsb-distilroberta-base'],
                   help="desc")
    p.add_argument("--token-impt-meas-model", default='cross-encoder/stsb-roberta-large',
                   choices=['cross-encoder/stsb-roberta-large',
                            'cross-encoder/stsb-distilroberta-base'],
                   help="desc")
    p.add_argument('--methods',
                   default=[
                       'sar',
                       'sentence-sar',
                       'token-sar',
                       'semantic-entropy',
                       'len-normed-predictive-entropy',
                       'predictive-entropy',
                       'lexical-similarity'
                   ],
                   nargs='+')
    p.add_argument('--temperature', default=0.001, type=float)
    p.add_argument('--threshold', default=0.5, type=float)
    p.add_argument('--num-generation', default=5, type=int)
    p.add_argument('--metrics', default=['rougeL_to_target', 'sentsim'], nargs='+')
    p.add_argument('--run-name', default='huggyllama/llama-13b/trivia_qa/numbeams-1/max_len_of_gen-128')

    return (p.parse_args())


if __name__ == '__main__':
    args = cmdline_args()
    uncertainty_estimation(config, args)
