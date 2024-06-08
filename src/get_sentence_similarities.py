

import pickle
import argparse
import os
import tqdm
import config
from sentence_transformers.cross_encoder import CrossEncoder


def get_sentence_similarities(args):
    measure_model = CrossEncoder(model_name=args.measurement_model, num_labels=1)

    with open(f'{config.output_dir}/{args.run_name}/generations.pkl',
              'rb') as infile:
        generations = pickle.load(infile)
    similarity_list = []
    for sample_idx, gen in enumerate(tqdm.tqdm(generations)):
        generated_texts = gen['cleaned_generated_texts']
        similarities = {}
        for i in range(len(generated_texts)):
            similarities[i] = []
        question = gen['question']

        for i in range(len(generated_texts)):
            for j in range(i+1, len(generated_texts)):
                gen_i = question + generated_texts[i]
                gen_j = question + generated_texts[j]
                similarity_i_j = measure_model.predict([gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        similarity_list.append(similarities)

    measure_model_name = args.measurement_model.replace('/', '-')
    with open(os.path.join(
            f'{config.output_dir}/{args.run_name}/sentence_similarities_promptQuestion_model{measure_model_name}.pkl'),
              'wb') as f:
        pickle.dump(similarity_list, f)


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--measurement-model", default='cross-encoder/stsb-roberta-large',
                   choices=['cross-encoder/stsb-roberta-large',
                            'cross-encoder/stsb-distilroberta-base'],
                   help="desc")
    p.add_argument('--run-name', default='')

    return (p.parse_args())


if __name__ == '__main__':
    args = cmdline_args()
    get_sentence_similarities(args)

