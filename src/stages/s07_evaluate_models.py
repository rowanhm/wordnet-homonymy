# Eval the quality of each alignment in the alignments folder
import glob
import itertools
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, f1_score, accuracy_score

from src.common import open_pickle, info, warn, flatten, save_csv
from src.global_variables import ox_dictionary_file, test_data_file, mapping_dir, wn_dictionary_file, \
    results_file

random.seed(0)

info('Loading data')
test_clusters = open_pickle(test_data_file)
ox_dict = open_pickle(ox_dictionary_file)
wn_dict = open_pickle(wn_dictionary_file)

info('Loading alignments')
alignments = {}
for file in glob.iglob(mapping_dir + '*.pkl'):
    model_name = '.'.join(os.path.basename(file).split('.')[:-1])
    alignments[model_name] = open_pickle(file)

results = defaultdict(dict)
outputs_lemmas = dict()
outputs_senses = dict()

info(f'Computing results')
for model_name, alignment in alignments.items():
    cluster_index = 0

    # Lists of lists
    all_correct_clusterings = []
    all_pred_clusterings = []
    all_correct_ids = []
    all_pred_ids = []

    for (word, pos), datapoints in test_clusters.items():
        if (word, pos) not in wn_dict.keys() or (word, pos) not in ox_dict.keys():
            warn(f'({word}, {pos}) missing')
            continue

        wn_sense_ids = []
        predicted_ox_sense_ids = []
        correct_ox_sense_ids = []
        predicted_clusters = []
        correct_clusters = []

        if len(datapoints) == 0:
            continue

        for wn_sense_id, datapoint in datapoints.items():

            if wn_sense_id in alignment.keys():
                predicted_ox_sense_id = alignment[wn_sense_id]
                predicted_ox_sense_ids.append(predicted_ox_sense_id)

                correct_ox_sense_id = datapoint['ox_id']
                correct_ox_sense_ids.append(correct_ox_sense_id)

                correct_cluster = datapoint['cluster']
                predicted_cluster = ox_dict[(word, pos)][predicted_ox_sense_id]['homograph_cluster_v1']
                predicted_clusters.append(predicted_cluster)
                correct_clusters.append(correct_cluster)
            else:
                warn(f'{wn_sense_id} missing from alignment {model_name}')

        # If the prediction is in the correct options, make that the correct one, otherwise choose randomly
        # Filter unlabelled
        corr_pred = [(c, p) for (c, p) in zip(correct_ox_sense_ids, predicted_ox_sense_ids) if c != '']
        if len(corr_pred) > 0:
            correct_ox_sense_ids, predicted_ox_sense_ids = zip(*corr_pred)
            all_correct_ids.append(correct_ox_sense_ids)
            all_pred_ids.append(predicted_ox_sense_ids)

        # Skip any which have only a single homograph after removing excluded
        ox_definitions = ox_dict[(word, pos)].values()
        all_correct_clusterings.append(correct_clusters)
        all_pred_clusterings.append(predicted_clusters)

    # Filter data
    all_correct_clusterings_filtered = []
    all_pred_clusterings_filtered = []
    for correct_cluster, pred_cluster in zip(all_correct_clusterings, all_pred_clusterings):
        correct_filtered = []
        pred_filtered = []
        for c, p in zip(correct_cluster, pred_cluster):
            if c != '':
                correct_filtered.append(c)
                pred_filtered.append(p)
        assert len(correct_filtered) == len(pred_filtered)
        if len(correct_filtered) > 0:
            all_correct_clusterings_filtered.append(correct_filtered)
            all_pred_clusterings_filtered.append(pred_filtered)

    sets = [(all_correct_clusterings, all_pred_clusterings, 'unfiltered'),
            (all_correct_clusterings_filtered, all_pred_clusterings_filtered, 'filtered')]

    # Save output
    outputs_lemmas[model_name] = flatten(all_pred_clusterings)
    if 'true' in outputs_lemmas.keys():
        assert outputs_lemmas['true'] == flatten(all_correct_clusterings)
    else:
        outputs_lemmas['true'] = flatten(all_correct_clusterings)
    outputs_senses[model_name] = flatten(all_pred_ids)
    if 'true' in outputs_senses.keys():
        assert outputs_senses['true'] == flatten(all_correct_ids)
    else:
        outputs_senses['true'] = flatten(all_correct_ids)

    # Cluster results
    for (correct, predicted, name) in sets:
        results[model_name][f'micro_ami_{name}'] = np.mean(
            [adjusted_mutual_info_score(correct_cluster, predicted_cluster) for (correct_cluster, predicted_cluster)
             in zip(correct, predicted)])
        results[model_name][f'micro_cluster_acc_{name}'] = np.mean(
            [accuracy_score(correct_cluster, predicted_cluster) for (correct_cluster, predicted_cluster) in
             zip(correct, predicted)])
        results[model_name][f'micro_cluster_f1_{name}'] = np.mean(
            [f1_score(correct_cluster, predicted_cluster, average='weighted') for
             (correct_cluster, predicted_cluster) in zip(correct, predicted)])

        correct_clusters_flattened = flatten(correct)
        pred_clusters_flattened = flatten(predicted)

        results[model_name][f'macro_ami_{name}'] = adjusted_mutual_info_score(
            correct_clusters_flattened, pred_clusters_flattened)
        results[model_name][f'macro_cluster_acc_{name}'] = accuracy_score(correct_clusters_flattened,
                                                                                      pred_clusters_flattened)
        results[model_name][f'macro_cluster_f1_{name}'] = f1_score(correct_clusters_flattened,
                                                                               pred_clusters_flattened,
                                                                               average='weighted')

    # Sense results
    correct_senses_flattened = flatten(all_correct_ids)
    pred_senses_flattened = flatten(all_pred_ids)
    results[model_name][f'micro_sense_acc'] = np.mean(
        [accuracy_score(correct_senses, predicted_senses) for (correct_senses, predicted_senses) in
         zip(all_correct_ids, all_pred_ids)])
    results[model_name][f'micro_sense_f1'] = np.mean(
        [f1_score(correct_senses, predicted_senses, average='weighted') for (correct_senses, predicted_senses)
         in zip(all_correct_ids, all_pred_ids)])
    results[model_name][f'macro_sense_acc'] = accuracy_score(correct_senses_flattened,
                                                                         pred_senses_flattened)
    results[model_name][f'macro_sense_f1'] = f1_score(correct_senses_flattened,
                                                                  pred_senses_flattened, average='weighted')

info('Printing and saving')
model_name_map = {
    'sentence-t5-xxl:cosine:all': 'Sentence-T5',
    'random:only_mains': 'random',
    'majority:all': 'most',
    'lesk:all': 'LESK',
    'average_word_embeddings_glove.6B.300d:cosine:all': 'GloVe',
    'all-roberta-large-v1:cosine:all': 'RoBERTa',
    'all-mpnet-base-v2:cosine:all': 'MPNet'}

results_list = []
for model_name, result_dict in results.items():
    result_dict['MODEL_NAME'] = model_name
    results_list.append(result_dict)

for result in sorted(results_list, key=lambda r: r['MODEL_NAME']):
    if result['MODEL_NAME'] not in model_name_map.keys():
        continue
    numbers = []
    for key in ['macro_cluster_acc_unfiltered', 'macro_cluster_f1_unfiltered', 'macro_sense_acc',
                'macro_sense_f1']:
        numbers.append('$'+f'{result[key]:.2f}'[1:]+'$')
    print(model_name_map[result['MODEL_NAME']] + ' & ' + ' & '.join(numbers) + ' \\\\')

save_csv(results_file, results_list)

info('Significance')


def significance(assignments_1, assignments_2, correct, metric, r=10000):
    assert len(assignments_1) == len(assignments_2)
    assert len(assignments_2) == len(correct)
    length = len(correct)

    def evaluate_diff(ass_1, ass_2):
        result_1 = metric(correct, ass_1)
        result_2 = metric(correct, ass_2)
        return np.abs(result_1 - result_2)

    observed_diff = evaluate_diff(assignments_1, assignments_2)

    s = 0  # s is number of times the difference is greater that observed

    for i in range(r):

        # Shuffle
        shuffled_1, shuffled_2 = zip(
            *[(a, b) if bool(random.getrandbits(1)) else (b, a) for (a, b) in zip(assignments_1, assignments_2)])
        assert len(shuffled_1) == length
        assert len(shuffled_2) == length
        shuffled_diff = evaluate_diff(shuffled_1, shuffled_2)

        if shuffled_diff >= observed_diff:
            s += 1

    p = (s + 1) / (r + 1)
    return p


acc_metric = lambda corr, pred: accuracy_score(corr, pred)
f1_metric = lambda corr, pred: f1_score(corr, pred, average='weighted')
metrics = [('f1', f1_metric), ('acc', acc_metric)]
datasets = [('sense', outputs_senses), ('lemma', outputs_lemmas)]

for data_name, data in datasets:
    perfect = [(k if k != '' else 'wrong') for k in data['true']]
    p = significance(data['sentence-t5-xxl:cosine:all'], perfect, data['true'], acc_metric)
    sig = p <= 0.01
    info(f'Difference in {data_name} between Sentence-T5 and perfect under acc ' + (
        'significant' if sig else 'INsignificant') + f' (p={p})')

for ((model_key_1, name_1), (model_key_2, name_2)) in itertools.combinations(list(model_name_map.items()), 2):
    if {name_1, name_2} not in [{'Sentence-T5', 'RoBERTa'}, {'GloVe', 'LESK'}, {'Sentence-T5', 'MPNet'}, {'Sentence-T5', 'GloVe'}]:
        continue
    for data_name, data in datasets:
        for metric_name, metric in metrics:
            p = significance(data[model_key_1], data[model_key_2], data['true'], metric)
            sig = p <= 0.01
            info(f'Difference in {data_name} between {name_1} and {name_2} under {metric_name} ' + (
                'significant' if sig else 'INsignificant') + f' (p={p})')
