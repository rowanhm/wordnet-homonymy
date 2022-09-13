# Compute an alignment between wn and ox for the test items
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import euclidean_distances

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.common import save_pickle, info, open_pickle, warn
from src.global_variables import mapping_dir, wn_dictionary_file, ox_dictionary_file, test_data_file

lemmatizer = WordNetLemmatizer()

info('Initialising sentence embedding models')

model_names = ['all-mpnet-base-v2', 'average_word_embeddings_glove.6B.300d', 'all-roberta-large-v1',
               'gtr-t5-xxl', 'sentence-t5-xxl']
models = [(m, SentenceTransformer(m)) for m in model_names]

similarity_metrics = [('cosine', lambda a, b: util.cos_sim(a, b).numpy()),
                      ('dot_prod', lambda a, b: util.dot_score(a, b).numpy()),
                      ('euc_dist', lambda a, b: -euclidean_distances(a, b))]


def tokens_sentence(sentence):
    return {lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sentence)
            if w.lower() not in stopwords.words('english') and w.lower() not in set(punctuation)}


info('Loading dictionaries')
wn_dict = open_pickle(wn_dictionary_file)
ox_dict = open_pickle(ox_dictionary_file)
test_items = set(open_pickle(test_data_file).keys())
alignments = defaultdict(dict)

info('Aligning...')
for j, (word, pos), in enumerate(test_items):

    if j % 10 == 0:
        info(f'On word {j + 1}/{len(test_items)}')

    if (word, pos) not in wn_dict.keys():
        warn(f"{word} ({pos}) for in WordNet keys")
        continue

    if (word, pos) not in ox_dict.keys():
        warn(f"{word} ({pos}) for in Oxford keys")
        continue

    wn_senses = wn_dict[(word, pos)].values()
    wn_poses = [defn['pos'] for defn in wn_senses]
    wn_ids = [defn['id'] for defn in wn_senses]
    wn_defs = [defn['definition'] for defn in wn_senses]

    ox_sense_subsets = [('only_mains', [s for s in ox_dict[(word, pos)].values() if s['main']]),
                        ('all', ox_dict[(word, pos)].values())]

    ox_infos = {subset_name: {
        'ox_poses': [defn['pos'] for defn in ox_senses],
        'ox_ids': [defn['id'] for defn in ox_senses],
        'ox_defs': [defn['definition'] for defn in ox_senses],
        'clusters': [defn['homograph_cluster_v1'] for defn in ox_senses]
    } for (subset_name, ox_senses) in ox_sense_subsets}

    similarities = {}

    # First do all models
    for model_name, model in models:
        wn_embeddings = model.encode(wn_defs)
        for subset_name in ox_infos.keys():
            ox_defs = ox_infos[subset_name]['ox_defs']
            ox_embeddings = model.encode(ox_defs)
            for metric_name, metric in similarity_metrics:
                # Calculate similarities; all are shape |WN| x |OX|
                similarities[f'{model_name}:{metric_name}:{subset_name}'] = metric(wn_embeddings, ox_embeddings)

    # Now do baselines
    for subset_name in ox_infos.keys():
        ox_defs = ox_infos[subset_name]['ox_defs']
        similarities[f'random:{subset_name}'] = np.random.rand(len(wn_defs), len(ox_defs))

        # Now do majority
        clusters = ox_infos[subset_name]['clusters']
        cluster_count = defaultdict(int)
        for cluster in clusters:
            cluster_count[cluster] += 1
        similarities[f'majority:{subset_name}'] = np.repeat(np.expand_dims(
            np.array([cluster_count[cluster] for cluster in clusters]), 0), len(wn_defs), 0)

        # Now do LESK
        ox_defs_toks = [tokens_sentence(sentence) for sentence in ox_defs]
        lesk_similarities = []
        for defn in wn_defs:
            sims = []
            wn_toks = tokens_sentence(defn)
            for ox_toks in ox_defs_toks:
                sims.append(len(wn_toks.intersection(ox_toks)) / min(len(wn_toks), len(ox_toks)))

            lesk_similarities.append(sims)
        similarities[f'lesk:{subset_name}'] = np.array(lesk_similarities)

    for model_name, sims in similarities.items():
        for wn_pos, wn_id, similarity in zip(wn_poses, wn_ids, sims):
            subset_name = model_name.split(':')[-1]
            ox_poses = ox_infos[subset_name]['ox_poses']
            ox_ids = ox_infos[subset_name]['ox_ids']

            best = ox_ids[np.argmax(similarity)]

            assert wn_id not in alignments[model_name].keys()
            alignments[model_name][wn_id] = best

info('Saving')
for model_name, alignment in alignments.items():
    save_pickle(mapping_dir + f'{model_name}.pkl', alignment)

info('Done')
