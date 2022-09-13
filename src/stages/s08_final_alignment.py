# Compute an alignment between wn and ox
import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.common import save_pickle, info, open_pickle, warn
from src.global_variables import wn_dictionary_file, ox_dictionary_file, full_alignment_file

info('Initialising sentence embedding model')

model = SentenceTransformer('sentence-t5-xxl')
metric = lambda a, b: util.dot_score(a, b).numpy()

info('Loading dictionaries')

wn_dict = open_pickle(wn_dictionary_file)
ox_dict = open_pickle(ox_dictionary_file)
all_items = list(wn_dict.keys())

alignment = {}

info('Aligning...')
for j, (word, pos), in enumerate(all_items):

    if j % 10 == 0:
        info(f'On word {j + 1}/{len(all_items)}')

    assert (word, pos) in wn_dict.keys()
    if (word, pos) not in ox_dict.keys():
        warn(f"{word} ({pos}) not in Oxford keys")
        continue

    wn_senses = wn_dict[(word, pos)].values()
    wn_ids = [defn['id'] for defn in wn_senses]
    wn_defs = [defn['definition'] for defn in wn_senses]

    ox_senses = ox_dict[(word, pos)].values()
    ox_ids = [defn['id'] for defn in ox_senses]
    ox_defs = [defn['definition'] for defn in ox_senses]

    similarities = {}

    wn_embeddings = model.encode(wn_defs)
    ox_embeddings = model.encode(ox_defs)
    sims = metric(wn_embeddings, ox_embeddings)

    for wn_id, similarity in zip(wn_ids, sims):
        best = ox_ids[np.argmax(similarity)]

        assert wn_id not in alignment.keys()
        alignment[wn_id] = best

info('Saving')
save_pickle(full_alignment_file, alignment)

info('Done')
