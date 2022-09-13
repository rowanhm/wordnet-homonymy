from collections import defaultdict

from src.common import open_pickle, info, save_pickle
from src.global_variables import full_alignment_file, wn_dictionary_file, ox_dictionary_file, between_pos_pkl_file, \
    within_pos_pkl_file, raw_pkl_file
from src.homograph_coarsener_v1 import HomographCoarsenerV1

alignment = open_pickle(full_alignment_file)
wn_dict = open_pickle(wn_dictionary_file)
ox_dict = open_pickle(ox_dictionary_file)

hc = HomographCoarsenerV1()


def build_lemma_dict(ordered_lemmas, word, pos):
    clusters = hc.coarsen_homographs(ordered_lemmas)

    # Rename clusters
    all_clusters = set(clusters)
    cluster_name_lookup = {}
    for index, clust in enumerate(all_clusters):
        if pos is None:
            name = f'{word}.{index+1}'
        else:
            name = f'{word}.{pos}.{index+1}'
        cluster_name_lookup[clust] = name

    lemma_to_cluster_lookup = {lem: cluster_name_lookup[clust] for lem, clust in zip(ox_lemmas_ordered, clusters)}

    return lemma_to_cluster_lookup


words = defaultdict(set)
for (word, pos) in wn_dict.keys():
    words[word].add(pos)

between_pos_homographs = {}
within_pos_homographs = {}
raw_homographs = {}

for i, (word, poses) in enumerate(words.items()):

    if i % 100 == 0:
        info(f'Processing word {i}/{len(words.items())}')

    all_wn_ids = []
    all_ox_ids = []
    all_ox_lemmas_lookup = {}

    # Compute within_pos
    for pos in poses:

        if (word, pos) not in ox_dict.keys():
            continue

        wn_ids = list(wn_dict[(word, pos)].keys())
        ox_aligned_ids = [alignment[wn_id] for wn_id in wn_ids]

        # Get possible lemmas
        ox_entries = ox_dict[(word, pos)]
        ox_lemmas_filtered = {ox_id: entry['coarse_lemma_id'] for ox_id, entry in ox_entries.items()
                              if ox_id in ox_aligned_ids}

        # Save this info for the between_pos compute
        all_wn_ids.extend(wn_ids)
        all_ox_ids.extend(ox_aligned_ids)
        for ox_id, lemma in ox_lemmas_filtered.items():
            assert ox_id not in all_ox_lemmas_lookup.keys()
            all_ox_lemmas_lookup[ox_id] = lemma

        # Get homograph clusters
        ox_lemmas_ordered = list(set(ox_lemmas_filtered.values()))
        ox_lemma_to_cluster = build_lemma_dict(ox_lemmas_ordered, word, pos)

        # Check no clusters with these names exist
        for cluster in set(ox_lemma_to_cluster.values()):
            assert cluster not in within_pos_homographs.values()

        # Add to clustering
        for wn_id, ox_id in zip(wn_ids, ox_aligned_ids):
            assert wn_id not in within_pos_homographs.keys()
            assert wn_id not in raw_homographs.keys()

            ox_lemma = ox_lemmas_filtered[ox_id]
            cluster = ox_lemma_to_cluster[ox_lemma]

            within_pos_homographs[wn_id] = cluster
            raw_homographs[wn_id] = ox_lemma

    # Compute between_pos
    ox_lemmas_ordered = list(all_ox_lemmas_lookup.values())
    ox_lemma_to_cluster = build_lemma_dict(ox_lemmas_ordered, word, None)

    # Check no clusters with these names exist
    for cluster in set(ox_lemma_to_cluster.values()):
        assert cluster not in between_pos_homographs.values()

    # Add to clustering
    for wn_id, ox_id in zip(all_wn_ids, all_ox_ids):
        assert wn_id not in between_pos_homographs.keys()

        ox_lemma = all_ox_lemmas_lookup[ox_id]
        cluster = ox_lemma_to_cluster[ox_lemma]

        between_pos_homographs[wn_id] = cluster

info('Saving')
save_pickle(between_pos_pkl_file, between_pos_homographs)
save_pickle(within_pos_pkl_file, within_pos_homographs)
save_pickle(raw_pkl_file, raw_homographs)

hc.save()
