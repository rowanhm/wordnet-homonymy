from collections import defaultdict
from nltk.corpus import wordnet as wn

from src.common import open_pickle, info
from src.global_variables import between_pos_pkl_file, within_pos_pkl_file, wn_dictionary_file, ox_dictionary_file, \
    raw_pkl_file
from src.homograph_coarsener_v1 import HomographCoarsenerV1

wn_dict = open_pickle(wn_dictionary_file)
ox_dict = open_pickle(ox_dictionary_file)

hc = HomographCoarsenerV1()

words = defaultdict(set)
for (word, pos) in wn_dict.keys():
    words[word].add(pos)

for name, cluster_dict_file in [('raw', raw_pkl_file), ('within', within_pos_pkl_file), ('between', between_pos_pkl_file)]:

    info(f'Analysing {name}')

    word_stats_by_pos = defaultdict(set)
    homographs = set()

    cluster_dict = open_pickle(cluster_dict_file)

    for word, poses in words.items():

        if name == 'between':
            ox_lemmas = set()
            pos_to_lemmas = {}
            for pos in poses:
                if (word, pos) in ox_dict.keys():
                    pos_lemmas = {ox_data['coarse_lemma_id'] for ox_data in ox_dict[(word, pos)].values()}
                    ox_lemmas = ox_lemmas.union(pos_lemmas)
                    pos_to_lemmas[pos] = pos_lemmas

            if len(ox_lemmas) > 0:  # Skips words where all pos missing
                ox_lemmas = list(ox_lemmas)
                if len(ox_lemmas) > 1:
                    ox_coarse_homographs = hc.coarsen_homographs(ox_lemmas)
                    ox_lemma_map = {lem: clust for lem, clust in zip(ox_lemmas, ox_coarse_homographs)}
                else:
                    ox_lemma_map = {ox_lemmas[0]: ox_lemmas[0]}

        elif name == 'raw':
            ox_lemmas = set()
            pos_to_lemmas = {}
            for pos in poses:
                if (word, pos) in ox_dict.keys():
                    pos_lemmas = {ox_data['coarse_lemma_id'] for ox_data in ox_dict[(word, pos)].values()}
                    ox_lemmas = ox_lemmas.union(pos_lemmas)
                    pos_to_lemmas[pos] = pos_lemmas
            ox_lemma_map = {}
            for lemma in ox_lemmas:
                ox_lemma_map[lemma] = lemma

        for pos in poses:

            entries = wn_dict[(word, pos)]

            word_stats_by_pos[pos + ':total'].add(word)

            if (word, pos) not in ox_dict.keys():
                word_stats_by_pos[pos+':missing'].add(word)
                continue

            wn_ids = entries.keys()
            clusters = set()
            for wn_id in wn_ids:
                clusters.add(cluster_dict[wn_id])

            assert len(clusters) > 0
            if len(clusters) > 1:
                word_stats_by_pos[pos+':homographs'].add(word)
                homographs.add((word,pos))

            # Get extra stats
            if name == 'within':
                ox_lemmas = list({ox_data['coarse_lemma_id'] for ox_data in ox_dict[(word, pos)].values()})
                if len(ox_lemmas) > 1:
                    ox_coarse_homographs = set(hc.coarsen_homographs(ox_lemmas))
                    if len(ox_coarse_homographs) > 1:
                        word_stats_by_pos[pos + ':potential_homographs'].add(word)
            else:
                assert name == 'between' or name == 'raw'
                pos_lemmas = pos_to_lemmas[pos]
                coarse_clusters = set()
                for lemma in pos_lemmas:
                    coarse_clusters.add(ox_lemma_map[lemma])
                if len(coarse_clusters) > 1:
                    word_stats_by_pos[pos + ':potential_homographs'].add(word)

    poses = ['noun', 'verb', 'adj', 'adv']  # , sum, total
    codes = ['total', 'missing', 'potential_homographs', 'homographs']

    print('pos & ' + ' & '.join(codes) + " \\\\")
    for pos in poses:
        output = pos + ' & ' + " & ".join(['$'+str(len(word_stats_by_pos[f'{pos}:{code}']))+'$' for code in codes]) + " \\\\"
        # output += " & $" + str(sum([len(word_stats_by_pos[f'{pos}:{code}']) for pos in poses])) + '$'
        print(output)
    total = []
    any = []
    for code in codes:
        all_words = set()
        for pos in poses:
            all_words = all_words.union(word_stats_by_pos[f'{pos}:{code}'])
        any.append(len(all_words))
        total.append(sum([len(word_stats_by_pos[f'{pos}:{code}']) for pos in poses]))
    print("total & " + " & ".join(['$' + str(stat) +'$' for stat in total]) + " \\\\")
    print("any & " + " & ".join(['$' + str(stat) +'$' for stat in any]) + " \\\\")
    print(', '.join(sorted(list({'\\word{' + word + '}' for (word, _) in homographs}))))
    print({word for (word, _) in homographs})

    print(sum([len(wn.synsets(w)) for w in {word for (word, _) in homographs}]))

info('Done')
hc.save()
