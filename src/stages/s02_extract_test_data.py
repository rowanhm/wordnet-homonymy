from collections import defaultdict

from src.common import open_dict_csv, save_pickle, info
from src.global_variables import test_data_file, test_alignment_file

alignments = open_dict_csv(test_alignment_file)

test_data = defaultdict(dict)

info('Extracting...')
for alignment in alignments:
    word = alignment['word']
    pos = alignment['pos']

    wn_id = alignment['wn_id']
    ox_id = alignment['ox_id']
    if ox_id is None:
        ox_id = ''
    cluster = alignment['lemma']
    if cluster is None:
        cluster = ''

    test_data[(word, pos)][wn_id] = {
        'ox_id': ox_id,
        'cluster': cluster
    }


info('Done.')
save_pickle(test_data_file, test_data)
