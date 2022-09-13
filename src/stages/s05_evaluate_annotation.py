from collections import defaultdict

from sklearn.metrics import cohen_kappa_score

from src.common import open_dict_csv, info, open_pickle
from src.global_variables import annotator_1_alignment_file, ox_dictionary_file, wn_dictionary_file, \
    annotator_2_alignment_file, test_alignment_file
from src.homograph_coarsener_v1 import HomographCoarsenerV1

annotator_1 = sorted(open_dict_csv(annotator_1_alignment_file), key=lambda d: f"{d['word']}:{d['pos']}:{d['wn_id']}")
annotator_2 = sorted(open_dict_csv(annotator_2_alignment_file), key=lambda d: f"{d['word']}:{d['pos']}:{d['wn_id']}")

ox_dict = open_pickle(ox_dictionary_file)
wn_dict = open_pickle(wn_dictionary_file)

# % Agreements
sense_percent = sum([anno1['ox_id'] == anno2['ox_id'] for (anno1, anno2) in zip(annotator_1, annotator_2)]) / len(annotator_1)
lemma_percent = sum([anno1['lemma'] == anno2['lemma'] for (anno1, anno2) in zip(annotator_1, annotator_2)]) / len(annotator_1)

info(f'Sense percent agreement {sense_percent}, lemma {lemma_percent}')

# Get number unassigned
unassigned1 = sum([1 for i in annotator_1 if i['ox_id'] == '' and i['lemma'] == ''])
unassigned2 = sum([1 for i in annotator_2 if i['ox_id'] == '' and i['lemma'] == ''])
info(f'Annotator 1 left {unassigned1} unassigned; annotator 2 {unassigned2}')

# Get number lemma only
lemma1 = sum([1 for i in annotator_1 if i['ox_id'] == '' and i['lemma'] != ''])
lemma2 = sum([1 for i in annotator_2 if i['ox_id'] == '' and i['lemma'] != ''])
info(f'Annotator 1 only assigned lemma for {lemma1}; annotator 2 {lemma2}')

# Rearrange to (word,pos) mapping
anno1_datapoints = defaultdict(dict)
anno2_datapoints = defaultdict(dict)
for d1, d2 in zip(annotator_1, annotator_2):
    assert d1['word'] == d2['word']
    assert d1['pos'] == d2['pos']
    assert d1['wn_id'] == d2['wn_id']
    wn_id = d1['wn_id']
    key = (d1['word'], d1['pos'])
    anno1_datapoints[key][wn_id] = (d1['ox_id'], d1['lemma'])
    anno2_datapoints[key][wn_id] = (d2['ox_id'], d2['lemma'])

# Flatten into list of binary sense pairings
a1pred_sense = []
a2pred_sense = []
a1pred_lemma = []
a2pred_lemma = []

lemma_categories = []
sense_categories = []
lemma_preds_1 = []
lemma_preds_2 = []
sense_preds_1 = []
sense_preds_2 = []

hc = HomographCoarsenerV1()

for (word, pos) in anno1_datapoints.keys():
    a1d = anno1_datapoints[(word, pos)]
    a2d = anno2_datapoints[(word, pos)]
    ox_senses = set(ox_dict[(word, pos)].keys())
    ox_senses.add('')
    ox_lemmas = set(hc.coarsen_homographs(list({entry['coarse_lemma_id'] for entry in ox_dict[(word, pos)].values()})))
    ox_lemmas.add('')

    lemma1s = []
    lemma2s = []
    sense1s = []
    sense2s = []

    for wn_id, (a1_ox_id, a1_lem_id) in a1d.items():
        (a2_ox_id, a2_lem_id) = a2d[wn_id]

        lemma1s.append(a1_lem_id)
        lemma2s.append(a2_lem_id)
        sense1s.append(a1_ox_id)
        sense2s.append(a2_ox_id)

        # Get lemma datapoints
        for lem_id in ox_lemmas:
            if a1_lem_id == lem_id:
                a1pred_lemma.append(1)
            else:
                a1pred_lemma.append(0)

            if a2_lem_id == lem_id:
                a2pred_lemma.append(1)
            else:
                a2pred_lemma.append(0)

        # Get ox sense datapoints
        for ox_id in ox_senses:
            if a1_ox_id == ox_id:
                a1pred_sense.append(1)
            else:
                a1pred_sense.append(0)

            if a2_ox_id == ox_id:
                a2pred_sense.append(1)
            else:
                a2pred_sense.append(0)

    lemma_preds_1.append(lemma1s)
    lemma_preds_2.append(lemma2s)
    lemma_categories.append(ox_lemmas)
    sense_preds_1.append(sense1s)
    sense_preds_2.append(sense2s)
    sense_categories.append(ox_senses)

info(f'Sense level kappa: {cohen_kappa_score(a1pred_sense, a2pred_sense)}')
info(f'Lemma level kappa: {cohen_kappa_score(a1pred_lemma, a2pred_lemma)}')

info(f'Lemmas: {sum([len(c)-1 for c in lemma_categories])}')
info(f'Lemmas to choose from: {sum([(len(c)-1)*len(l) for c, l in zip(lemma_categories, lemma_preds_1)])/sum([len(l) for l in lemma_preds_1])}')

full_anno = open_dict_csv(test_alignment_file)
absent = 0
total = len(full_anno)
for entry in full_anno:
    if entry['lemma'] == '' or entry['lemma'] is None:
        absent += 1
info(f'{100*absent/total} percent absent')
