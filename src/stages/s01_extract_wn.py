# Extract wn definitions
from collections import defaultdict
from nltk.corpus import wordnet as wn

from src.common import save_pickle, info, open_pickle
from src.global_variables import wn_dictionary_file

pos_lookup = {
    'n': 'noun',
    'v': 'verb',
    'a': 'adj',
    's': 'adj',
    'r': 'adv'
}

info('Loading definitions')
extracted_definitions = open_pickle('data/definitions.pkl')

info('Generating dictionary')
wn_dict = defaultdict(dict)
word_pos_options = defaultdict(set)
all_lemma_ids = set()

for synset in wn.all_synsets():

    # Attempt to filter Proper Nouns and phrases
    lemmas = {l for l in synset.lemmas() if (l.name().lower() == l.name() and '_' not in l.name())}
    if len(lemmas) == 0:
        continue

    for lemma in lemmas:
        sense_id = lemma.key()
        lemma_name = lemma.name()
        pos = pos_lookup[synset.pos()]

        assert sense_id not in all_lemma_ids
        all_lemma_ids.add(lemma_name)

        wn_dict[(lemma_name, pos)][sense_id] = {
            'definition': extracted_definitions[synset.name()],
            'synonyms': {l.name() for l in lemmas} - {lemma_name},
            'id': sense_id,
            'pos': pos
        }

        word_pos_options[lemma_name].add(pos)

info('Filtering monosemous words')
overall_num_senses = 0
filtered_num_senses = 0
for lemma, pos_options in word_pos_options.items():
    assert len(pos_options) > 0
    num_senses = 0
    for pos in pos_options:
        pos_senses = len(wn_dict[(lemma, pos)].values())
        assert pos_senses > 0
        num_senses += pos_senses

    overall_num_senses += num_senses

    if num_senses == 1:  # Monosemous
        filtered_num_senses += 1
        for pos in pos_options:
            del wn_dict[(lemma, pos)]

info(f'Filtered {filtered_num_senses}/{overall_num_senses} senses, leaving {overall_num_senses-filtered_num_senses}')

info('Saving')
save_pickle(wn_dictionary_file, wn_dict)

info('Done')
