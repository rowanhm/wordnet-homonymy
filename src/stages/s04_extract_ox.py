# extract every word in the ox_raw folder
import os
from collections import defaultdict

from src.common import open_pickle, save_pickle, info, warn
from src.global_variables import ox_download_dir, ox_processed_file, ox_dictionary_file, ox_lemma_info_file, \
    test_data_file
from src.homograph_coarsener_v1 import HomographCoarsenerV1

processed = open_pickle(ox_processed_file)

ox_pos_reverse = {
    'NN': 'noun',
    'NNS': 'noun',
    'NNP': 'noun',
    'NNPS': 'noun',
    'VB': 'verb',
    'VBD': 'verb',
    'VBG': 'verb',
    'VBN': 'verb',
    'VBP': 'verb',
    'VBZ': 'verb',
    'RB': 'adv',
    'RBR': 'adv',
    'RBS': 'adv',
    'JJ': 'adj',
    'JJR': 'adj',
    'JJS': 'adj'
}

definitions = defaultdict(dict)
lemma_info = defaultdict(dict)
encountered_entries = set()

for word_number, word in enumerate(processed):

    if word_number % 100 == 0:
        info(f'On word {word_number}/{len(processed)}')

    file_name = ox_download_dir + f'words/{word[0]}/{word}.pkl'
    if not os.path.exists(file_name):
        continue

    parsed_data = open_pickle(file_name)
    ids = {datapoint['id'] for datapoint in parsed_data['data']}
    id_to_sense = {}
    for id in ids:
        path = ox_download_dir + f'entries/{id[0].lower()}/{id}.pkl'
        id_to_sense[id] = open_pickle(path)

    for entry in parsed_data['data']:  # Each lemma
        entry_id = entry['id']

        derived_from = {e['target_id'] for e in entry['etymology']['etymons'] if 'target_id' in e.keys() and e['part_of_speech'] != 'SUFFIX'}

        etymologies = entry['etymology']['etymon_language']
        if etymologies == [['Other sources', 'origin uncertain']]:
            etymologies = entry['etymology']['source_language']
        if etymologies == [['English']]:
            etymologies = [['Indo-European', 'Germanic', 'West Germanic', 'English']]

        if entry['etymology']['etymology_type'] == 'acronym':
            derivations = {('acronym', entry_id)}  # Special case to handle acronyms
        elif entry['etymology']['etymology_type'] in {'properName', 'properNameHybrid'}:
            derivations = {('proper_name', entry_id)}  # Special case to handle acronyms
        else:
            derivations = {tuple(e) for e in etymologies}

        pronunciations = entry['pronunciations']

        combined_data = {
            'full_etymology': entry['etymology'],
            'derivation_chain': derived_from,
            'etymology_lookup': derivations,
            'pronunciation': pronunciations
        }

        if entry_id in lemma_info.keys():
            assert lemma_info[entry_id] == combined_data
        else:
            lemma_info[entry_id] = combined_data

        main_definition = entry['definition']

        poses = {ox_pos_reverse[p] for p in entry['parts_of_speech'] if p in ox_pos_reverse.keys()}
        if len(poses) == 0:
            continue
        elif len(poses) > 1:
            warn(f'Multiple POS ({poses}) for word {word} defined as {main_definition}')

        found_main = False
        added = 0
        for i, sub_entry in enumerate(id_to_sense[entry_id]['data']):
            added += 1
            start = sub_entry['daterange']['start']
            end = sub_entry['daterange']['end']
            categories = sub_entry['categories']['topic']
            definition = sub_entry['definition']
            main = False
            if definition == main_definition:
                main = True
                found_main = True

            for pos in poses:
                if definition is not None and definition != '':

                    sub_entry_id = sub_entry['id'] + ':' + word + ':' + pos

                    assert sub_entry_id not in encountered_entries
                    encountered_entries.add(sub_entry_id)

                    definitions[(word, pos)][sub_entry_id] = {
                        'id': sub_entry_id,
                        'coarse_lemma_id': entry_id,
                        'definition': definition,  # if definition is not None else '',
                        'pos': pos,
                        'start': start,
                        'end': end,
                        'categories': categories,
                        'main': main,
                    }

        if added > 0:
            if not found_main:
                warn(f'Missing main for word {word}')
                # NB this could break if the main one has an excluded POS

info('Adding v1 homograph cluster annotation to test items, used in anno')
definitions_new = definitions.copy()
test_items = set(open_pickle(test_data_file).keys())
hc = HomographCoarsenerV1()

for (word, pos) in test_items:
    entries_dict = definitions[(word, pos)]
    sub_entry_ids = list(entries_dict.keys())
    sub_entry_lemmas = [entries_dict[sub_entry_id]['coarse_lemma_id'] for sub_entry_id in sub_entry_ids]
    sub_entry_homographs = hc.coarsen_homographs(sub_entry_lemmas)
    for (sub_entry_id, sub_entry_homograph) in zip(sub_entry_ids, sub_entry_homographs):
        entries_dict[sub_entry_id]['homograph_cluster_v1'] = sub_entry_homograph

    # Update the entries
    definitions_new[(word, pos)] = entries_dict

info('Saving')
save_pickle(ox_dictionary_file, definitions_new)
save_pickle(ox_lemma_info_file, lemma_info)
hc.save()
