# Input with a set of lemmas (dicts), coarsens their IDs
import itertools
from collections import defaultdict

import requests

from src.common import warn, open_pickle, get_credentials, save_pickle, info
from src.global_variables import ox_lemma_info_file

app_id, ox_key = get_credentials()


class HomographCoarsenerV1:

    def __init__(self):
        self.lemma_info = open_pickle(ox_lemma_info_file)
        self.updated_lemmas = False

    def coarsen_homographs(self, lemmas_codes, warn_cycles=False):

        derivation_dict = {}
        all_full_etymologies = defaultdict(list)

        for course_id in set(lemmas_codes):
            etymology = set()
            queue = [course_id]
            seen = set()
            add_all_seen = False
            while len(queue) > 0:
                next_step_id = queue.pop()

                skip_addition = False
                # Check for cycles
                if next_step_id in seen:
                    if warn_cycles:
                        warn(f'Cycle detected for {course_id}')
                    skip_addition = True
                    add_all_seen = True
                else:
                    seen.add(next_step_id)

                if next_step_id not in self.lemma_info.keys():
                    # Find it from online if it isn't local
                    parsed_sub_data = requests.get(
                        f'https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2/word/{next_step_id}',
                        headers={"app_id": app_id, "app_key": ox_key}).json()
                    instance_derivations_chain = {e['target_id'] for e in
                                                  parsed_sub_data['data']['etymology']['etymons'] if
                                                  'target_id' in e.keys() and e['part_of_speech'] != 'SUFFIX'}
                    instance_etymology_lookup = {tuple(e) for e in
                                                 parsed_sub_data['data']['etymology']['etymon_language']}
                    instance_full_etymology = parsed_sub_data['data']['etymology']
                    instance_pronunciations = parsed_sub_data['data']['pronunciations']

                    # Add to lemma_dict and save
                    self.lemma_info[next_step_id] = {
                        'full_etymology': instance_full_etymology,
                        'derivation_chain': instance_derivations_chain,
                        'etymology_lookup': instance_etymology_lookup,
                        'pronunciation': instance_pronunciations
                    }

                    self.updated_lemmas = True

                next_step_is_derived_from = self.lemma_info[next_step_id]['derivation_chain']

                if len(next_step_is_derived_from) == 0:
                    for et in self.lemma_info[next_step_id]['etymology_lookup']:
                        etymology.add(et)
                    all_full_etymologies[course_id].append(self.lemma_info[next_step_id]['full_etymology'])
                else:
                    if not skip_addition:
                        queue.extend(list(next_step_is_derived_from))

            # If there is a cycle with no bottoming out, use all their etymologies
            if add_all_seen:
                for seen_id in seen:
                    for et in self.lemma_info[seen_id]['etymology_lookup']:
                        etymology.add(et)
                    all_full_etymologies[course_id].append(self.lemma_info[seen_id]['full_etymology'])

            derivation_dict[course_id] = etymology

        # info('Remove senses without etymology')
        skip_etymologies = set()
        for coarse_id, etymologies in all_full_etymologies.items():
            skip = True
            for etymology in etymologies:
                # Skip this entry if its etymology is unknown
                if etymology['etymology_type'] != 'unknown' and \
                        etymology['etymon_language'] != [['undetermined']] and \
                        (not (etymology['etymon_language'] == [['Other sources', 'origin uncertain']] and etymology[
                            'source_language'] == [])):
                    skip = False
                    break
            if skip:
                skip_etymologies.add(coarse_id)

        # info('Combining for items')
        homograph_dict = {}
        current_key = 0
        index_mapping = {}  # cluster index -> derivations it contains
        homograph_mapping = {}  # cluster index -> ids contained

        for coarse_id in set(lemmas_codes):

            # Exclude ones without etmyology
            if coarse_id in skip_etymologies:
                homograph_dict[coarse_id] = 'exclude'
                continue

            set_of_derivations = derivation_dict[coarse_id]
            found = False
            for key in list(index_mapping.keys()):
                keys_derivations = index_mapping[key]
                if len(keys_derivations.intersection(set_of_derivations)) != 0:
                    found = True
                    # add all derivations it has
                    index_mapping[key] = keys_derivations.union(set_of_derivations)
                    homograph_mapping[key].add(coarse_id)
                    break
            if not found:
                # Add a new set
                key = current_key
                index_mapping[key] = set_of_derivations
                homograph_mapping[key] = {coarse_id}
                current_key += 1
            else:
                # Check consistency
                unstable = True
                while unstable:
                    unstable = False
                    for key1, key2 in itertools.combinations(list(index_mapping.keys()), 2):
                        assert key1 != key2
                        derivation_set_1 = index_mapping[key1]
                        derivation_set_2 = index_mapping[key2]
                        if len(derivation_set_1.intersection(derivation_set_2)) != 0:
                            # Overlap - combine into a single set
                            unstable = True
                            del index_mapping[key2]
                            assert key2 not in index_mapping.keys()
                            index_mapping[key1] = derivation_set_1.union(derivation_set_2)
                            key_2_ids = homograph_mapping[key2]
                            del homograph_mapping[key2]
                            assert key2 not in homograph_mapping.keys()
                            for key_2_id in key_2_ids:
                                homograph_mapping[key1].add(key_2_id)
                            break

        # Make sure one does not subclass another; if so, merge
        unstable = True
        while unstable:
            unstable = False
            for key1, key2 in itertools.combinations(list(index_mapping.keys()), 2):
                assert key1 != key2
                derivation_set_1 = index_mapping[key1]
                derivation_set_2 = index_mapping[key2]
                for derivation_chain_1 in derivation_set_1:
                    assert len(derivation_chain_1) > 0

                    for derivation_chain_2 in derivation_set_2:
                        assert len(derivation_chain_2) > 0

                        if derivation_chain_1 == derivation_chain_2[:len(derivation_chain_1)] or \
                                derivation_chain_2 == derivation_chain_1[:len(derivation_chain_2)]:
                            unstable = True
                            break

                    if unstable:
                        break

                # Merge
                if unstable:
                    del index_mapping[key2]
                    assert key2 not in index_mapping.keys()
                    index_mapping[key1] = derivation_set_1.union(derivation_set_2)
                    key_2_ids = homograph_mapping[key2]
                    del homograph_mapping[key2]
                    assert key2 not in homograph_mapping.keys()
                    for key_2_id in key_2_ids:
                        homograph_mapping[key1].add(key_2_id)
                    break

        # Reformat
        for i, coarse_ids_cluster in enumerate(homograph_mapping.values()):
            for coarse_id in coarse_ids_cluster:
                code = coarse_id
                assert code not in homograph_dict.keys()  # , 'Encountered same coarse_id twice?'
                homograph_dict[code] = str(i)


        homograph_clusters = [homograph_dict[lemma] for lemma in lemmas_codes]
        if 'exclude' in homograph_clusters:
            # If one is excluded, exclude all
            homograph_clusters = ['exclude'] * len(homograph_clusters)
        else:
            # Rename clusters
            cluster_ids_to_lemmas = defaultdict(set)
            for lemma, cluster_id in zip(lemmas_codes, homograph_clusters):
                cluster_ids_to_lemmas[cluster_id].add(lemma)
            cluster_ids_to_names = {cluster_id: '/'.join(sorted(list(lemmas))) for cluster_id, lemmas in
                                    cluster_ids_to_lemmas.items()}
            homograph_clusters = [cluster_ids_to_names[cluster_id] for cluster_id in homograph_clusters]

        return homograph_clusters

    def save(self):
        if self.updated_lemmas:
            save_pickle(ox_lemma_info_file, self.lemma_info)
        else:
            info('No new lemmas downloaded')


# For debug
if __name__ == "__main__":
    hc = HomographCoarsenerV1()
    # ruler = open_pickle('data/ox_raw/words/r/ruler.pkl')
    # ruler = open_pickle('data/ox_raw/entries/r/ruler_nn01.pkl')
    print(hc.coarsen_homographs(['ruler_nn01', 'ruler_nn02']))