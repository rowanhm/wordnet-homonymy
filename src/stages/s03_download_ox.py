# Download the ox data
import os
import time
from datetime import datetime

import requests

from src.common import open_pickle, save_pickle, info, get_credentials
from src.global_variables import per_minute_requests, ox_download_dir, wn_dictionary_file, ox_processed_file

app_id, key = get_credentials()


class Requestor:
    def __init__(self):
        self.num_requests = 0
        self.time = datetime.now()
        self.repeat_time = 61

    def request(self, url):
        if self.num_requests >= per_minute_requests:
            # Max requests reached
            time_now = datetime.now()
            time_diff = (self.time - time_now).total_seconds() + 60
            if time_diff > 0:
                time.sleep(time_diff)
            # Reset
            self.time = datetime.now()
            self.num_requests = 0

        parsed_data = requests.get(url, headers={"app_id": app_id, "app_key": key}).json()
        self.num_requests += 1
        return parsed_data


if not os.path.exists(ox_processed_file):
    save_pickle(ox_processed_file, set())

info('Loading words')
words = {word for (word, pos) in open_pickle(wn_dictionary_file).keys()}
words_processed = open_pickle(ox_processed_file)
words_to_do = (words.difference(words_processed))

info(f'{len(words)} words total ({len(words_processed)} done; {len(words_to_do)} remain)')

requestor = Requestor()
url_base = "https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2"

info(f'Downloading with key {key}')
processed = 0
while len(words_to_do) > 0:

    if processed % 100 == 0 and processed > 0:
        info(f'{processed} words processed...')

    # First get the word
    word = words_to_do.pop(0)
    url = url_base + f"/words/?lemma={word.lower()}"
    parsed_data = requestor.request(url)

    ids = {datapoint['id'] for datapoint in parsed_data['data']}

    if len(ids) > 0:
        # Save the word
        letter = word[0].lower()
        dir = ox_download_dir + 'words/' + letter
        os.makedirs(dir, exist_ok=True)
        save_pickle(dir + f'/{word}.pkl', parsed_data)

        # Now, get and save all entries
        for id in ids:
            dir = ox_download_dir + 'entries/' + id[0].lower()
            path = dir + f'/{id}.pkl'
            if not os.path.exists(path):
                url = url_base + f"/word/{id}/senses/"
                os.makedirs(dir, exist_ok=True)
                parsed_data = requestor.request(url)
                save_pickle(path, parsed_data)

    processed += 1
    words_processed.add(word)
    save_pickle(ox_processed_file, words_processed)

info(f'Done.')
