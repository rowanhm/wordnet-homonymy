import csv
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_credentials():
    from src.global_variables import credentials_file
    credentials = open_dict_csv(credentials_file)[0]
    return credentials['app_id'], credentials['key']


def flatten(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list


def info(text):
    logging.info(text)


def warn(text):
    logging.warning(text)


def open_dict_csv(file, delimiter=None):
    ftype = file[-4:]
    if delimiter is None:
        if ftype == '.tsv':
            delimiter = '\t'
        elif ftype == '.csv':
            delimiter = ','
        else:
            # update
            print("Invalid file extension: {}".format(file))
            exit()
    all_lines = []
    with open(file, 'r') as csv_file:
        for line in csv.DictReader(csv_file, delimiter=delimiter):
            all_lines += [line]
    return all_lines


def save_csv(file, all_lines):
    ftype = file[-4:]
    delimiter = ''
    if ftype == '.tsv':
        delimiter = '\t'
    elif ftype == '.csv':
        delimiter = ','
    else:
        # update
        print("Invalid file extension: {}".format(file))
        exit()
    with open(file, 'w') as csv_file:
        dict_writer = csv.DictWriter(csv_file, fieldnames=all_lines[0].keys(), delimiter=delimiter,
                                     quoting=csv.QUOTE_NONE, escapechar='\\')
        dict_writer.writeheader()
        dict_writer.writerows(all_lines)
    return


def open_pickle(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_pickle(file, data):
    with open(file, 'wb') as fp:
        pickle.dump(data, fp)
