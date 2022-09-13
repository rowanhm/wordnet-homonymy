from src.common import open_pickle, save_csv
from src.global_variables import within_pos_pkl_file, between_pos_pkl_file, within_pos_csv_file, between_pos_csv_file, raw_pkl_file, raw_csv_file

for load_file, save_file in [(within_pos_pkl_file, within_pos_csv_file), (between_pos_pkl_file, between_pos_csv_file), (raw_pkl_file, raw_csv_file)]:
    cluster_dict = open_pickle(load_file)
    output = []
    for lemma, cluster in cluster_dict.items():
        output.append({
            'wn_sense': lemma,
            'lemma': cluster
        })
    save_csv(save_file, output)
