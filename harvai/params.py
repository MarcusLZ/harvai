import os


def get_path_data(path):
    if os.path.basename(path) == 'harvai':
        return "raw_data/LEGITEXT000006074228.pdf"
    else:
        return "../raw_data/LEGITEXT000006074228.pdf"
