import numpy as np
import csv
import json


def flatten_for_csv(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(-1, 1)      # scalar to (1,1)
    if arr.ndim == 1:
        return arr[:, None]            # 1D to (N,1)
    return arr.reshape(arr.shape[0], -1)  # 2D or more


def dict_to_csv(dictionary,file):
    data = [flatten_for_csv(dictionary[k]) for k in dictionary]
    rows = np.hstack(data)
    writer = csv.writer(file, delimiter=' ')
    file.write("# ")
    writer.writerow(dictionary.keys())
    writer.writerows(rows)

myGroups = {
    'x_i': ['x1', 'y1', 'z1'],
    'x_j': ['x2', 'y2', 'z2'],
    'q_i': ['qx1', 'qy1', 'qz1', 'qw1'],
    'q_j': ['qx2', 'qy2', 'qz2', 'qw2'],
    'F_i': ['f1x', 'f1y', 'f1z'],
    'F_j': ['f2x', 'f2y', 'f2z'],
    'T_i': ['t1x', 't1y', 't1z'],
    'T_j': ['t2x', 't2y', 't2z'],
    't': ['t']
}

def load_grouped_csv(filename, groups=myGroups):
    # Read CSV header and data
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        header = next(reader)
        if header[0].startswith('#'):
            header = header[1:]  # Remove comment char if present
        data = np.array([list(map(float, row)) for row in reader if row and not row[0].startswith('#')])

    # Build dictionary with grouped arrays
    result = {}
    for group_name, cols in groups.items():
        idxs = [header.index(col) for col in cols]
        result[group_name] = data[:, idxs]
    return result

    # Example usage:
    #data_dict = load_grouped_csv('test_results.out')
    # Now data_dict['pos1'] is an (N,3) array, etc.


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj

def dict_to_json(_dict_, filename):
    json.dump(make_json_serializable(_dict_), open(filename, 'w'))

def json_to_dict(filename):
    def to_ndarray(obj):
        if isinstance(obj, list):
            # Recursively convert lists of lists to arrays
            if obj and isinstance(obj[0], list):
                return np.array(obj)
            # 1D arrays
            elif obj and isinstance(obj[0], (int, float)):
                return np.array(obj)
            else:
                return [to_ndarray(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_ndarray(v) for k, v in obj.items()}
        else:
            return obj

    with open(filename, 'r') as f:
        data = json.load(f)
    return to_ndarray(data)