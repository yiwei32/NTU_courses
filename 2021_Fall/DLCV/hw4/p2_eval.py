import sys, csv
import numpy as np
args = sys.argv

# read your prediction file
with open(args[1], mode='r') as pred:
    reader = csv.reader(pred)
    next(reader, None)  # skip the headers
    pred_dict = {int(rows[0]): rows[2] for rows in reader}

# read ground truth data
with open(args[2], mode='r') as gt:
    reader = csv.reader(gt)
    next(reader, None)  # skip the headers
    gt_dict = {int(rows[0]) : rows[2] for rows in reader}

if len(pred_dict) != len(gt_dict):
    sys.exit("Test case length mismatch.")

hit = 0
for key, value in pred_dict.items():
    if key not in gt_dict:
        sys.exit("id mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))

    hit += (gt_dict[key] == value)


print('Accuracy: {:.2f} %'.format(hit / len(pred_dict) * 100))