import argparse
import json

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    POS_WORDS = ['Yes', 'yes', 'YES']

    assert len(out) == 1
    for line in out:
        line = line.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        elif any(word in POS_WORDS for word in words):
            pred_list.append(1)
        else:
            continue

    return pred_list

def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t\n")
    print("{}\t{}\t{}\t{}\n".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    print("Accuracy: {}\n".format(acc))
    print("Precision: {}\n".format(precision))
    print("Recall: {}\n".format(recall))
    print("F1 score: {}\n".format(f1))
    print("Yes ratio: {}\n".format(yes_ratio))


parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
parser.add_argument("--ans-file", type=str, help="answer file")
args = parser.parse_known_args()[0]

lines = open(args.ans_file).read().split("\n")

pred_list = []
label_list = []

for line in lines:
    if len(line) == 0:
        break

    line = json.loads(line)

    len_before = len(pred_list)
    pred_list = recorder([line["ans"]], pred_list)
    len_after = len(pred_list)
    if len_before == len_after:
        continue

    if isinstance(line["label"], int):
        label_list += [line["label"]]
    else:
        label_list = recorder([line["label"]], label_list)

assert len(pred_list) == len(label_list)

yes_ratio = pred_list.count(1) / len(pred_list)

print_acc(pred_list, label_list)
