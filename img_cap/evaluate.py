import argparse
import pandas as pd

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.util import ptb_tokenize
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction_file", type=str)
    parser.add_argument("-r", "--reference_file", type=str)
    parser.add_argument("-o", "--output_file", type=str)
    args = parser.parse_args()
    prediction_df = pd.read_json(args.prediction_file)
    # [n, var_len]
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    # [n, 5, var_len]
    captions = open(args.reference_file, "r").read().strip().split("\n")
    key_to_refs = {}
    for i, row in enumerate(captions):
        row = row.split("\t")
        row[0] = row[0][: len(row[0]) - 2]  # filename#0 caption
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider()]
    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)

    output = {"SPIDEr": 0}
    with open(args.output_file, "w") as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == "Bleu":
                for n in range(4):
                    print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
                scores = scores[3] # 只考虑bleu4
            else:
                print(f"{method}: {score:.3f}", file=writer)
            if method in ["CIDEr", "SPICE"]:
                output["SPIDEr"] += score

            # Find best and worst sample
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)
            best_key = list(key_to_pred.keys())[best_idx]
            worst_key = list(key_to_pred.keys())[worst_idx]
            best_pred = key_to_pred[best_key]
            worst_pred = key_to_pred[worst_key]
            best_ref = key_to_refs[best_key]
            worst_ref = key_to_refs[worst_key]
            best_val = scores[best_idx]
            worst_val = scores[worst_idx]
            print(f"Best sample: {best_key} ({best_val:.3f})", file=writer)
            print(f"Best prediction: {best_pred}", file=writer)
            print(f"Best reference: {best_ref}", file=writer)
            print(f"Worst sample: {worst_key} ({worst_val:.3f})", file=writer)
            print(f"Worst prediction: {worst_pred}", file=writer)
            print(f"Worst reference: {worst_ref}", file=writer)
        output["SPIDEr"] /= 2
        print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)

