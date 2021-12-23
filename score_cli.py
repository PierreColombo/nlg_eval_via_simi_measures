#!/usr/bin/env python
import os
import argparse
import torch

from bary_score import BaryScoreMetric
from depth_score import DepthScoreMetric
from infoscores import InfoScore
from tqdm import tqdm
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("Calculate BERTScore")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text',
    )
    parser.add_argument(
        "-m", "--model", default=None, help="BERT model name (default: bert-base-uncased) or path to a pretrain model",
    )
    parser.add_argument("-l", "--num_layers", type=int, default=None, help="use first N layer in BERT (default: 8)")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--nthreads", type=int, default=4, help="number of cpu workers (default: 4)")
    parser.add_argument("--idf", action="store_true", help="BERT Score with IDF scaling")

    parser.add_argument("--baseline_path", default=None, type=str, help="path of custom baseline csv file")
    parser.add_argument("--use_fast_tokenizer", action="store_false", help="whether to use HF fast tokenizer")
    parser.add_argument("-s", "--seg_level", action="store_true", help="show individual score of each pair")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-r", "--ref", type=str, nargs="+", required=True, help="reference file path(s) or a string")
    parser.add_argument(
        "-c", "--cand", type=str, required=True, help="candidate (system outputs) file path or a string",
    )

    args = parser.parse_args()

    if os.path.isfile(args.cand):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        refs = []
        for ref_file in args.ref:
            assert os.path.exists(ref_file), f"reference file {ref_file} doesn't exist"
            with open(ref_file) as f:
                curr_ref = [line.strip() for line in f]
                assert len(curr_ref) == len(cands), f"# of sentences in {ref_file} doesn't match the # of candidates"
                refs.append(curr_ref)
        refs = list(zip(*refs))
    elif os.path.isfile(args.ref[0]):
        assert os.path.exists(args.cand), f"candidate file {args.cand} doesn't exist"
    else:
        cands = [args.cand]
        refs = [args.ref]
        assert not args.idf, "do not support idf mode for a single pair of sentences"

    # do for loops :)
    if args.metric_name == 'infoscore':
        metric = InfoScore(model_name=args.model, temperature=0.25, measure_to_use='fisher_rao',
                           use_idf_weights=True, alpha=None, beta=None)
    elif args.metric_name == 'depthscore':
        metric = DepthScoreMetric(model_name=args.model, layers_to_consider=9, considered_measure='irw', p=None,
                                  eps=None, n_alpha=None)
    elif args.metric_name == 'baryscore':
        metric = BaryScoreMetric(model_name=args.model, last_layers=5, use_idfs=True, sinkhorn_ref=0.01)
    else:
        raise NotImplementedError
    all_preds = []
    idf_hyps, idf_ref = metric.prepare_idfs(all_candidates, all_hypothesis)
    for golden_batch, candidate_batch in tqdm(zip(), 'Evaluation in Progress'):
        preds = metric.evaluate_batch(candidate_batch, golden_batch, idf_hyps=idf_hyps, idf_ref=idf_ref)
        all_preds.append(preds)

    for k in preds.keys():
        preds = []
        for pred in all_preds:
            preds.append(pred)
        logging.info('Metric {} : {}'.format(k, sum(preds) / len(preds)))


if __name__ == "__main__":
    main()
