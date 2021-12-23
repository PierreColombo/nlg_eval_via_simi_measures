# NLG evaluation via Statistical Measures of Similarity: BaryScore, DepthScore, InfoLM

Automatic Evaluation Metric described in the
paper [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) (ICLR 2020). We now support
about 130 models (see
this [spreadsheet](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?usp=sharing)
for their correlations with human evaluation). Currently, the best model is `microsoft/deberta-xlarge-mnli`, please
consider using it instead of the default `roberta-large` in order to have the best correlation with human evaluation.

#### Authors:

* [Pierre Colombo](https://scholar.google.com/citations?user=yPoMt8gAAAAJ&hl=fr)
* [Guillaume Staerman](https://scholar.google.com/citations?user=Zb2ax0wAAAAJ&hl=fr)
* [Chloé Clavel](https://scholar.google.fr/citations?user=TAZbfksAAAAJ&hl=en)
* [Pablo Piantanida](https://scholar.google.com/citations?user=QyBEFv0AAAAJ&hl=fr)

### Overview

BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference
sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and system-level
evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different
language generation tasks.

If you find this repo useful, please cite our papers:

```
@article{infolm_aaai2022,
  title={InfoLM: A New Metric to Evaluate Summarization \& Data2Text Generation},
  author={Colombo, Pierre and Clavel, Chloe and Piantanida, Pablo},
  journal={arXiv preprint arXiv:2112.01589},
  year={2021}
}
```

```
@inproceedings{colombo-etal-2021-automatic,
    title = "Automatic Text Evaluation through the Lens of {W}asserstein Barycenters",
    author = "Colombo, Pierre  and
      Staerman, Guillaume  and
      Clavel, Chlo{\'e}  and
      Piantanida, Pablo",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.817",
    pages = "10450--10466",
    abstract = "A new metric BaryScore to evaluate text generation based on deep contextualized embeddings (\textit{e.g.}, BERT, Roberta, ELMo) is introduced. This metric is motivated by a new framework relying on optimal transport tools, \textit{i.e.}, Wasserstein distance and barycenter. By modelling the layer output of deep contextualized embeddings as a probability distribution rather than by a vector embedding; this framework provides a natural way to aggregate the different outputs through the Wasserstein space topology. In addition, it provides theoretical grounds to our metric and offers an alternative to available solutions (\textit{e.g.}, MoverScore and BertScore). Numerical evaluation is performed on four different tasks: machine translation, summarization, data2text generation and image captioning. Our results show that BaryScore outperforms other BERT based metrics and exhibits more consistent behaviour in particular for text summarization.",
}
```

```
@article{depth_score,
  title={Depth-based pseudo-metrics between probability distributions},
  author={Staerman, Guillaume and Mozharovskyi, Pavlo and Cl{\'e}men{\c{c}}on, St{\'e}phan and d'Alch{\'e}-Buc, Florence},
  journal={arXiv preprint arXiv:2103.12711},
  year={2021}
}
```

### Usage

#### Python Function

On a high level, we provide a python function `bert_score.score` and a python object `bert_score.BERTScorer`. The
function provides all the supported features while the scorer object caches the BERT model to faciliate multiple
evaluations. Check our [demo](./example/Demo.ipynb) to see how to use these two interfaces. Please refer
to [`bert_score/score.py`](./bert_score/score.py) for implementation details.

Running our metrics can be computationally intensive (because it relies on pretrained models). Therefore, a GPU is
usually necessary. If you don't have access to a GPU, you can use light pretrained representations such as: [tinybert]

#### Command Line Interface (CLI)

We provide a command line interface (CLI) of BERTScore as well as a python module. For the CLI, you can use it as
follows:

1. To evaluate English text files:

We provide example inputs under `./example`.

```sh
bert-score -r example/refs.txt -c example/hyps.txt --lang en
```

You will get the following output at the end:

roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0) P: 0.957378 R: 0.961325 F1: 0.959333

where "roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0)" is the hash code.

Starting from version 0.3.0, we support rescaling the scores with baseline scores

```sh
bert-score -r example/refs.txt -c example/hyps.txt --lang en --rescale_with_baseline
```

2. To evaluate text files in other languages:

We currently support the 104 languages in multilingual
BERT ([full list](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)).

Please specify the two-letter abbreviation of the language. For instance, using `--lang zh` for Chinese text.

See more options by `bert-score -h`.

3. To load your own custom model:
   Please specify the path to the model and the number of layers to use by `--model` and `--num_layers`.

```sh
bert-score -r example/refs.txt -c example/hyps.txt --model path_to_my_bert --num_layers 9
```

#### Practical Tips

* Unlike BERT, RoBERTa uses GPT2-style tokenizer which creates addition " " tokens when there are multiple spaces
  appearing together. It is recommended to remove addition spaces by `sent = re.sub(r' +', ' ', sent)`
  or `sent = re.sub(r'\s+', ' ', sent)`.
* Using inverse document frequency (idf) on the reference sentences to weigh word importance may correlate better with
  human judgment. However, when the set of reference sentences become too small, the idf score would become
  inaccurate/invalid. To use idf, please set `--idf` when using the CLI tool or
  `idf=True` when calling `bert_score.score` function.
* When you are low on GPU memory, consider setting `batch_size` when calling
  `bert_score.score` function.
* To use a particular model please set `-m MODEL_TYPE` when using the CLI tool or `model_type=MODEL_TYPE` when
  calling `bert_score.score` function.
* We tune layer to use based on WMT16 metric evaluation dataset. You may use a different layer by setting `-l LAYER`
  or `num_layers=LAYER`. To tune the best layer for your custom model, please follow the instructions
  in [tune_layers](tune_layers) folder.
* __Limitation__: Because pretrained representations have learned positional embeddings with
  max length 512, our scores are undefined between sentences longer than 510 (512 after adding \[CLS\] and \[SEP\] tokens)
  . The sentences longer than this will be truncated. Please consider using larger models which can support much longer inputs.
