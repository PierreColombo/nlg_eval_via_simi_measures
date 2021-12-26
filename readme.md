header-includes: | \usepackage{tikz,pgfplots} \usepackage{fancyhdr} \pagestyle{fancy} \fancyhead[CO,CE]{This is fancy}
\fancyfoot[CO,CE]{So is this} \fancyfoot[LE,RO]{\thepage}

# NLG evaluation via Statistical Measures of Similarity: BaryScore, DepthScore, InfoLM

Automatic Evaluation Metric described in the papers [BaryScore](https://arxiv.org/abs/2108.12463) (EMNLP 2021)
, [DepthScore](https://arxiv.org/abs/2103.12711) (Submitted), [InfoLM](https://arxiv.org/abs/2112.01589) (AAAI2022)

#### Authors:

* [Pierre Colombo](https://scholar.google.com/citations?user=yPoMt8gAAAAJ&hl=fr)
* [Guillaume Staerman](https://scholar.google.com/citations?user=Zb2ax0wAAAAJ&hl=fr)

### Overview

We start by giving an overview of the proposed metrics.

#### DepthScore

DepthScore is a single layer metric based on pretrained contextualized representation. Similar to BertScore it embeds
both candidate (C: It is freezing this morning) and the reference (R: The weather is cold today) using a single layer of
Bert to obtain discrete probability
measures <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_{.,l}^R">
and  <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_{.,l}^C">. Then a similarity score is
computed using the pseudo metric introduce [here](https://arxiv.org/abs/2103.12711).
<div align="center">
<figure>
    <img style="width:100%" src="images/depthscore.png">
<figcaption>Depth Score</figcaption>
</figure>
</div>

#### BaryScore

<div align="center">
<figure>
    <img style="width:100%" src="images/baryscore.jpeg">
<figcaption>BaryScore (left) vs MoverScore (right)</figcaption>
</figure>
</div>

#### InfoLM

<div align="center">
<figure>
    <img style="width:100%" src="images/infolm.jpeg">
<figcaption>InfoLM</figcaption>
</figure>
</div>


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

Running our metrics can be computationally intensive (because it relies on pretrained models). Therefore, a GPU is
usually necessary. If you don't have access to a GPU, you can use light pretrained representations such as TinyBERT,
DistilBERT.

We provide example inputs under `<metric>.py`. For example for BaryScore

```
metric_call = BaryScoreMetric()

ref = [
        'I like my cakes very much',
        'I hate these cakes!']
hypothesis = ['I like my cakes very much',
                  'I like my cakes very much']

metric_call.prepare_idfs(ref, hypothesis)
final_preds = metric_call.evaluate_batch(ref, hypothesis)
print(final_preds)
```

#### Command Line Interface (CLI)

We provide a command line interface (CLI) of BERTScore as well as a python module. For the CLI, you can use it as
follows:

```
export metric=infolm
export measure_to_use=fisher_rao
CUDA_VISIBLE_DEVICES=0 python score_cli.py --ref="samples/refs.txt" --cand="samples/hyps.txt" --metric_name=${metric} --measure_to_use=${measure_to_use}
 ```

See more options by `python score_cli.py -h`.

#### Practical Tips

* Unlike BERT, RoBERTa uses GPT2-style tokenizer which creates addition " " tokens when there are multiple spaces
  appearing together. It is recommended to remove addition spaces by `sent = re.sub(r' +', ' ', sent)`
  or `sent = re.sub(r'\s+', ' ', sent)`.
* Using inverse document frequency (idf) on the reference sentences to weigh word importance may correlate better with
  human judgment. However, when the set of reference sentences become too small, the idf score would become
  inaccurate/invalid. To use idf, please set `--idf` when using the CLI tool.
* When you are low on GPU memory, consider setting `batch_size` to a low number.

#### Practical Limitation

* Because pretrained representations have learned positional embeddings with max length 512, our scores are undefined
  between sentences longer than 510 (512 after adding \[CLS\] and \[SEP\] tokens)
  . The sentences longer than this will be truncated. Please consider using larger models which can support much longer
  inputs.

#### Acknowledgements