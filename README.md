# GRAPHBPE: Molecular Graphs Meet Byte-Pair Encoding [[paper]](https://openreview.net/forum?id=tUEug4t6kG)

# Quick reviews for GraphBPE
*Research question*: 

Can (2D) graphs be tokenized similarly to that of words?

*Motivation*: 
1. words can be viewed as (undirected) line graphs (e.g., "low" as "l"-"o"-"w"), so there could potentially be a graph tokenization algorithm whose special case (e.g., if we convert all words into line graphs) would be the tokenization algorithm (e.g., BPE) for words.
2. by simply contracting rings in the molecular graphs, we can achieve performance gains across different GNNs and datasets (see Table 2 on Page 3 of [our paper](https://openreview.net/forum?id=tUEug4t6kG))

*Result & Quick conclusion*: 

GraphBPE is effective for small classification datasets, given a fixed number of tokenization steps (e.g., 100), and the performance might further be improved if we customize the contextualizer for different dataset. For detailed results, please check out [our paper](https://openreview.net/forum?id=tUEug4t6kG)!

*Potential impact & Future direction*:
1. GraphBPE can increase the complexity of node vocabulary (e.g., w/o tokenization, the maximum possible node vocabulary will be 118 for molecular graphs), which might be beneficial to pre-training (see `Substructures for molecular machine learning` in the **Related Work** section of [our paper](https://openreview.net/forum?id=tUEug4t6kG)).
2. By customizing the contextualizer of GraphBPE, we can derive new tokenization algorithms / graph grammars in a non-parametric fashion for novel molecule generation.
# Dependencies
you can find the packages needed in `requirements.txt`, for the dependencies of `h2g`, please check the official 
`hgraph2graph` implementation [[here]](https://github.com/wengong-jin/hgraph2graph)
# Step 1. Process dataset
to process dataset, run ``python graph_tokenizer.py dataset=xxx `` (if it is run the first time, it will first download all the datasets used in the paper [i.e., specified in `supported.py`], and then tokenize the specified dataset `xxx`)
# Step 2. Run experiment
to run experiment on GNNs, and to run with different configurations, you can use the `run_gnn_baselines.sh` and `run_gnns.sh` under `scripts`:
- for GNN baselines, run ``python train_GNN.py dataset=xxx model=xxx train=train``
- for GNN wih GraphBPE, run ``python train_GNN_wBPE.py dataset=xxx model=xxx train=train``

to run experiment on HGNNs (for hyper-graphs), and to run with different configurations, you can use the `run_tokenization_baselines.sh` and `run_hgnns.sh` under `scripts`:
- for HGNN baselines, run ``python train_HGNN.py dataset=xxx model=xxx train=train``
- for HGNN with GraphBPE, run ``python train_HGNN_wBPE.py dataset=xxx model=xxx train=train``

you can always change the configuration following the tutorials at [Hydra](https://hydra.cc/docs/intro/)

# Citation
If you find our work interesting / useful, please consider citing our paper, thank you!
```
@inproceedings{
shen2024graphbpe,
title={Graph{BPE}: Molecular Graphs Meet Byte-Pair Encoding},
author={Yuchen Shen and Barnabas Poczos},
booktitle={ICML 2024 AI for Science Workshop},
year={2024},
url={https://openreview.net/forum?id=tUEug4t6kG}
}
```
