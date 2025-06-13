# R-GCN-deduplication

This project implements a Relational Graph Convolutional Network (R-GCN) for entity deduplication using RDF data. It aims to identify duplicate nodes based on relational information.

## How to Use
1. Clone the repository.
2. Open the Jupyter notebook:
   R-GCN Deduplication
3. Run the notebook to train and evaluate the R-GCN model.

The notebook will handle data loading, training, and evaluation.

## Datasets
- cora_dup
- kgdl_dup
- countries_dup

These should be placed in the `data/` directory.

## Evaluation Metrics

The model evaluates performance using:
- Hits@1 and Hits@10
- Mean Rank (MR)
- Precision, Recall, and F1 score for the `sameAs` relation
