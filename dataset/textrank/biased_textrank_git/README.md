# Biased TextRank
This repository contains code and data for our paper: 
**[Biased textrank: Unsupervised Graph-Based Content Extraction: Ashkan Kazemi, Veŕonica Pérez-Rosas, and Rada Mihalcea. COLING 2020](https://www.aclweb.org/anthology/2020.coling-main.144/)**.

Biased TextRank is an unsupervised, graph-based method for extracting content from text with a given focus. In this repository,
you can find code for two experiments described in our paper; 1) focused summarization of US presidential debates and 2)
supporting explanation extraction for fact-checking of political claims. 

### Requirements
To install the required packages for running the codes on your machine, please run ``pip install -r requirements.txt``
first. 

### Content
* ``/data/``: This directory contains the two datasets used in the experiments. The ``/data/liar/`` directory contains files
for the LIAR-PLUS dataset. The ``/data/us-presidential-debates/``  directory contains the novel presidential debates 
dataset described in the paper.
* ``/src/``: This directory contains implementations of the described experiments in the paper. To run the *biased summarization*
experiment, run ``/src/biased_summarization.py``. For the *explanation extraction* experiment, run 
``/src/explanation_generation.py``. 

### Citation
If you plan to use our methods or data, please cite our work using the following bibtex:

```
@inproceedings{kazemi-etal-2020-biased,
    title = "Biased {T}ext{R}ank: Unsupervised Graph-Based Content Extraction",
    author = "Kazemi, Ashkan  and
      P{\'e}rez-Rosas, Ver{\'o}nica  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.144",
    pages = "1642--1652",
    abstract = "We introduce Biased TextRank, a graph-based content extraction method inspired by the popular TextRank algorithm that ranks text spans according to their importance for language processing tasks and according to their relevance to an input {``}focus.{''} Biased TextRank enables focused content extraction for text by modifying the random restarts in the execution of TextRank. The random restart probabilities are assigned based on the relevance of the graph nodes to the focus of the task. We present two applications of Biased TextRank: focused summarization and explanation extraction, and show that our algorithm leads to improved performance on two different datasets by significant ROUGE-N score margins. Much like its predecessor, Biased TextRank is unsupervised, easy to implement and orders of magnitude faster and lighter than current state-of-the-art Natural Language Processing methods for similar tasks.",
}
```
