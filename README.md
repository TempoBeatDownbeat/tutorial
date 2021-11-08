# Tempo, Beat and Downbeat Estimation

**By Matthew E. P. Davies, Sebastian Böck and Magdalena Fuentes**

Resources and Jupyter Book for the ISMIR 2021 tutorial on Tempo, Beat and Downbeat estimation.

To jump directly to the hands on part:
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tuOqNyO9gdMmYJsj33fP_QOfpRsm2tmt?usp=sharing)

To go directly to the book:
https://tempobeatdownbeat.github.io/tutorial


### Install

We recommended you do this within a virtual environment:

```bash
conda create -n beat python=3.8
conda activate beat
git clone git@github.com:magdalenafuentes/rhythm_tutorial.git
cd rhythm_tutorial/
pip install -e .
```

### Building the book


```bash
cd rhythm_tutorial/
jupyter-book clean book/  # to remove any existing builds
jupyter-book build book/
```

A fully-rendered HTML version of the book will be built in `book/_build/html/`.



