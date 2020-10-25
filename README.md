[![DOI](https://zenodo.org/badge/306959618.svg)](https://zenodo.org/badge/latestdoi/306959618)
# untangling-visual-search

Experiments to test whether the untangling mechanism proposed for object recognition 
can also account for behavior measured in visual search tasks, using 
deep neural network models of the primate ventral visual stream

## Installation
Experiments were run in an environment created with [`conda`](https://docs.conda.io/en/latest/) on Ubuntu 16.04.

There are two main dependencies:  
* the [`visual-search-nets` package](https://github.com/NickleDave/visual-search-nets)
* [`jupyterlab`](http://jupyterlab.io/)

A similar environment can be created with `conda` on Ubuntu using the `spec-file.txt` in this repository as follows:

```console
$ git clone https://github.com/NickleDave/untangling-visual-search.git
$ cd untangling-visual-search
$ conda create --name untangling-search --file spec-file.txt
```

You may also be able to create a suitable environment on other linux platforms using the `environment.yml` file.

```console
$ git clone https://github.com/NickleDave/untangling-visual-search.git
$ cd untangling-visual-search
$ conda env create -f environment.yml

```

## Acknowledgements
- Research funded by the Lifelong Learning Machines program, 
DARPA/Microsystems Technology Office, 
DARPA cooperative agreement HR0011-18-2-0019
- David Nicholson was partially supported by the 
2017 William K. and Katherine W. Estes Fund to F. Pestilli, 
R. Goldstone and L. Smith, Indiana University Bloomington.

## Citation
If you use / adapt this code, please cite its DOI:
[![DOI](https://zenodo.org/badge/306959618.svg)](https://zenodo.org/badge/latestdoi/306959618)
