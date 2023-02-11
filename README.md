# [ocpdet](https://pypi.org/project/ocpdet/)
OCPDet is a Python package for online changepoint detection, implementing state-of-the-art algorithms and a novel approach.

[![PyPI version](https://badge.fury.io/py/ocpdet.svg)](https://badge.fury.io/py/ocpdet) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7232039.svg)](https://doi.org/10.5281/zenodo.7232039)

A Python package for online changepoint detection in univariate and multivariate data.

Algorithms implemented in ocpdet are

- **CUSUM**: Cumulative Sum algorithm, proposed by Page (1954)
- **EWMA**: Exponentially Weighted Moving Average algorithm, proposed by Roberts (1959)
- **Two Sample tests**: Nonparametric hypothesis testing for changepoint detection, proposed by Ross et al. (2011)
- **Neural Networks**: Novel approach based on sequentially learning neural networks, proposed by Hushchyn et al. (2020) and extended to online context (Master's Thesis) 

## Installation

```bash
pip install ocpdet
``` 

## Examples

- [CUSUM.ipynb](https://nbviewer.jupyter.org/github/vkhamesi/ocpdet/tree/main/docs/CUSUM.ipynb)  
- [EWMA.ipynb](https://nbviewer.jupyter.org/github/vkhamesi/ocpdet/tree/main/docs/EWMA.ipynb)  
- [TwoSample.ipynb](https://nbviewer.jupyter.org/github/vkhamesi/ocpdet/tree/main/docs/TwoSample.ipynb)  
- [NeuralNetwork.ipynb](https://nbviewer.jupyter.org/github/vkhamesi/ocpdet/tree/main/docs/NeuralNetwork.ipynb)  

## How to cite this work

Here is a suggestion to cite this GitHub repository:

> Victor Khamesi. (2022). ocpdet: A Python package for online changepoint detection in univariate and multivariate data. (Version v0.0.4). Zenodo. https://doi.org/10.5281/zenodo.7232039

And a possible BibTeX entry:

```tex
@software{victor_khamesi_2022,
  author       = {Victor Khamesi},
  title        = {ocpdet: A Python package for online changepoint detection in univariate and multivariate data.},
  month        = oct,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.0.5},
  doi          = {10.5281/zenodo.7232039},
  url          = {https://doi.org/10.5281/zenodo.7232039}
}
```

## License

The non-software content of this project is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/), and the software code is licensed under the [BSD-2 Clause license](https://opensource.org/licenses/BSD-2-Clause).