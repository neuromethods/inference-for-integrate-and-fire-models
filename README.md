# inference-for-integrate-and-fire-models

Implementations of the inference methods for neuronal integrate-and-fire circuit models described in: __Ladenbauer and Ostojic,__ ___Inferring mechanistic models of neuronal microcircuits from single-trial spike trains___ (submitted for peer review 2018)

The code contains examples for the estimation of 

- background input statistics (Results section 2)
- input perturbations (Results section 3)
- neuronal adaptation parameters (Results section 4)
- synaptic coupling strengths (Results section 5)

__How to use__ 
run one of `baseline_input_inference.py`, `input_perturbation_inference.py`, `network_inference.py`, `adaptation_inference.py` with Python 2.x

_Required Python libraries_ 
numpy, scipy, numba, multiprocessing, math, os, collections, tables, time, matplotlib, warnings

These libraries can be conveniently obtained, for example, via a recent [Anaconda distribution](https://www.anaconda.com/download/) (Python 2.x)

_Remark_: the code is a strongly condensed version of the original implementation used for the paper

For questions please contact Josef Ladenbauer
