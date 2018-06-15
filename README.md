# inference-for-integrate-and-fire-models

Implementations of the inference methods for neuronal integrate-and-fire circuit models described in: __Ladenbauer, McKenzie, English, Hagens, Ostojic,__ ___Inferring and validating mechanistic models of neural microcircuits based on spike-train data___ (submitted 2018, [bioRxiv preprint](https://www.biorxiv.org/content/early/2018/02/07/261016))

The code contains examples for the estimation of 

- background input statistics (Results section 2)
- input perturbations (Results section 3)
- synaptic coupling strengths (Results section 4)
- neuronal adaptation parameters (Results section 5)

__How to use__ 
run one of `baseline_input_inference.py`, `input_perturbation_inference.py`, `network_inference.py`, `adaptation_inference.py` with Python 2.x

_Required Python libraries_ 
numpy, scipy, numba, multiprocessing, math, os, collections, tables, time, matplotlib, warnings

These libraries can be conveniently obtained, for example, via a recent [Anaconda distribution](https://www.anaconda.com/download/) (Python 2.x)

_Remark_: the code is a strongly condensed version of the original implementations used for the paper

For questions please contact Josef Ladenbauer
