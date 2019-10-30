# inference-for-integrate-and-fire-models

Implementations of the inference methods for integrate-and-fire circuit models described in: 
__Ladenbauer et al.__ ___Inferring and validating mechanistic models of neural 
microcircuits based on spike-train data___ [Nature Communications 10:4933 (2019)](https://www.nature.com/articles/s41467-019-12572-0) [[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/261016v4)]

The code contains examples for inference of 

- background inputs
- input perturbations
- synaptic coupling
- neuronal adaptation

__How to use:__ 
run one of `baseline_input_inference.py`, `input_perturbation_inference.py`, `network_inference.py`, 
`adaptation_inference.py` (tested with Python 2.7 and 3.7)

Each script generates output graphs similar to those of the respective 
results section in the paper, typical run times are indicated in the scripts

_Required Python libraries_: 
numpy, scipy, numba, multiprocessing, math, os, collections, tables, time, matplotlib, warnings

These libraries can be conveniently obtained, for example, via a recent 
[Anaconda distribution](https://www.anaconda.com/download/)

<!--- _Remark_: the code is a strongly condensed version of the original implementations used for the paper -->

For questions please contact Josef Ladenbauer
