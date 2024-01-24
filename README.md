# nsd_visuo_semantics

[![License BSD-3](https://img.shields.io/pypi/l/nsd_visuo_semantics.svg?color=green)](https://github.com/KietzmannLab/nsd_visuo_semantics/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nsd_visuo_semantics.svg?color=green)](https://pypi.org/project/nsd_visuo_semantics)
[![Python Version](https://img.shields.io/pypi/pyversions/nsd_visuo_semantics.svg?color=green)](https://python.org)
[![tests](https://github.com/KietzmannLab/nsd_visuo_semantics/workflows/tests/badge.svg)](https://github.com/KietzmannLab/nsd_visuo_semantics/actions)
[![codecov](https://codecov.io/gh/KietzmannLab/nsd_visuo_semantics/branch/main/graph/badge.svg)](https://codecov.io/gh/KietzmannLab/nsd_visuo_semantics)


Code to reproduce results from Doerig et al. "Semantic scene descriptions as an objective of the human visual system"

----------------------------------

This [KietzmannLab] package was generated with [Cookiecutter] using [@KietzmannLab]'s [cookiecutter-template] template.



## Installation


To install latest development version :

    git clone https://github.com/KietzmannLab/nsd_visuo_semantics.git
    cd code_directory/
    pip install -e . 

    
To install the most stable version, please select a tagged release on Github (currently v0.0.2):

    pip install  git+https://github.com/KietzmannLab/pytorch-dataset-loaders.git@v0.0.2
    

## Downloading the NSD dat and RCNN weights

### Download the required elements of NSD

NSD is hosted on AWS. We will download the required parts of the dataset using boto3.
You will need to create an AWS account and configure your access keys as described here:
[https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)

Then, you can download the data using the following command:
```python
import nsd_visuo_semantics.utils.download_nsd_visuo_semantics as dl
dl.get_nsd('path_to_desired_download_location')
```

### Download RCNN weights

You also need to download the RCNN weights. [NEED TO UPDATE WHEN WE DECIDE HOW TO DO IT]


### Running analyses

Please note that most analyses require a large amount of memory. The searchlight analyses benefit from running on GPUs.


### Plotting brain maps

This is done in MATLAB and requires an installation of the following libraries:

Freesurfer: wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-centos7_x86_64-7.4.1.tar.gz
cvncode: git clone https://github.com/cvnlab/cvncode.git
knkutils: git clone https://github.com/cvnlab/knkutils.git
npy-matlab: git clone https://github.com/kwikteam/npy-matlab.git

Then edit the paths in the matlab plot_brains.m script to point to the locations where this is downloaded.

## Citation

Please cite Doerig et al., Nature, 2023 if you use any of this code in your work.


## License

Distributed under the terms of the [BSD-3] license,
"nsd_visuo_semantics" is free and open source software.

