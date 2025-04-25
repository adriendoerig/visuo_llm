# High-level visual representations in the human brain are aligned with large language models
**Authors: Adrien Doerig, Tim C Kietzmann, Emily Allen, Yihan Wu, Thomas Naselaris, Kendrick Kay, & Ian Charest**

Accepted in Nature Machine Intelligence
🔗 You can find our preprint [here](https://arxiv.org/abs/2209.11737) 🔗

### Abstract
*The human brain extracts complex information from visual inputs, including objects, their spatial and semantic interrelations, and their interactions with the environment. However, a quantitative approach for studying this information remains elusive. Here, we test whether the contextual information encoded in large language models (LLMs) is beneficial for modelling the complex visual  information extracted by the brain from natural scenes. We show that LLM embeddings of scene captions successfully characterise brain activity evoked by viewing the natural scenes. This mapping captures selectivities of different brain areas, and is sufficiently robust that accurate scene captions can be reconstructed from brain activity. Using carefully controlled model comparisons, we then proceed to show that the accuracy with which LLM representations match brain representations derives from the ability of LLMs to integrate complex information contained in scene captions beyond that conveyed by individual words. Finally, we train deep neural network models to transform image inputs into LLM representations. Remarkably, these networks learn representations that are better aligned with brain representations than a large number of state-of-the-art alternative models, despite being trained on orders-of-magnitude less data. Overall, our results suggest that LLM embeddings of scene captions provide a representational format that accounts for complex information extracted by the brain from visual inputs.*


## Installation

To install latest development version :

    git clone https://github.com/adriendoerig/visuo_llm.git
    cd visuo_llm
    pip install -e . 
    
## Reproducing paper results
The ```./examples``` folder contains code to run the paper's analyses.

## Downloading the NSD data and RCNN weights

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

Analyses based on RCNN models require downloading the RCNN weights. They will be uploaded soon. In the interval, feel free to send an email to the paper's corresponding author.


### Running analyses

Please note that most analyses require a large amount of memory. The searchlight analyses benefit from running on GPUs.


### Plotting brain maps

Paper brain maps are created using MATLAB and requires an installation of the following libraries:

Freesurfer: ```wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-centos7_x86_64-7.4.1.tar.gz```
cvncode: ```git clone https://github.com/cvnlab/cvncode.git```
knkutils: ```git clone https://github.com/cvnlab/knkutils.git```
npy-matlab: ```git clone https://github.com/kwikteam/npy-matlab.git```

Then edit the paths in the matlab examples/plot_searchlight_brain_maps.m or other relevant scripts to point to the locations where this is downloaded.

Alternatively, you can use other python braim plotting functions (e.g. nibabel).


## Citation

@misc{doerig2024visualrepresentationshumanbrain,
      title={Visual representations in the human brain are aligned with large language models}, 
      author={Adrien Doerig and Tim C Kietzmann and Emily Allen and Yihan Wu and Thomas Naselaris and Kendrick Kay and Ian Charest},
      year={2024},
      eprint={2209.11737},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2209.11737}, 
}

## License

This project is licensed under the [MIT License](LICENSE).