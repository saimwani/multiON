# MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation
This repository hosts the code for the following paper:
* Saim Wani*, Shivansh Patel*, Unnat Jain*, Angel X. Chang, Manolis Savva, _MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation_ in NeurIPS, 2020 ([PDF](https://shivanshpatel35.github.io/multi-ON/resources/MultiON.pdf))

[![Conference](http://img.shields.io/badge/NeurIPS-2020-4b44ce.svg)](https://nips.cc/)

Project Website: https://shivanshpatel35.github.io/multi-ON/

![](docs/main_visualization.gif)

## Abstract
Navigation tasks in photorealistic 3D environments are challenging because they require perception and effective planning under partial observability. Recent work shows that map-like memory is useful for long-horizon navigation tasks. However, a focused investigation of the impact of maps on navigation tasks of varying complexity has not yet been performed.
We propose the multiON task, which requires navigation to an episode-specific sequence of objects in a realistic environment. MultiON generalizes the ObjectGoal navigation task [[1](https://arxiv.org/abs/1807.06757), [2](https://arxiv.org/abs/1705.08080)] and explicitly tests the ability of navigation agents to locate previously observed goal objects. We perform a set of multiON experiments to examine how a variety of agent models perform across a spectrum of navigation task complexities. Our experiments show that: i) navigation performance degrades dramatically with escalating task complexity; ii) a simple semantic map agent performs surprisingly well relative to more complex neural image feature map agents; and iii) even oracle map agents achieve relatively low performance, indicating the potential for future work in training embodied navigation agents using maps.


## Installing dependencies:


### Installing habitat-sim:

```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
pip install -r requirements.txt; 
python setup.py install --headless # (for headless machines)
python setup.py install # (for machines with display attached)
```

### Installing habitat-api:
```
git clone https://github.com/facebookresearch/habitat-api.git
cd habitat-api
pip install -e .
```

Install pytorch from https://pytorch.org/ according to your machine configuration. The code is tested on pytorch v1.4.0.

## Setup
Clone the repository and install the requirements:

```
git clone https://github.com/saimwani/multi-on.git
cd multi-on
pip install -r requirements.txt
```

Download Matterport3D data for Habitat by following the instructions mentioned [here](https://github.com/facebookresearch/habitat-api#data)

Run the following commands to download multiON dataset and cached oracle maps:
```
cd data/datasets/
wget -O multinav.zip "https://www.dropbox.com/s/src4dy0d5vnbpb8/multinav.zip?dl=0?dl=1"
unzip multinav.zip && rm multinav.zip
cd ../../
mkdir oracle_maps
cd oracle_maps
wget -O map300.pickle "https://www.dropbox.com/s/j25enox7kv76m3y/map300.pickle?dl=0?dl=1"
cd ../
```

The Matterport scene dataset and multiON dataset should be in data folder in the following format:

```
Multi-ON/
	data/
	  scene_datasets/
			mp3d/
				1LXtFkjw3qL/
					1LXtFkjw3qL.glb
					1LXtFkjw3qL.navmesh
					...
				...
      datasets/
        multinav/
          3_ON/
            train/
              ...
            val/
              val.json.gz
          2_ON
            ...
          1_ON
            ...

```				

## Usage

### Training

For training `ProjNeuralMap` agent, run this: - 

```
python habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_multinav.yaml --agent-type proj-neural --run-type train
```
### Pre-trained models

```
mkdir pretrained_models
``` 
Download relevant model checkpoint and place inside `pretrained_models` directory created above. 

| Model      | Checkpoint URL                                                                               |
|------------|:--------------------------------------------------------------------------------------------:|
| NoMap      | [link](https://drive.google.com/file/d/1gqco6r0s2fegftMFgLoFSC2RU6KLgiRY/view?usp=sharing)   |
| ProjNeural | [link](https://drive.google.com/file/d/1qpZ4dUNbGE9PpdDB2Agzyb7k2ZN0vZjJ/view?usp=sharing)   |
| ObjRecog   | [link]()                                                                                     |
| OracleMap  | [link]()                                                                                     |

### Evaluation

To evaluate the `ProjNeuralMap` pretrained model on the 3-ON test dataset, run the following command from the root folder (`multiON/`).

```
python habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_multinav.yaml --agent-type proj-neural --run-type eval
``` 
Average evaluation metrics are printed on the console when evaluation ends. Detailed metrics are placed in `eval/metrics` directory. 

## Citation
>Saim Wani*, Shivansh Patel*, Unnat Jain*, Angel X. Chang, Manolis Savva, 2020. MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation in Neural Information Processing Systems (NeurIPS). [PDF](https://shivanshpatel35.github.io/multi-ON/resources/MultiON.pdf)

## Bibtex
```
  @inproceedings{wani2020multion,
  title={Multi-ON: Benchmarking Semantic Map Memory using Multi-Object Navigation},
  author={Saim Wani and Shivansh Patel and Unnat Jain and Angel X. Chang and Manolis Savva},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2020},
}
```

## Acknowledgements
This repository is built upon habitat-lab's(https://github.com/facebookresearch/habitat-lab) repository.