# Capturing Graphs with Hypo-Elliptic Diffusions
---
A novel GNN utilizing tensor features of random walks for (deep) parametric feature extraction
---
This code package contains supplementary code to the NeurIPS 2022 paper <a href="https://arxiv.org/abs/2205.14092" title="Capturing Graphs with Hypo-Elliptic Diffusions">Capturing Graphs with Hypo-Elliptic Diffusions</a>.

## Installation
- Please create a clean conda environment, the code was tested with python==3.7.9, so ideally use the command:
```
  conda create -n test_env python=3.7.9
  conda activate test_env
```
- Install the prerequisites using `pip install -r requirements.txt`
## Running the experiments:
- Remove/rename the results directory, since the run script skips all existing results
- Optional: Open the file configs.yaml, and remove/change the dataset/model configurations as required
- Call `python run_experiments.py [GPU_ID]` for running on GPU, or leave GPU_ID empty for CPU
## Printing the results:
- The results for each dataset/model/seed combination are included in the results directory.
- To print a summary of results, please run `python get_results.py`
## Further remarks:
- Compared to the paper certain naming conventions are different in the code (TODO):
    - The names of the models: `GSAN == G2TAN`, `GSN == G2TN`
    - In the variations: `BP == ZeroStart`, `Embed == AlgOpt`
----------------------------------------------------------------------------------------------------
