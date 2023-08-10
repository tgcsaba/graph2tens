----------------------------------------------------------------------------------------------------
This code package contains supplementary code for the NeurIPS 2022 submission
    Capturing Graphs with Hypo-Elliptic Diffusions
----------------------------------------------------------------------------------------------------
0. Installing prerequisites:
    - Please create a clean conda environment, the code was tested with python==3.7.9, so use
        `conda create -n test_env python=3.7.9
         conda activate test_env` 
    - Install the prerequisites using `pip install -r requirements.txt`
1. Printing the results:
    - The results for each dataset/model/seed combination are included in the results directory.
    - To print a summary of results, please run `python get_results.py`
2. Running the experiments:
    - Remove/rename the results directory, since the run script skips all existing results
    - Optional: Open configs.yaml, and remove/change the dataset/model configurations as required
    - Call `python run_experiments.py [GPU_ID]` for running on GPU, or leave GPU_ID empty for CPU
3. Further remarks:
    - Compared to the paper certain naming conventions are different in the code:
        - GSAN = G2TAN
        - GSN = G2TN
        - BP = ZeroStart
        - Embed = AlgOpt
----------------------------------------------------------------------------------------------------
