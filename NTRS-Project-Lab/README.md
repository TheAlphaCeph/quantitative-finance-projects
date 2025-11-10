# Disclaimer: 
Queries are designed to retrieve data from WRDS. Users must have valid access to WRDS to use them. No proprietary data is included in this repository

# Note: 
You can either run the fundamentals_data_2.1.ipynb file or download the full_fundamental_dataset.csv from the shared drive to get the data. You can include it when you are running the scripts on your local computer. Other large files are also stored remotely in the shared drive.

To run the scripts, it is recommended to use Google Colab so that you can use GPUs to reduce the runtime. You would need to use Conda to access the RAPIDs packages. See more details below. 

As mentioned above, some results files are too large to be stored in Git. You can download them from the shared NTRS folder (NTRS_models/Prepared Data/data and NTRS_models/Prepared Data/full_results).

Setting up RAPIDS:
[user@midway3-login4 ~]$ module load python/miniforge-24.1.2  # Change the version if needed; run "module avail python" to see the list \
[user@midway3-login4 ~]$ module load cuda/12.2  # Change the version if needed; run "module avail cuda" to see the list \
[user@midway3-login4 ~]$ conda install cuda-cudart cuda-version=12 \
[user@midway3-login4 ~]$ conda create -n msfm \
-c rapidsai -c conda-forge \
-c nvidia rapids=25.02 python=3.11 \
'cuda-version>=12.0,<=12.8'  # this command was generated using the configurator at https://docs.rapids.ai/install/ \
[user@midway3-login4 ~]$ source activate msfms

Refer to SLURM guide for more details on managing jobs: https://rcc-uchicago.github.io/user-guide/slurm/sbatch/
For CPU clusters, use caslake; for GPU clusters, use gpu. Set up log files and error files for each individual model for better debugging.
