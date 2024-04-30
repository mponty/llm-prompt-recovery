# LLM Prompt Recovery Fifth Place Solution
### Team Z.D.Z Model Training Code

Hello!

Below is an outline for reproducing our solution for the **LLM Prompt Recovery** competition. Should you encounter any issues with the setup or code, or if you have any questions, please feel free to contact us at `dmitry.abulkhanov@gmail.com`.

### Repository Contents
```
data                    : Folder containing input data
output                  : Folder for storing training outputs
prompt_data_generation  : Contains scripts for data generation (execute in Kaggle kernel)
prompt_set_preparation  : Scripts for filtering the final prompt set
ranker_training         : Scripts to train ranking models
```

### Hardware Specifications
* **Rankers Training Setup:**
  * Ubuntu 22.04 LTS (1 TB boot disk)
  * 128 vCPUs, 512 GB RAM
  * 8 x NVIDIA Tesla A100 GPUs
* **Additional Data Generation Setup:**
  * Kaggle TPU-v3 kernel
* **Inference Setup:**
  * Kaggle P100 kernel

### Software Requirements
* Python 3.10.13
* CUDA 11.8
* (For detailed Python package requirements, see `requirements.txt`)

### Training Data Setup
Ensure the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed. Execute the following shell command to download the required datasets into the `./data` folder:
```
kaggle datasets download -d dmitriyab/llm-prompt-recovery-ranker-training-data -p data/ --unzip
```

### Model Training
Execute the following commands to train models:
```
cd ranker_training
sh run.sh
```
This will train the models and save them in the `./output` folder.

### Additional Data Generation
Data generation is performed using a Kaggle TPU-v3 kernel:
https://www.kaggle.com/code/dmitriyab/gemma-7b-tpu-generation-2/

### Submission Prompts Preparation
Run the following commands to prepare the submission prompts set:
```
cd prompt_set_preparation
sh run.sh
```
This will prepare the submission prompts set and save it in the `./output` folder.

### Final Submission
Upon completion of training and preparation, use the contents of the `./output` folder in the following Kaggle kernel for submitting your solution:
https://www.kaggle.com/code/dmitriyab/fifth-place-solution
