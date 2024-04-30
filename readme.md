# LLM Prompt Recovery Fifth Place Solution
### Team Z.D.Z Model Training Code

Hello!

Below you can find a outline of how to reproduce our solution for the **LLM Prompt Recovery** competition.
If you run into any trouble with the setup/code or have any questions please contact us at `dmitry.abulkhanov@gmail.com`

### HARDWARE: (The following specs were used to create the original solution)
* Rankers training setup:
  * Ubuntu 22.04 LTS (1 Tb boot disk)
  * 128 vCPUs, 512 GB memory
  * 8 x NVIDIA Tesla A100
* Additional data generation setup:
  * Kaggle TPU-v3 kernel
* Inference setup:
  * Kaggle P100 kernel

### SOFTWARE (python packages are detailed separately in `requirements.txt`):
* Python 3.10.13
* CUDA 11.8

### TRAINING DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
run  the shell command below and ensure that it's placed downloaded content into `./data` folder
```
kaggle datasets download -d dmitriyab/llm-prompt-recovery-ranker-training-data -p data/ --unzip

```

### MODEL TRAINING
run the following commands
```
cd ranker_training
sh run.sh
```
It should train models and place them into `./output` folder

### ADDITIONAL DATA GENERATION
We use kaggle TPU-v3 kernel for generation additional data:
https://www.kaggle.com/code/dmitriyab/gemma-7b-tpu-generation-2/


### SUBMISSION PROMPTS PREPARATION
run the following commands
```
cd ranker_training
sh run.sh
```
It should prepare submission prompt set and place them into `./output` folder

### FINAL SUBMISSION

https://www.kaggle.com/code/dmitriyab/fifth-place-solution

