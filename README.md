# ISCA50_AE_Extend

We provide our fault injection framework for various workloads. The methodology to inject faults into the DNN training program is similar for all workloads. We will open-source the complete fault injection framework for all DNN workloads.

In each fault injection experiment, we pick a random training epoch, a random training step, a random layer (selected from both layers in the forward pass and the backward pass), and a random software fault model, and continue training the workload to observe the outcome.

In order to inject faults to the backward pass and also correctly propagate the error effects, we manually implemented the backward pass for each workload, which can be found in the `fault_injection/models` folder.

We have performed 2.9M fault injection experiments to obtain statistical results. In this artifact evaluation, we provide three reproducible examples of fault injections that correspond to three outcomes (Masked, Immediate INFs/NaNs, and SlowDegrade) reported in our paper. 

We also provide instructions for running more fault injection experiments.

## System Requirements
### Hardware dependencies
Our framework runs on Google Cloud TPU VMs.
### Software dependencies
Our framework requires the following tools:

```
Tensorflow 2.6.0
Numpy 1.19.5
Gdown 4.6.4
```

## Installation 

### Step 1. create Google Cloud TPU VM

```
export PROJECT_ID=${PROJECT_ID}
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone={TPU_LOCATION} --accelerator-type={TPU_TYPE} --version=v2-alpha
```

```
PROJECT_ID: The Google cloud user ID.
TPU_NAME: A user defined name.
TPU_LOCATION: The cloud region, e.g., us-central1-a.
TPU_TYPE: The type of the cloud TPU, e.g., v2-8.
```

For more details on creating TPU VMs, please check [this page](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).


### Step 2. ssh to the TPU VM:

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${TPU_LOCATION} --project ${PROJECT_ID}
```

### Step 3. check numpy and tensorflow versions

```
import numpy
numpy.__version__
import tensorflow
tensorflow.__version__
```
Make sure that the version of numpy is 1.19.5, and the version of tensorflow is 2.6.0. If the versions don't match, please install the correct versions.


### Step 4. clone our github repo.
```
git clone git@github.com:YLab-UChicago/ISCA_AE_Extend.git
```

### Step 5. Download checkpoints for various workloads from Google Drive.

```
pip install gdown 
gdown --folder https://drive.google.com/drive/folders/1HVRFWY7NI5xr5qzR8yNeSKCRVnJNnqFf?usp=sharing
```
If gdown cannot be found, specify the full path where gdown is installed, mostly likely in `\~/.local/bin`.


## Experiment workflow

The `reproduce_injections.py` file is the top-level program to perform the entire workflow of a fault injection experiment, which takes in one argument `--file`, which specifies the injection configs, e.g., the target training epoch, target training step, target layer, faulty values, etc. The configs of our three examples are provided in the `injections` folder.

For each injection, the program generates an output file named `replay_inj_TARGET_INJECTION.txt` file under the `fault_injection` directory, which records the training loss, training accuracy for each training iteration, and test loss and test accuracy for each epoch. For examples that generate INFs/NaNs, the file will also record when INF/NaN values are observed.

To execute each example, run:

```
cd fault_injection
python3 reproduce_injections.py --file injections/WORKLOAD/inj_TARGET_INJECTION.csv
```

    
## Experiment customization
To run other examples, one can modify the `inj_TARGET_INJECTION.csv` files under the `injection` folder and specify different training epochs, training steps, target layers, and faulty values. The evaluation process is similar to the examples provided.
