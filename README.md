# Are Vision LLMs Road-Ready? A Comprehensive Benchmark for Safety-Critical Driving Video Understanding

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> The benchmark toolbox for the paper: **Are Vision LLMs Road-Ready? A Comprehensive Benchmark for Safety-Critical Driving Video Understanding**

## 📝 Project Overview

DVBench is a comprehensive benchmark designed to evaluate the video understanding capabilities of Vision Large Language Models (VLLMs) in safety-critical driving scenarios. This benchmark focuses on assessing models' ability to understand driving videos, which is crucial for the safe deployment of autonomous and assisted driving technologies.

### Main Contributions

- Problem Identification: We are among the first to investigate VLLMs' capabilities in perception and reasoning within safety-critical (Crash, Near-Crash) driving scenarios and systematically define the hierarchical abilities essential for evaluating the safety of autonomous driving systems in high-risk contexts.

- Benchmark Development: We introduce DVBench, the first comprehensive benchmark for safety-critical driving video understanding, featuring 10,000 curated multiple-choice questions across 25 key driving-related abilities. DVBench is designed to rigorously assess perception and reasoning in dynamic driving environments.

- Systematic Evaluation: We evaluate 14 state-of-the-art VLLMs, providing an in-depth analysis of their strengths and limitations in safety-critical driving scenarios. This paper establishes structured evaluation protocols and infrastructure to enable fair comparisons and guide future advancements in VLLMs for autonomous driving.

### VLLM Performance
<h4>Detailed Performance of the VLLMs on DVBench using GroupEval (L2 abilities)</h2>

<p>The following abbreviations are used:</p>
<ul>
    <li><b>Perception Abilities:</b> EC (Environmental Conditions), PI (Physical Infrastructure), OC (Operational Constraints), Obj (Objects), Zone (Zones)</li>
    <li><b>Reasoning Abilities:</b> EU (Event Understanding), BMA (Behavior & Maneuver Analysis), SR (Spatial Reasoning), RHA (Risk & Hazard Assessment), CR (Causal & Responsibility)</li>
</ul>
<table>
  <thead>
    <tr>
      <th rowspan="2"><strong>VLLMs</strong></th>
      <th colspan="6" style="text-align:center;"><strong>Perception</strong></th>
      <th colspan="6" style="text-align:center;"><strong>Reasoning</strong></th>
    </tr>
    <tr>
      <th><strong>Overall</strong></th>
      <th><strong>EC</strong></th>
      <th><strong>PI</strong></th>
      <th><strong>OC</strong></th>
      <th><strong>Obj</strong></th>
      <th><strong>Zone</strong></th>
      <th><strong>Overall</strong></th>
      <th><strong>EU</strong></th>
      <th><strong>BMA</strong></th>
      <th><strong>SR</strong></th>
      <th><strong>RHA</strong></th>
      <th><strong>CR</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LLaMA-VID-7B<sup>[1]</sup></td>
      <td>12.2%</td><td>26.7%</td><td>2.8%</td><td>13.5%</td><td>11.4%</td><td>9.1%</td>
      <td>10.4%</td><td>2.0%</td><td>3.3%</td><td>7.5%</td><td>19.3%</td><td>11.4%</td>
    </tr>
    <tr>
      <td>LLaMA-VID-13B<sup>[1]</sup></td>
      <td>12.8%</td><td>30.0%</td><td>0.0%</td><td>8.1%</td><td>15.9%</td><td>12.1%</td>
      <td>9.6%</td><td>3.9%</td><td>0.0%</td><td>6.0%</td><td>21.6%</td><td>4.5%</td>
    </tr>
    <tr>
      <td>LLaVA-One-Vision-0.5B<sup>[2]</sup></td>
      <td>15.0%</td><td>40.0%</td><td>2.8%</td><td>16.2%</td><td>13.6%</td><td>6.1%</td>
      <td>11.1%</td><td>5.9%</td><td>10.0%</td><td>9.0%</td><td>21.6%</td><td>0.0%</td>
    </tr>
    <tr>
      <td>Qwen2-VL-7B<sup>[3]</sup></td>
      <td>25.6%</td><td>43.3%</td><td>2.8%</td><td>27.0%</td><td>27.3%</td><td>30.3%</td>
      <td>27.1%</td><td>33.3%</td><td>13.3%</td><td>17.9%</td><td>33.0%</td><td>31.8%</td>
    </tr>
    <tr>
      <td>LLaVA-Next-Video-7B<sup>[4]</sup></td>
      <td>18.3%</td><td>43.3%</td><td>2.8%</td><td>16.2%</td><td>18.2%</td><td>15.2%</td>
      <td>16.1%</td><td>7.8%</td><td>13.3%</td><td>10.4%</td><td>27.3%</td><td>13.6%</td>
    </tr>
    <tr>
      <td>Video-LLaVa-7B<sup>[5]</sup></td>
      <td>21.1%</td><td>70.0%</td><td>2.8%</td><td>8.1%</td><td>22.7%</td><td>9.1%</td>
      <td>18.6%</td><td>5.9%</td><td>13.3%</td><td>10.4%</td><td>34.1%</td><td>18.2%</td>
    </tr>
    <tr>
      <td>PLLaVA-7B<sup>[6]</sup></td>
      <td>20.0%</td><td>56.7%</td><td>2.8%</td><td>10.8%</td><td>18.2%</td><td>18.2%</td>
      <td>17.9%</td><td>3.9%</td><td>10.0%</td><td>10.4%</td><td>34.1%</td><td>18.2%</td>
    </tr>
    <tr>
      <td>LLaVA-Next-Video-34B<sup>[4]</sup></td>
      <td>23.3%</td><td>50.0%</td><td>25.0%</td><td>21.6%</td><td>11.4%</td><td>15.2%</td>
      <td>16.1%</td><td>5.9%</td><td>10.0%</td><td>17.9%</td><td>27.3%</td><td>6.8%</td>
    </tr>
    <tr>
      <td>LLaVA-One-Vision-7B<sup>[2]</sup></td>
      <td>28.3%</td><td>70.0%</td><td>19.4%</td><td>21.6%</td><td>20.5%</td><td>18.2%</td>
      <td>20.0%</td><td>3.9%</td><td>20.0%</td><td>16.4%</td><td>37.5%</td><td>9.1%</td>
    </tr>
    <tr>
      <td>PLLaVA-13B<sup>[6]</sup></td>
      <td>23.9%</td><td>63.3%</td><td>11.1%</td><td>18.9%</td><td>18.2%</td><td>15.2%</td>
      <td>15.4%</td><td>5.9%</td><td>0.0%</td><td>9.0%</td><td>34.1%</td><td>9.1%</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B<sup>[3]</sup></td>
      <td>32.8%</td><td>50.0%</td><td>25.0%</td><td>35.1%</td><td>22.7%</td><td>36.4%</td>
      <td>33.9%</td><td>41.2%</td><td>13.3%</td><td>31.3%</td><td>42.0%</td><td>27.3%</td>
    </tr>
    <tr>
      <td>Qwen2-VL-2B<sup>[3]</sup></td>
      <td>31.7%</td><td>76.7%</td><td>27.8%</td><td>24.3%</td><td>20.5%</td><td>18.2%</td>
      <td>26.4%</td><td>7.8%</td><td>26.7%</td><td>16.4%</td><td>44.3%</td><td>27.3%</td>
    </tr>
    <tr>
      <td>MiniCPM-V<sup>[7]</sup></td>
      <td>39.4%</td><td>70.0%</td><td>19.4%</td><td>45.9%</td><td>36.4%</td><td>30.3%</td>
      <td>35.4%</td><td>39.2%</td><td>16.7%</td><td>22.4%</td><td>53.4%</td><td>27.3%</td>
    </tr>
    <tr>
      <td>LLaVA-One-Vision-72B<sup>[2]</sup></td>
      <td>36.7%</td><td>66.7%</td><td>16.7%</td><td>40.5%</td><td>34.1%</td><td>30.3%</td>
      <td>36.4%</td><td>33.3%</td><td>30.0%</td><td>29.9%</td><td>46.6%</td><td>34.1%</td>
    </tr>
  </tbody>
</table>


## 🛠️ Installation Guide

### Prerequisites

- Python 3.10+
- PyTorch
- CUDA-supported GPU (for larger models)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/tong-zeng/DVBench.git
cd DVBench
```

2. Environment Setup:

   We recommend using Conda for efficient environment management.
   
   We have exported our conda environments for different models in the `envs` folder. You can create a conda environment using these yml files:
   
   To create an environment for vllm models:

   ```bash
   conda env create -f envs/driving_vllm_bench_vllm_latest.yml
   # Activate the environment
   conda activate driving_vllm_bench_vllm_latest
   ```
   To create environments for huggingface models:

   ```bash
   conda env create -f envs/driving_vllm_bench_llamavid.yml
   # Activate the environment
   conda activate driving_vllm_bench_llamavid
   ```

   Available environment files:
   - `driving_vllm_bench_llamavid.yml` - Environment for LLaVA-Vid
   - `driving_vllm_bench_chatunivi.yml` - Environment for Chat-UniVi
   - `driving_vllm_bench_minicpmv.yml` - Environment for MiniCPM-V
   - `driving_vllm_bench_pllava.yml` - Environment for PLLaVA
   - `driving_vllm_bench_videochat2.yml` - Environment for VideoChat2
   - `driving_vllm_bench_videochatgpt.yml` - Environment for VideoChatGPT
   - and more...
   
   For more information on managing conda environments, please refer to the [official conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).


## 📊 Usage

### Running the Benchmark

We have prepared inference scripts for all supported models to simplify the benchmark process. The benchmark process consists of two main steps: running inference and calculating performance metrics.

#### Step 1: Running Inference

We provide ready-to-use bash scripts for each model in the `scripts` directory:
- HuggingFace models: `scripts/hf/`
- VLLM models: `scripts/vllm/`

```bash
# For example, to run inference with LLaVA-Vid 7B model
cd DVBench
bash scripts/hf/llama_vid_7b.sh

# For MiniCPM-V model
bash scripts/vllm/minicpmv.sh

# You may need to modify the scripts to adjust paths or GPU settings
```

The inference scripts will:
1. Set up the appropriate environment variables
2. Run the model on the test dataset
3. Save model responses to the specified output directory

#### Step 2: Calculating Performance

After running inference, use the `auto_accuracy.py` script to calculate performance metrics:

```bash
# Calculate performance metrics for all models
python dvbench/benchmarking/auto_accuracy.py
```

This will generate comprehensive performance metrics across all evaluation dimensions and save the results in the specified output directory.


## 📊 The Fine-Tuning code and Fine-Tuned Models

DVBench evaluates vision language models on the following key aspects:

Please refer to https://github.com/tong-zeng/qwen2-vl-finetuning.git

## 🔄 Supported Models

DVBench currently supports evaluation of the following vision language models:

- LLaMA-VID-7B
- LLaMA-VID-13B
- LLaVA-One-Vision-0.5B
- Qwen2-VL-7B
- LLaVA-Next-Video-7B
- Video-LLaVa-7B
- PLLaVA-7B
- LLaVA-Next-Video-34B
- LLaVA-One-Vision-7B
- PLLaVA-13B
- Qwen2-VL-72B
- Qwen2-VL-2B
- MiniCPM-V
- LLaVA-One-Vision-72B

To add a new model, please refer to the example implementations in the `dvbench/inference/models/` directory.

## 📚 Project Structure

```
DVBench/
├── dvbench/                # Main code package
│   ├── benchmarking/       # Evaluation benchmark related code
│   ├── inference/          # Model inference and interfaces
│   │   ├── configs/        # Model configuration files
│   │   ├── models/         # Model implementations
│   │   └── utils/          # Utility functions
│   └── visual_processing/  # Video and image processing tools
├── envs/                   # Conda environment yml files
├── scripts/                # Running scripts
│   ├── hf/                 # HuggingFace model inference scripts
│   └── vllm/               # VLLM model inference scripts
├── all_experiments/        # Experiment results storage
└── videos/                 # Test video set
```

## 📄 License

DVBench is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).


## 📧 Contact

For any questions or suggestions, please open an issue.

---

**Project Repository**: [https://github.com/tong-zeng/DVBench](https://github.com/tong-zeng/DVBench)
