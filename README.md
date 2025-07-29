## Setup

```sh
cd code

conda create -n silly-rewards python=3.10 
conda activate silly-rewards

pip install -r requirements.txt
pip install flash_attn==2.7.0.post2
pip install -e .
```

## Training
```sh
bash scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh
```

## Configurations

### Data
We include filtered and majority-labeled data in the paper. You may find a complete list in the `code/data` directory. For example, the ground truth data is termed `DeepScaleR`, and Llama 3.2 3B instruct labeled data, filtered to keep only the incorrect labels, is in the `DeepScaleR_mv_labeled_llama3.2_3b_instruct_incorrect` folder. You may change the data source by changing the variable `TASK` in `code/scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh`. 

### Rewards
We include a list of rewards used in the paper below. Furthermore, note that for models without a chat template, be sure to add `_r1_only` as the suffix. You may change the reward function by changing the variable `REWARD` in `code/scripts/rlvr_deepscaler_grpo_qwen_ground_truth.sh`. 

- `math`: Mathematical equivalence reward, which is the default
- `box_only_format`: Box-only formatting reward
- `contain_python_wo_backticks`: Mentioning of Python reward
- `random0.5`: Random reward with 50% returning 1


## Evaluations
To reproduce our evaluation results, use the following commands:

```sh
cd code

# For MATH-500 evaluation (requires NVIDIA A100 80GB PCIe for exact reproduction)
python scripts/eval_checkpoint.py --model_path Qwen/Qwen2.5-Math-7B --datasets MATH-500,AIME-2024,AIME-2025,AMC

# For MATH-500 evaluation matching our reported scores in wandb using checkpoints (requires NVIDIA H200 for exact reproduction)
python scripts/eval_checkpoint.py --model_path {} --datasets MATH-500,AIME-2024,AIME-2025,AMC --shards 2
```

Note: To exactly reproduce `temperature = 0` results, both the GPU type and `--shards` parameter must match the original evaluation setup. This is because the batch size passed into VLLM can cause generation fluctuations.

## Acknowledgments
This repository is built based on [TTRL](https://github.com/PRIME-RL/TTRL), which is built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We added asynchronous evaluation among other custom features to the codebase. 