# mlx-training-rs

A CLI to generate __synthetic__ data for MLX traning. The CLI is largely translated from the php version [here](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/?utm_source=pocket_reader).

## Fine-Tuning LLM for Dummy on Apple Silicon

Based on [this](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/?utm_source=pocket_reader), [this](https://www.reddit.com/r/LocalLLaMA/comments/191s7x3/a_simple_guide_to_local_llm_finetuning_on_a_mac/?share_id=hH4Vu8gxZgwYRvl_fIyOu&utm_content=1&utm_medium=ios_app&utm_name=ioscss&utm_source=share&utm_term=1), [this](https://www.reddit.com/r/LocalLLaMA/comments/18ujt0n/using_gpus_on_a_mac_m2_max_via_mlx_update_on/) and [this](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/).

### Install HomeBrew

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Setting up Python3

```shell
brew install python@3.11
```

### Download MLX and the Model to Fine-tuning

Clone MLX
```sh
git clone https://github.com/ml-explore/mlx-examples.git
```


```sh
python convert.py --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q --mlx-path ./Mistral-7B-Instruct-v0.2-mlx-4bit 
```

## Get started

```sh
export OPENAI_API_KEY=[you key]
mlxt --topic="Fine Tuning MLX"
```

will generate traning data for MLX, by a topic you are interested in using __knowledge distillation__(not really) by talking to GPT.
```
train.jsonl
valid.jsonl
```
