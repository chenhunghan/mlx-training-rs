# mlx-training-rs

<p align="center">
  <img src="./assets/logo.jpeg" width="320" height="320" alt="mlxt logo" />
</p>

A CLI to generate __synthetic__ data for MLX fine-tuning. The CLI is largely translated from the php version [here](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/?utm_source=pocket_reader).

## Demo

<p align="center">
  <img src="./assets/demo.gif" width="900" alt="Demo of mlxt" />
</p>

## QLoRa fine-tuning for dummies on Apple Silicon

Based on [this](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/?utm_source=pocket_reader), [this](https://www.reddit.com/r/LocalLLaMA/comments/191s7x3/a_simple_guide_to_local_llm_finetuning_on_a_mac/?share_id=hH4Vu8gxZgwYRvl_fIyOu&utm_content=1&utm_medium=ios_app&utm_name=ioscss&utm_source=share&utm_term=1), [this](https://www.reddit.com/r/LocalLLaMA/comments/18ujt0n/using_gpus_on_a_mac_m2_max_via_mlx_update_on/) and [this](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on/).

### Preparing

Install HomeBrew, it's a package manager that help use to install all other dependencies.

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Setting up Python3 (if you haven't)
```shell
brew install python@3.11
```

Clone MLX and download the model for fine-tuning.
```sh
git clone https://github.com/ml-explore/mlx-examples.git
```

Download and convert [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

```sh
cd mlx-examples/llm/hf-lllm
pip install -r requirements.txt # or pip3
python convert.py --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q --mlx-path ./Mistral-7B-Instruct-v0.2-mlx-4bit
```
We are adding `-q` for coverting into a 4-bit quantized MLX model to `./Mistral-7B-Instruct-v0.2-mlx-4bit`

It will tale some time...

The converted MLX version has something we don't need when fine-tuning the model, edit `./Mistral-7B-Instruct-v0.2-mlx-4bit/config.json`, replace all with:

```json
{
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
    "quantization": {
        "group_size": 64,
        "bits": 4
    }
}
```

### Generating Training Data

Delete example data in `mlx-examples/lora/data`, you can delete everything inside.

Install `mlxt`, the tool in this repo.
```sh
brew install chenhunghan/homebrew-formulae/mlx-training-rs
```

Generate a training on a topic you are interested in.
```sh
export OPENAI_API_KEY=[don't tell me your key]
mlxt --topic="[the topic you are interested, e.g. Large Language Model]"
```

### Fine-tuning!

```sh
cd mlx-examples/lora
pip install -r requirements.txt # or pip3
python lora.py --train --model ../llms/hf_llm/Mistral-7B-Instruct-v0.2-mlx-4bit --data ./data --batch-size 1 --lora-layers 4
```

To chat with your fine-tuned model, see [here](https://github.com/ml-explore/mlx-examples/tree/main/lora#generate)
