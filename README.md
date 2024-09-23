Improving Tool Acquisition in Large Language Models
===

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)


Supporting material for the thesis "Thinking Forwards, Backwards, and in Code: Improving Tool Acquisition in Large Language Models"

Authors: **Huey Sun** 

## Repository Structure 

- `Results:` The ToolSandbox evaluations results can be found [here](/results/). Each result has a summary at the end, which was used to create the tables in the thesis. 

- `Data:` The data used to finetune the models can be found [here](/data/). As Mistral's lack of chat formatting means that incremental masking is impossible (the output is not perfectly autoregressive at each conversation turn), each assistant message has been individually preformatted.

- `Scripts:` The scripts used to convert json conversations to Mistral's expected formatting can be found [here](/scripts/).

## Other Code

You can find my fork of the torchtune library [here](https://github.com/itshuey/torchtune), where I add support for tool chat and Mistral formatting in training.
