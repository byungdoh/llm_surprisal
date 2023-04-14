# Why Does Surprisal From Larger Transformer-Based Language Models Provide a Poorer Fit to Human Reading Times?

## Introduction
This is the code repository for the paper [Why Does Surprisal From Larger Transformer-Based Language Models Provide a Poorer Fit to Human Reading Times?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00548), including code for calculating surprisal using autoregressive LMs from the [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) library.

## Setup
Install the following major dependencies:
- [PyTorch](https://pytorch.org) (v1.13.0 used in this work)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/installation) (v4.24.0 used in this work)

## Surprisal Calculation
The command `python main.py INPUT_FILE MODEL_NAME > OUTPUT_FILE` (e.g. `python main.py my_stimuli.sentitems gpt2 > my_stimuli.gpt2.surprisal`) can be used to calculate by-word surprisal predictors.
The input file is split according to `!ARTICLE` delimiters and assigned to different batches. 

```
$ head my_stimuli.sentitems
!ARTICLE
If you were to journey to the North of England, you would come to a valley that is surrounded by moors as high as mountains.
```

The LMs that were used in this work are as follows:

```
GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"
```

The output should be a space-delimited two-column file containing the word and LM surprisal.

```
$ head my_stimuli.gpt2.surprisal
word llm_surp
If 7.758239269256592
you 0.8086858987808228
were 5.417673110961914
to 2.094998598098755
journey 14.621834754943848
to 2.1376590728759766
the 2.6726458072662354
North 6.569814682006836
of 7.013529300689697
England, 4.587411642074585
```

## Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.531@osu.edu](mailto:oh.531@osu.edu)).
