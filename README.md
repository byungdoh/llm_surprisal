# Frequency Explains the Inverse Correlation of Large Language Models’ Size, Training Data Amount, and Surprisal’s Fit to Reading Times

## Introduction
This is the code repository for the paper [Frequency Explains the Inverse Correlation of Large Language Models’ Size, Training Data Amount, and Surprisal’s Fit to Reading Times](https://arxiv.org/pdf/2402.02255.pdf), including code for calculating surprisal using autoregressive LMs from the [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) library.

## Setup
Install the following major dependencies:
- [PyTorch](https://pytorch.org) (v1.13.0 used in this work)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/installation) (v4.24.0 used in this work)

## LM Surprisal Calculation
The command `python get_llm_surprisal.py INPUT_FILE MODEL_NAME (PYTHIA_CHECKPOINT) MODE > OUTPUT_FILE` (e.g. `python get_llm_surprisal.py my_stimuli.sentitems EleutherAI/pythia-70m step143000 word > my_stimuli.pythia-70m-143k-word.surprisal`) can be used to calculate LM surprisal predictors.
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

Pythia family:
"EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
"EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
each with checkpoints specified by training steps:
"step1", "step2", "step4", ..., "step142000", "step143000"
```

The output should be a space-delimited two-column file containing the word and LM surprisal.

```
$ head my_stimuli.pythia-70m-143k-word.surprisal
word llmsurp
If 8.806196212768555
you 1.4949685335159302
were 5.1340837478637695
to 6.376556396484375
journey 15.824409484863281
to 2.454831123352051
the 2.214837074279785
North 7.6205949783325195
of 4.805300712585449
England, 2.236417591571808
```

The repository also supports token-level surprisal calculation when `MODE` is set to `token` instead of `word`.

```
$ head my_stimuli.pythia-70m-143k-token.surprisal
word llmsurp
If 8.806196212768555
you 1.4949685335159302
were 5.1340837478637695
to 6.376556396484375
journey 15.824409484863281
to 2.454831123352051
the 2.214837074279785
North 7.6205949783325195
of 4.805300712585449
England 0.8403297066688538
, 1.396087884902954
```

## Unigram Surprisal Calculation
Likewise, the command `python get_unigram_surprisal.py INPUT_FILE MODE > OUTPUT_FILE` (e.g. `python get_unigram_surprisal.py my_stimuli.sentitems word > my_stimuli.unigram-word.surprisal`) can be used to calculate unigram surprisal predictors.
Unigram surprisal is based on counts from 16k training batches (~33B tokens) of the Pile provided by the developers of [Pythia](https://github.com/EleutherAI/pythia), saved in an array of length $|V|$ under `data/the_pile_16k_unigrams.npy`.

```
$ head my_stimuli.unigram-word.surprisal
word unigramsurp
If 12.574996883891082
you 8.711964911935148
were 9.396668283684079
to 6.280557174231113
journey 15.774369150470388
to 6.280557174231113
the 5.17205129965733
North 14.034198096118661
of 5.986146708647226
England, 19.830069835326498
```

```
$ head my_stimuli.unigram-token.surprisal
word unigramsurp
If 12.574996883891082
you 8.711964911935148
were 9.396668283684079
to 6.280557174231113
journey 15.774369150470388
to 6.280557174231113
the 5.17205129965733
North 14.034198096118661
of 5.986146708647226
England 14.813701157394885
, 5.016368677931613
```

## Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.531@osu.edu](mailto:oh.531@osu.edu)).
