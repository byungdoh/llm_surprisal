"""
Calculates unigram surprisal based on counts from 16k training batches (~33B tokens) of the Pile,
saved in an array of length |V| under data/the_pile_16k_unigrams.npy
"""

import os, sys
import numpy as np
from transformers import AutoTokenizer
from get_llm_surprisal import generate_stories


def main():
    stories = generate_stories(sys.argv[1])
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision="step143000")
    counts = np.load("data/the_pile_16k_unigrams.npy").squeeze()
    log_total_counts = np.log2(np.sum(counts))
    mode = sys.argv[-1]
    assert mode in {"token", "word"}, ValueError('Calculation mode must be "token" or "word"')

    batches = []
    words = []
    for story in stories:
        words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        batches.append(tokenizer_output.input_ids)

    print("word unigramsurp")
    curr_word_ix = 0
    for batch in batches:
        toks = tokenizer.convert_ids_to_tokens(batch)
        surp = (log_total_counts - np.log2(counts[batch]))

        if mode == "token":
            # token-level surprisal
            for i in range(len(toks)):
                cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
                print(cleaned_tok, surp[i])

        elif mode == "word":
            # word-level surprisal
            curr_word_surp = []
            curr_toks = []
            for i in range(len(toks)):
                # for word-level surprisal
                curr_word_surp.append(surp[i])
                curr_toks += [toks[i]]
                curr_toks_str = tokenizer.convert_tokens_to_string(curr_toks)
                # summing token-level surprisal
                if words[curr_word_ix] == curr_toks_str.strip():
                    print(curr_toks_str.strip(), sum(curr_word_surp))
                    curr_word_surp = []
                    curr_toks = []
                    curr_word_ix += 1


if __name__ == "__main__":
    main()
