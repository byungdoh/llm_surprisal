"""
GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"
"""

import os, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories


def main():
    stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]

    if "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(sys.argv[2])
    elif "gpt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "opt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    else:
        raise ValueError("Unsupported LLM variant")

    model = AutoModelForCausalLM.from_pretrained(sys.argv[2])
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    for story in stories:
        words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask
        start_idx = 0

        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            # for GPT-NeoX (bos_id not appended by default)
            if "gpt-neox" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]).unsqueeze(0),
                                                           "attention_mask": torch.tensor([1] + attn[:ctx_size-1]).unsqueeze(0)}),
                                start_idx))
            # for GPT-2/GPT-Neo (bos_id not appended by default)
            elif "gpt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]),
                                                            "attention_mask": torch.tensor([1] + attn[:ctx_size-1])}),
                                start_idx))
            # for OPT (bos_id appended by default)
            else:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
                                start_idx))

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

        # remaining tokens
        if "gpt-neox" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids).unsqueeze(0),
                                                       "attention_mask": torch.tensor([1] + attn).unsqueeze(0)}),
                           start_idx))
        elif "gpt" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids),
                                                       "attention_mask": torch.tensor([1] + attn)}),
                           start_idx))
        else:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn).unsqueeze(0)}),
                            start_idx))

    curr_word_ix = 0
    curr_word_surp = []
    curr_toks = ""

    print("word llm_surp")
    for batch in batches:
        batch_input, start_idx = batch
        output_ids = batch_input.input_ids.squeeze(0)[1:]

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]
        index = torch.arange(0, output_ids.shape[0])
        surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids])

        for i in range(start_idx, len(toks)):
            # necessary for diacritics in Dundee
            cleaned_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")

            # for token-level surprisal
            # print(cleaned_tok, surp[i].item())

            # for word-level surprisal
            curr_word_surp.append(surp[i].item())
            curr_toks += cleaned_tok
            # summing subword token surprisal ("rolling")
            words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
            if words[curr_word_ix] == "":
                print(curr_toks, sum(curr_word_surp))
                curr_word_surp = []
                curr_toks = ""
                curr_word_ix += 1


if __name__ == "__main__":
    main()
