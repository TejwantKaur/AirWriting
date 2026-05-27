import os
import torch
from transformers import pipeline

next_word_generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from pathlib import Path
from collections import Counter

# PATHS
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "entertainment_training_data.txt"
)

# GPT2 MODEL
MODEL_NAME = "AventIQ-AI/gpt2-next-word-prediction"

nw_model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME
)
nw_tokenizer = GPT2Tokenizer.from_pretrained(
    MODEL_NAME
)

# LOAD TITLE
def load_title_list(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        print(
            f"Warning: {file_path} not found"
        )
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        return [
            line.strip().upper()
            for line in f
            if line.strip()
        ]

TITLE_LIST = load_title_list()

# NEXT WORD SUGGESTIONS
def get_next_word_suggestions(
    sentence,
    limit=3
):
    if not sentence.strip():
        return []
    sentence_upper = sentence.strip().upper()
    last_word = sentence_upper.split()[-1]
    suggestions = []
    seen = set()

   
    # TITLE PREFIX MATCH
    for line in TITLE_LIST:

        if (
            line.startswith(sentence_upper)
            and line != sentence_upper
        ):

            remainder = line[
                len(sentence_upper):
            ].strip()

            if not remainder:
                continue

            next_word = remainder.split()[0]

            next_word_clean = "".join(
                c
                for c in next_word
                if c.isalpha()
            )

            if (
                len(next_word_clean) >= 3
                and next_word_clean not in seen
            ):

                seen.add(next_word_clean)

                suggestions.append(
                    next_word_clean
                )

        if len(suggestions) >= limit:
            break

    # LAST WORD FALLBACK
    if (
        len(suggestions) < limit
        and len(last_word) > 2
    ):
        counter = Counter()
        for line in TITLE_LIST:
            words = line.split()
            for i, word in enumerate(words):
                if (
                    word == last_word
                    and i + 1 < len(words)
                ):
                    next_word_clean = "".join(
                        c
                        for c in words[i + 1]
                        if c.isalpha()
                    )
                    if len(next_word_clean) >= 3:
                        counter[next_word_clean] += 1
        for word, _ in counter.most_common(limit):
            if word not in seen:
                seen.add(word)
                suggestions.append(word)
            if len(suggestions) >= limit:
                break

    # GPT2 FALLBACK
    remaining = limit - len(suggestions)
    if remaining > 0:
        input_ids = nw_tokenizer.encode(
            sentence.strip(),
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = nw_model(input_ids)
            logits = outputs.logits[:, -1, :]
            top_k = torch.topk(
                logits,
                k=300,
                dim=-1
            )
        for idx in top_k.indices[0].tolist():
            word = nw_tokenizer.decode(
                [idx],
                skip_special_tokens=True
            ).strip()

            word_clean = "".join(
                c
                for c in word
                if c.isalpha()
            ).upper()

            if len(word_clean) < 2:
                continue

            if word_clean not in seen:
                seen.add(word_clean)
                suggestions.append(word_clean)

            if len(suggestions) >= limit:
                break

    return suggestions[:limit]