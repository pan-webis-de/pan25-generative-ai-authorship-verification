#!/usr/bin/env bash
# Copyright 2025 Janek Bevendorff, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Main train / val / test split (excludes o1 and deepseek for riddell-juola)
# shellcheck disable=SC2046
dataset-sample sample-balanced \
    -h data/summaries/pan25-gutenberg-19c-fiction.jsonl \
    $(for m in data/text-llm-jsonl/gutenberg-19c-fiction/*; do echo "-m $m"; done | grep -v o1) \
    --scramble-ids \
    --genre fiction \
    --output-file data/sampled/gutenberg-19c-fiction.jsonl \
    --max-imbalance 1

dataset-sample sample-balanced \
    -h data/summaries/pan25-riddell-juola.jsonl \
    $(for m in data/text-llm-jsonl/riddell-juola/*; do echo "-m $m"; done | grep -v 'o1\|deepseek') \
    --scramble-ids \
    --genre essays \
    --output-file data/sampled/riddell-juola.jsonl \
    --max-imbalance 4

dataset-sample sample-balanced \
    -h data/summaries/pan25-brennan-greenstadt.jsonl \
    $(for m in data/text-llm-jsonl/brennan-greenstadt/*; do echo "-m $m"; done | grep -v o1) \
    --scramble-ids \
    --genre essays \
    --output-file data/sampled/brennan-greenstadt.jsonl \
    --max-imbalance 4

dataset-sample sample-balanced \
    -h data/summaries/pan24-train-human.jsonl \
    $(for m in data/text-llm-jsonl/pan24-train/*; do echo "-m $m"; done | grep -v o1) \
    --scramble-ids \
    --genre news \
    --output-file data/sampled/pan24.jsonl \
    --max-imbalance 4

# Held-back o1 texts
dataset-sample sample-balanced \
    -m data/text-llm-jsonl/gutenberg-19c-fiction/gutenberg-19c-fiction-o1.jsonl \
    -m data/text-llm-jsonl/gutenberg-19c-fiction/gutenberg-19c-fiction-o1-mini.jsonl \
    --scramble-ids \
    --genre fiction \
    --max-machine 200 \
    --output-file data/sampled/gutenberg-19c-fiction-o1.jsonl

dataset-sample sample-balanced \
    -m data/text-llm-jsonl/riddell-juola/riddell-juola-deepseek-r1-distill-qwen-32b.jsonl \
    --scramble-ids \
    --genre essays \
    --max-machine 100 \
    --output-file data/sampled/riddell-juola-o1-deepseek.jsonl

dataset-sample sample-balanced \
    -m data/text-llm-jsonl/pan24-train/pan24-train-openai-o1.jsonl \
    --scramble-ids \
    --genre news \
    --max-machine 100 \
    --output-file data/sampled/pan24-o1.jsonl

# Obfuscations
dataset-sample sample-balanced \
    $(for m in data/text-llm-jsonl/gutenberg-19c-fiction-obfuscated/*; do echo "-m $m"; done) \
    --scramble-ids \
    --genre fiction-obfs \
    --max-machine 100 \
    --output-file data/sampled/gutenberg-19c-fiction-obfuscated.jsonl

dataset-sample sample-balanced \
    -h data/summaries/pan25-brennan-greenstadt-obfuscated.jsonl \
    $(for m in data/text-llm-jsonl/brennan-greenstadt-obfuscated/*; do echo "-m $m"; done) \
    --scramble-ids \
    --genre essays-obfs \
    --max-machine 300 \
    --max-imbalance 0 \
    --output-file data/sampled/brennan-greenstadt-obfuscated.jsonl

dataset-sample sample-balanced \
    -h data/summaries/pan25-riddell-juola-obfuscated.jsonl \
    $(for m in data/text-llm-jsonl/riddell-juola-obfuscated/*; do echo "-m $m"; done) \
    --scramble-ids \
    --genre essays-obfs \
    --max-machine 600 \
    --max-imbalance 0 \
    --output-file data/sampled/riddell-juola-obfuscated.jsonl

# ELOQUENT
dataset-sample sample-balanced \
    -h data/eloquent25/human.jsonl \
    $(for m in data/eloquent25/*; do echo "-m $m"; done | grep -v human) \
    --scramble-ids \
    --prefix-ids \
    --genre mixed-obfs \
    --max-imbalance 0 \
    --output-file data/sampled/eloquent.jsonl
