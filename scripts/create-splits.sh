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

dataset-sample split data/sampled/pan24.jsonl -v .2 -t 0
dataset-sample split data/sampled/riddell-juola.jsonl -v .1 -t .2
dataset-sample split data/sampled/brennan-greenstadt.jsonl -v .1 -t .2
dataset-sample split data/sampled/gutenberg-19c-fiction.jsonl -v .1 -t .1

cat data/splits/*-train.jsonl > data/splits/train.jsonl
cat data/splits/*-val.jsonl > data/splits/val.jsonl
cat data/splits/*-test.jsonl > data/splits/test.jsonl
