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

# Main train / val / test split
dataset-sample split data/sampled/pan24.jsonl -v .2 -t 0
dataset-sample split data/sampled/riddell-juola.jsonl -v .1 -t .2
dataset-sample split data/sampled/brennan-greenstadt.jsonl -v .1 -t .2
dataset-sample split data/sampled/gutenberg-19c-fiction.jsonl -v .1 -t .1

# Extra test data
dataset-sample split data/sampled/riddell-juola-obfuscated.jsonl -v 0 -t 1
dataset-sample split data/sampled/brennan-greenstadt-obfuscated.jsonl -v 0 -t 1
dataset-sample split data/sampled/gutenberg-19c-fiction-obfuscated.jsonl -v 0 -t 1
dataset-sample split data/sampled/gutenberg-19c-fiction-o1.jsonl -v 0 -t 1
dataset-sample split data/sampled/pan24-o1.jsonl -v 0 -t 1
dataset-sample split data/sampled/riddell-juola-o1-deepseek.jsonl -v 0 -t 1

# ELOQUENT
dataset-sample split data/sampled/eloquent.jsonl -v 0 -t 1

# Combine splits (except ELOQUENT)
find data/splits/ -type f -name '*-train.jsonl' -print0 | sort -z | xargs -0 cat  | shuf --random-source=<(yes 42) > data/splits/train.jsonl
find data/splits/ -type f -name '*-val.jsonl' -print0 | sort -z | xargs -0 cat  | shuf --random-source=<(yes 42) > data/splits/val.jsonl
find data/splits/ -type f -name '*-test.jsonl' -a -! -name '*eloquent*' -print0 | sort -z | xargs -0 cat | shuf --random-source=<(yes 42) > data/splits/test.jsonl
find data/splits/ -type f -name '*-test-truth.jsonl' -a -! -name '*eloquent*' -print0 | sort -z | xargs -0 cat | shuf --random-source=<(yes 42) > data/splits/test-truth.jsonl

echo
cat <<EOF | tee data/splits/summary.txt
----------
Checksums:
----------
$(md5sum data/splits/*)

------------
Split sizes:
------------
$(wc -l data/splits/train.jsonl data/splits/val.jsonl data/splits/test.jsonl data/splits/eloquent-test.jsonl)

--------------
Class balance:
--------------
$(for s in "train" "val" "test-truth" "eloquent-test-truth"; do
    python3 -c "import pandas as pd; \
        df = pd.read_json('data/splits/${s}.jsonl', lines=True); \
        nm = df['label'].sum(); \
        nh = len(df) - nm; \
        print(f'{'$s'.replace('-truth', '').capitalize()} set balance human/machine: {nh}/{nm} ({nh/nm:.2f})')"
done)
EOF
