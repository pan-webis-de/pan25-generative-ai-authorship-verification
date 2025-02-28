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

import json
from itertools import chain, zip_longest
from pathlib import Path
import random

import click
from more_itertools import interleave_longest
from tqdm import tqdm


@click.group(context_settings={'show_default': True})
def main():
    pass


def _scan_jsonl_ids(file, shuffle=True):
    ids = []
    for l in tqdm(open(file), desc='Scanning IDs', leave=False):
        ids.append(json.loads(l)['id'])
    if shuffle:
        random.shuffle(ids)
    return ids


@main.command(help='Sample a training and test split from JSONL files')
@click.option('-h', '--human', type=click.Path(dir_okay=False, exists=True), multiple=True,
              help='Human input JSONL file')
@click.option('-m', '--machine', type=click.Path(dir_okay=False, exists=True), multiple=True,
              help='Machine input JSONL file')
@click.option('-o', '--output-prefix', type=click.Path(dir_okay=False, exists=False),
              help='Output file prefix without suffix (default: data/sampled)')
@click.option('-s', '--scramble-ids', is_flag=True, help='Scramble text IDs')
@click.option('-i', '--max-imbalance', type=click.FloatRange(0, min_open=True), default=4.0,
              help='Maximum class imbalance of machine/human')
@click.option('--seed', type=int, default=42, help='Random seed')
def split(human, machine, output_prefix, scramble_ids, max_imbalance, seed):
    random.seed(seed)

    if not human or not machine:
        raise click.UsageError('Need at least one human and one machine input file.')

    human = sorted(human)
    machine = sorted(machine)
    human_ids = {h: _scan_jsonl_ids(h) for h in human}
    machine_ids = {m: _scan_jsonl_ids(m) for m in machine}

    h_it = interleave_longest(*human_ids.values())
    m_it = interleave_longest(*machine_ids.values())

    selected_human = set()
    selected_machine = set()
    selected_both = set()
    for h_id, m_id in zip_longest(h_it, m_it):
        if len(selected_machine) > 0 and len(selected_machine) / len(selected_human) > max_imbalance:
            break

        if h_id in selected_both:
            raise click.UsageError(f'Human ID {h_id} is not unique.')
        if m_id in selected_both:
            raise click.UsageError(f'Machine ID {m_id} is not unique.')

        if h_id is not None:
            selected_human.add(h_id)
            selected_both.add(h_id)
        if m_id is not None:
            selected_machine.add(m_id)
            selected_both.add(m_id)

    print(len(selected_human), len(selected_machine))
