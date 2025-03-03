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
from itertools import zip_longest
import os
from pathlib import Path
import re
import random
import string
import uuid

import click
from more_itertools import interleave_longest
from tqdm import tqdm


@click.group(context_settings={'show_default': True})
def main():
    pass


def _scan_jsonl_ids(file, min_text_length=400, shuffle=True):
    ids = []
    for l in tqdm(open(file), desc='Scanning IDs', leave=False):
        j = json.loads(l)
        if j.get('text') and len(j['text']) >= min_text_length:
            ids.append(j['id'])
    if shuffle:
        random.shuffle(ids)
    return ids

@main.command(help='Convert folder of PAN\'24 texts to PAN\'25 folder structure')
@click.argument('test_ids', type=click.Path(dir_okay=False, exists=True))
@click.argument('human_dir', type=click.Path(file_okay=False, exists=True))
@click.argument('machine_dir', type=click.Path(file_okay=False, exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False, exists=False),
              default=os.path.join('data', 'pan24-converted'), help='Output base dir')
def convert_pan24(test_ids, human_dir, machine_dir, output_dir):
    human_dir = Path(human_dir)
    human_files = list(human_dir.rglob('**/*.txt'))
    machine_dir = [Path(m) for m in machine_dir]
    machine_files = [list(m.rglob('**/*.txt')) for m in machine_dir]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_ids = set(l.strip() for l in open(test_ids).readlines())
    for files, c in [(human_files, 'human'), *[(m, 'machine') for m in machine_files]]:
        for f in tqdm(files, desc=f'Copying {c} files'):
            f_rel = f.relative_to(f.parents[-len(f.parents) + 2])
            if f_rel.relative_to(f_rel.parents[-2]).with_suffix('').as_posix() in test_ids:
                dataset = 'pan24-test'
            else:
                dataset = 'pan24-train'

            f_rel_out = Path(dataset) / f_rel.relative_to(f_rel.parents[-2])
            f_rel_out = (f_rel.parents[-2] if c != 'human' else 'human') / f_rel_out

            outfile = output_dir / f_rel_out
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(f.read_bytes())


@main.command(help='Convert LLM text output folder to single JSONL file')
@click.argument('text_dir', type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option('-m', '--model-name', help='Model name (default: input dir name)')
@click.option('-o', '--output-dir', type=click.Path(file_okay=False, exists=False),
              help='Output file dir', default=os.path.join('data', 'text-llm-jsonl'))
@click.option('-p', '--id-prefix', help='Prefix to add to text IDs (default: MODEL_NAME)')
@click.option('-g', '--text-glob', default='**/*.txt', help='Summary text file glob')
def text2jsonl(text_dir, model_name, output_dir, id_prefix, text_glob):
    model_name_override = model_name
    output_dir = Path(output_dir)

    for td in tqdm(text_dir, desc='Processing input dirs'):
        td = Path(td).resolve()
        for t in [d for d in td.iterdir() if d.is_dir()]:
            model_name = model_name_override or t.parent.name
            dataset_name = t.name
            opath = output_dir / dataset_name / f'{dataset_name}-{model_name}.jsonl'
            opath.parent.mkdir(parents=True, exist_ok=True)
            with open(opath, 'w') as out:
                for f in tqdm(t.rglob(text_glob), desc='Converting text dir into JSONL', leave=False):
                    f_id = Path(dataset_name)
                    if model_name != 'human':
                        f_id /= model_name
                    f_id /= f.resolve().relative_to(t).with_suffix('')
                    if id_prefix:
                        f_id = Path(id_prefix) / f_id

                    record_data = {
                        'id': f_id.as_posix(),
                        'text': f.read_text().strip(),
                        'model': model_name or f_id.parents[-2].name
                    }
                    json.dump(record_data, out, ensure_ascii=False)
                    out.write('\n')


_WORD_CHAR_RE = re.compile(r'[\w,]')


def _fixup_midsentence_end(text):
    # Return unmodified if text doesn't end with word character
    if not _WORD_CHAR_RE.match(text.strip()[-1]):
        return text

    # Otherwise discard last paragraph if it isn't too long
    t1, t2 = text.strip().rsplit('\n\n', 1)
    if len(t2) < 500 and len(t1) > len(t2) * 3:
        return t1

    # Last attempt: try to find punctuation (other than comma) in last paragraph
    i = len(t2) // 2
    punct = set(string.punctuation.replace(',', ''))
    while i < len(t2) and t2[i] not in punct:
        i += 1
    return t1 + t2[:i + 1]


@main.command(help='Sample a balanced subset of texts as single JSON file')
@click.option('-h', '--human', type=click.Path(dir_okay=False, exists=True), multiple=True,
              help='Human input JSONL file')
@click.option('-m', '--machine', type=click.Path(dir_okay=False, exists=True), multiple=True,
              help='Machine input JSONL file')
@click.option('-o', '--output-file', type=click.Path(dir_okay=False, exists=False),
              help='Output file', default=os.path.join('data', 'sampled', 'sampled.jsonl'))
@click.option('-s', '--scramble-ids', is_flag=True, help='Scramble text IDs')
@click.option('--id-salt', help='Salt of ID scrambling', default='KLdCre0Vd')
@click.option('-i', '--max-imbalance', type=click.FloatRange(0, min_open=True), default=4.0,
              help='Maximum class imbalance of machine/human')
@click.option('--min-length', type=click.FloatRange(0, min_open=True), default=800,
              help='Minimum text length in characters')
@click.option('--no-end-fixup', is_flag=True, help='Don\'t fixup generations ending mid-sentence')
@click.option('--genre', help='Optional genre key to add to all texts')
@click.option('--seed', type=int, default=42, help='Random seed')
def sample_balanced(human, machine, output_file, scramble_ids, id_salt, max_imbalance, min_length,
                    no_end_fixup, genre, seed):
    random.seed(seed)

    if not human or not machine:
        raise click.UsageError('Need at least one human and one machine input file.')

    human = sorted(human)
    machine = sorted(machine)
    human_ids = {h: _scan_jsonl_ids(h, min_text_length=min_length, shuffle=True) for h in human}
    machine_ids = {m: _scan_jsonl_ids(m, min_text_length=min_length, shuffle=True) for m in machine}

    h_it = interleave_longest(*human_ids.values())
    m_it = interleave_longest(*machine_ids.values())
    selected_human = set()
    selected_machine = set()
    for h_id, m_id in tqdm(zip_longest(h_it, m_it), desc='Sampling IDs'):
        if len(selected_machine) > 0 and len(selected_machine) / len(selected_human) > max_imbalance:
            break

        if h_id in selected_human or h_id in selected_machine:
            raise click.UsageError(f'Human ID {h_id} is not unique.')
        if m_id in selected_machine or m_id in selected_human:
            raise click.UsageError(f'Machine ID {m_id} is not unique.')

        if h_id is not None:
            selected_human.add(h_id)
        if m_id is not None:
            selected_machine.add(m_id)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    id_file = None
    if scramble_ids:
        id_file = output_file.parent / (output_file.with_suffix('').name + '-orig-ids.jsonl')
        id_file = id_file.open('w')

    def _lines_it():
        for f in human:
            yield from open(f)
        for f in machine:
            yield from open(f)

    with output_file.open('w') as out:
        for l in tqdm(_lines_it(), desc='Generating sample'):
            record_data = json.loads(l)
            if record_data['id'] in selected_human:
                record_data['label'] = 0
                record_data['model'] = 'human'
            elif record_data['id'] in selected_machine:
                record_data['label'] = 1
            else:
                continue

            if not no_end_fixup:
                record_data['text'] = _fixup_midsentence_end(record_data['text'])

            if genre:
                record_data['genre'] = genre

            if scramble_ids and id_file:
                orig_id = record_data['id']
                record_data['id'] = str(uuid.uuid5(uuid.NAMESPACE_OID, id_salt + orig_id))
                json.dump({'id': record_data['id'], 'orig_id': orig_id}, id_file)
                id_file.write('\n')

            json.dump(record_data, out, ensure_ascii=False)
            out.write('\n')

    if id_file:
        id_file.close()

    click.echo(f'Output written to {output_file}. Don\'t forget to shuffle lines before distribution.')
