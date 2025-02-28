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

from functools import partial
import json
import logging
from multiprocessing import pool
import os
from pathlib import Path
import sys

import backoff
import click
import jinja2
import jsonschema
from openai import OpenAI, OpenAIError
from tqdm import tqdm

logger = logging.getLogger(__name__)

_SUM_TEMPLATES = ['news', 'essay', 'essay_obfs', 'essay_obfs_neighborhood', 'fiction_cont']


@click.group(context_settings={'show_default': True})
def main():
    pass


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
def _summarize(text: str, client: OpenAI, model_name: str, template: jinja2.Template):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': template.render(text=text)},
            {'role': 'user', 'content': text}
        ]
    )
    response = response.choices[0].message.content
    if response.startswith('```json'):
        response = response.strip('`')[len('json'):]
    return response.strip()


def _map_from_to_file(fnames, *args, fn, skip_existing=True, max_chars=None, **kwargs):
    """
    Call ``fn`` on tuples of input and output Path objects, reading input from one file and writing to another.
    """
    file_in, file_out = fnames
    if skip_existing and file_out.exists():
        return
    file_out.parent.mkdir(parents=True, exist_ok=True)

    result = fn(file_in.read_text()[:max_chars], *args, **kwargs)
    if not result:
        return
    file_out.write_text(result)


@main.command(help='Generate text summaries using OpenAI API')
@click.argument('prompt-template', type=click.Choice(_SUM_TEMPLATES))
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'summaries'))
@click.option('-k', '--api-key', type=click.Path(dir_okay=False, exists=True),
              help='File containing OpenAI API key (if not given, OPENAI_API_KEY env var must be set)')
@click.option('-n', '--assistant-name', default='pan-summarizer')
@click.option('-m', '--model-name', default='gpt-4o')
@click.option('-p', '--parallelism', default=10)
@click.option('-c', '--max-chars', default=8192,
              help='Maximum article length to send to OpenAI API in characters')
@click.option('-g', '--input-glob', default='**/*.txt', help='Input file glob')
def summarize(prompt_template, input_dir, output_dir, api_key, assistant_name, model_name,
              parallelism, max_chars, input_glob):
    if not api_key and not os.environ.get('OPENAI_API_KEY'):
        raise click.UsageError('Need one of --api-key or OPENAI_API_KEY!')

    client = OpenAI(api_key=open(api_key).read().strip() if api_key else os.environ.get('OPENAI_API_KEY'))

    env = jinja2.Environment(
        loader=jinja2.PackageLoader('pan25_genai_detection.dataset', 'prompt_templates')
    )
    template = env.get_template(f'sum_{prompt_template}.jinja2')
    input_dir = Path(input_dir)
    in_files = list(input_dir.rglob(input_glob))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_out_files = list((f, output_dir / f.relative_to(input_dir).with_suffix('.json')) for f in in_files)
    fn = partial(_map_from_to_file,
                 fn=_summarize,
                 max_chars=max_chars,
                 client=client,
                 model_name=model_name,
                 template=template)

    with pool.ThreadPool(processes=parallelism) as p:
        # noinspection PyStatementEffect
        [_ for _ in tqdm(p.imap(fn, in_out_files), desc='Generating summaries', unit='sum')]


@main.command(help='Validate LLM-generated JSON summary files', context_settings={'show_default': True})
@click.argument('schema', type=click.Choice(_SUM_TEMPLATES))
@click.argument('input_file', type=click.Path(dir_okay=False, exists=True), nargs=-1)
def validate(input_file, schema):
    if not input_file:
        raise click.UsageError('No input file specified')

    syntax_errors = []
    validation_errors = []
    schema = json.load(open(Path(__file__).parent / 'summary_schemas' / f'{schema}.json', 'r'))

    for fname in tqdm(input_file, desc='Validating JSON files', unit='files'):
        try:
            jsonschema.validate(instance=json.load(open(fname, 'r')), schema=schema)
        except json.JSONDecodeError as e:
            syntax_errors.append((e.msg, fname))
        except jsonschema.ValidationError as e:
            validation_errors.append((e.message, fname))

    if not syntax_errors and not validation_errors:
        click.echo('No errors.', err=True)
        sys.exit(0)

    if syntax_errors:
        click.echo('Syntax errors:', err=True)
        for e, f in sorted(syntax_errors, key=lambda x: x[1]):
            click.echo(f'  {f}: {e}', err=True)

    if validation_errors:
        click.echo('Validation errors:', err=True)
        for e, f in sorted(validation_errors, key=lambda x: x[1]):
            click.echo(f'  {f}: {e}', err=True)

    sys.exit(1)


@main.command(help='Combine source texts and validated summary JSON into JSONL')
@click.argument('sum_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('text_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-file', type=click.Path(dir_okay=False, exists=False), help='Output file',
              default=os.path.join('data', 'summaries', 'combined.jsonl'))
@click.option('-p', '--id-prefix', help='Prefix to add to text IDs')
@click.option('-g', '--summary-glob', default='**/*.json', help='Summary JSON file glob')
def combine(sum_dir, text_dir, output_file, id_prefix, summary_glob):
    sum_dir = Path(sum_dir)
    text_dir = Path(text_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as out:
        for s in tqdm(list(sum_dir.rglob(summary_glob)), desc='Combining texts and summaries'):
            s_rel = s.relative_to(sum_dir)
            t = text_dir / s_rel.with_suffix('.txt')
            text_id = str(s_rel.with_suffix(''))
            if id_prefix:
                text_id = os.path.join(id_prefix, text_id)
            article_data = {
                'id': text_id,
                'text': t.read_text().strip() if t.exists() else None,
                'summary': json.load(open(s, 'r'))
            }
            json.dump(article_data, out, ensure_ascii=False)
            out.write('\n')


if __name__ == "__main__":
    main()
