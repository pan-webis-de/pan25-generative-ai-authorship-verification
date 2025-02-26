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

import re
import string
from functools import partial
import json
import logging
from multiprocessing import pool, set_start_method
import os
from pathlib import Path

import backoff
import click
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
import jinja2
import markdown
from openai import OpenAI, OpenAIError
from resiliparse.extract import html2text
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.preview.generative_models import FinishReason, GenerativeModel, HarmCategory, HarmBlockThreshold


logger = logging.getLogger(__name__)
set_start_method('spawn')
set_seed(42)

_PROMPT_TEMPLATES = ['essay', 'essay_obfs_neighborhood', 'news_article', 'news_article_short', 'kaggle_paraphrase']


def _generate_instruction_prompt(article_data, template_name):
    """
    Generate an instruction prompt for generating an article from the given source article data.
    """

    target_paragraphs = article_data['text'].count('\n\n')
    target_words = round(int(len(re.split(r'\s+', article_data['text']))) + 9, -1)

    env = jinja2.Environment(
        loader=jinja2.PackageLoader('pan25_genai_detection.dataset', 'prompt_templates')
    )
    try:
        template = env.get_template(f'gen_{template_name}.jinja2')
    except jinja2.TemplateNotFound:
        raise click.UsageError(f'No such template: {template_name}.')
    return template.render(article_data=article_data, target_paragraphs=target_paragraphs, target_words=target_words)


def _apply_chat_template(tokenizer, model_type, messages):
    chat_template = tokenizer.chat_template

    if not chat_template:
        if 'alpaca' in tokenizer.name_or_path.lower():
            chat_template = (
                'Below is an instruction that describes a task. '
                'Write a response that appropriately completes the request.\n\n'
                '### Instruction:\n'
                '{% for message in messages -%}\n'
                '{{ message["content"] }}\n'
                '{% endfor %}\n'
                '{% if add_generation_prompt %}\n'
                '### Response:\n'
                '{% endif %}')
        elif model_type == 'llama':
            chat_template = (
                '{% for message in messages -%}\n'
                '### {{ message["role"].capitalize() }}:\n'
                '{{ message["content"] }}\n'
                '{% endfor %}\n'
                '{% if add_generation_prompt %}\n'
                '### Assistant:\n'
                '{% endif %}')
        else:
            chat_template = (
                'Task description:\n'
                '{% for message in messages -%}\n'
                '{{ message["content"] }}\n'
                '{% endfor %}\n'
                '{% if add_generation_prompt %}\n'
                'Response:\n'
                '{% endif %}')

    return tokenizer.apply_chat_template(
        messages, chat_template=chat_template, return_tensors='pt', add_generation_prompt=True)


def _iter_jsonl_files(in_files):
    for f in in_files:
        for l in open(f, 'r'):
            yield f, json.loads(l)


def _map_records_to_files(infile_and_record, *args, fn, out_dir: Path, skip_existing: bool = True,
                          out_file_suffix: str = '.txt', **kwargs):
    """
    Take a tuple of ``(topic name, parsed JSON record)``, apply ``fn`` on the JSON and write its output to
    individual text files based on the record's topic and ID under ``out_dir``.
    """

    in_file, record = infile_and_record
    if os.path.sep not in record['id']:
        out_dir /= in_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (out_dir / record['id']).with_suffix(out_file_suffix)

    if skip_existing and out_file.exists():
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = fn(record, *args, **kwargs)
    except Exception as e:
        logger.error('Failed to generate article: %s', str(e))
        logger.exception(e)
        return

    if not result:
        return

    open(out_file, 'w').write(result)


# noinspection PyStatementEffect
def _generate_articles(input_files, gen_fn, parallelism=1):
    it = _iter_jsonl_files(input_files)
    it = ((Path(f), a) for f, a in it)

    if parallelism == 1:
        [_ for _ in tqdm(map(gen_fn, it), desc='Generating articles', unit=' articles')]
        return

    with pool.ThreadPool(processes=parallelism) as p:
        [_ for _ in tqdm(p.imap(gen_fn, it), desc='Generating articles', unit=' articles')]


def _clean_text_quirks(text):
    """Clean up some common LLM text quirks."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=3)
def _openai_gen_article(article_data, client: OpenAI, model_name: str, prompt_template: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': _generate_instruction_prompt(article_data, prompt_template)}
        ]
    )
    response = html2text.extract_plain_text(markdown.markdown(response.choices[0].message.content)).strip()
    return _clean_text_quirks(response)


@backoff.on_exception(backoff.expo, GoogleAPIError, max_tries=3)
def _vertexai_gen_article(article_data, model_name: str, prompt_template: str, **model_args):
    prompt = _generate_instruction_prompt(article_data, prompt_template)

    if 'gemini' in model_name:
        model = GenerativeModel(model_name=model_name)
    elif model_name.startswith('chat-'):
        model = ChatModel.from_pretrained(model_name)
    else:
        model = TextGenerationModel.from_pretrained(model_name)

    citations_censored = False
    sex_censored = False
    max_tries = 4
    for _ in range(max_tries):
        if isinstance(model, GenerativeModel):
            # HarmBlockThreshold.BLOCK_NONE no longer possible after recent update without
            # being an invoiced billing customer
            response = model.generate_content(
                prompt,
                generation_config=model_args,
                safety_settings={h: HarmBlockThreshold.BLOCK_ONLY_HIGH for h in HarmCategory})
            candidates = response.candidates

        elif isinstance(model, ChatModel):
            chat = model.start_chat(context=prompt, **model_args)
            candidates = chat.send_message('Assistant response:').candidates

        else:
            candidates = model.predict(prompt, **model_args).candidates

        # Handle hard-coded safety filters
        filtered = not candidates
        citations_filtered = False
        if candidates and hasattr(candidates[0], 'finish_reason'):
            filtered |= candidates[0].finish_reason not in [FinishReason.STOP, FinishReason.MAX_TOKENS]
            citations_filtered = candidates[0].finish_reason == FinishReason.RECITATION

        # Probably hard-coded input or output blocking. Amend prompt and try again with higher temperature.
        if filtered:
            if citations_filtered and not citations_censored:
                prompt += ('\nAvoid direct citation of sources that could be used for misinformation. '
                           'Do not cite or mention any medical or pharmaceutical sources.')
                citations_censored = True
            elif not sex_censored:
                prompt = prompt.replace('sex', '&&&')
                prompt += '\nPhrase your response in a non-harmful way suitable for the general public.'
                sex_censored = True
            model_args['temperature'] = min(1.0, model_args.get('temperature', 0.5) + 0.1)
            continue

        # Success
        break
    else:
        raise RuntimeError(f'Generation failed for {article_data["id"]}')

    response = candidates[0].content.text if hasattr(candidates[0], 'content') else candidates[0].text
    if sex_censored:
        response = response.replace('&&&', 'sex')

    response = html2text.extract_plain_text(markdown.markdown(response)).strip()
    return _clean_text_quirks(response)


def _huggingface_chat_gen_article(article_data, model, tokenizer, prompt_template, headline_only=False, **kwargs):
    messages = [{'role': 'user', 'content': _generate_instruction_prompt(article_data, prompt_template)}]
    model_inputs = _apply_chat_template(tokenizer, model.config.model_type, messages).to(model.device)

    for _ in range(3):
        generated_ids = model.generate(
            model_inputs,
            do_sample='penalty_alpha' not in kwargs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs)

        response = tokenizer.decode(generated_ids[0][len(model_inputs[0]):], skip_special_tokens=True)

        # Strip markdown
        response = html2text.extract_plain_text(markdown.markdown(response)).strip()
        response = _clean_text_quirks(response)

        # Retry if response empty
        if not response:
            continue

        if headline_only:
            response = response.split('\n', 1)[0]       # Take only first line
        elif response[-1] in string.ascii_letters:
            trim_len = response.rfind('\n\n')
            if len(response) - trim_len > 500:
                trim_len = response.rfind('. ') + 1
            response = response[:trim_len]                          # Some models tend to stop mid-sentence

        return response.rstrip()

    return ''


@click.group(context_settings={'show_default': True})
def main():
    pass


@main.command(help='Generate articles using the OpenAI API')
@click.argument('prompt-template', type=click.Choice(_PROMPT_TEMPLATES))
@click.argument('input-jsonl', type=click.Path(dir_okay=False, exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'text-llm'))
@click.option('-n', '--outdir-name', help='Output subdirectory name (defaults to model name)')
@click.option('-k', '--api-key', type=click.Path(dir_okay=False, exists=True),
              help='File containing OpenAI API key (if not given, OPENAI_API_KEY env var must be set)')
@click.option('-m', '--model-name', default='gpt-4o')
@click.option('-p', '--parallelism', default=5)
def openai(prompt_template, input_jsonl, output_dir, outdir_name, api_key, model_name, parallelism):
    if not api_key and not os.environ.get('OPENAI_API_KEY'):
        raise click.UsageError('Need one of --api-key or OPENAI_API_KEY!')

    output_dir = Path(output_dir) / (outdir_name if outdir_name else model_name.lower())
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=open(api_key).read().strip() if api_key else os.environ.get('OPENAI_API_KEY'))

    fn = partial(
        _map_records_to_files,
        fn=_openai_gen_article,
        prompt_template=prompt_template,
        out_dir=output_dir,
        client=client,
        model_name=model_name)
    _generate_articles(input_jsonl, fn, parallelism)


@main.command(help='Generate articles using the VertexAI API')
@click.argument('prompt-template', type=click.Choice(_PROMPT_TEMPLATES))
@click.argument('input-jsonl', type=click.Path(dir_okay=False, exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'text-llm'))
@click.option('-m', '--model-name', default='gemini-pro')
@click.option('-n', '--outdir-name', help='Output subdirectory name (defaults to model name)')
@click.option('-p', '--parallelism', default=5)
@click.option('-t', '--temperature', type=click.FloatRange(0, 1), default=0.6,
              help='Model temperature')
@click.option('-x', '--max-output-tokens', type=click.IntRange(0, 1024), default=1024,
              help='Maximum number of output tokens')
@click.option('-k', '--top-k', type=click.IntRange(1, 40), default=None,
              help='Top-k sampling')
@click.option('--top-p', type=click.FloatRange(0, 1), default=0.95,
              help='Top-p sampling')
def vertexai(prompt_template, input_jsonl, output_dir, model_name, outdir_name, parallelism, **kwargs):
    output_dir = Path(output_dir) / (outdir_name if outdir_name else model_name.replace('@', '-').lower())
    output_dir.mkdir(parents=True, exist_ok=True)

    fn = partial(
        _map_records_to_files,
        fn=_vertexai_gen_article,
        prompt_template=prompt_template,
        out_dir=output_dir,
        model_name=model_name,
        **kwargs)

    try:
        _generate_articles(input_jsonl, fn, parallelism)
    except GoogleAuthError as e:
        raise click.UsageError('Authentication error:\n' + str(e))


@main.command(help='Generate texts using a Huggingface chat model')
@click.argument('model_name')
@click.argument('prompt-template', type=click.Choice(_PROMPT_TEMPLATES))
@click.argument('input-jsonl', type=click.Path(dir_okay=False, exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False),
              default=os.path.join('data', 'text-llm'), help='Output directory')
@click.option('-n', '--outdir-name', help='Output subdirectory name (defaults to model name)')
@click.option('-d', '--device', type=click.Choice(['auto', 'cuda', 'cpu']), default='auto',
              help='Select device to run model on')
@click.option('-m', '--min-length', type=click.IntRange(1), default=350,
              show_default=True, help='Minimum length in tokens')
@click.option('-x', '--max-new-tokens', type=click.IntRange(1), default=1000,
              show_default=True, help='Maximum new tokens')
@click.option('-s', '--decay-start', type=click.IntRange(1), default=500,
              show_default=True, help='Length decay penalty start')
@click.option('--decay-factor', type=click.FloatRange(1), default=1.01,
              show_default=True, help='Length decay penalty factor')
@click.option('-k', '--top-k', type=click.IntRange(0, min_open=True), default=0,
              show_default=True, help='Top-k sampling (0 to disable)')
@click.option('-p', '--top-p', type=click.FloatRange(0, 1, min_open=True), default=0.9,
              show_default=True, help='Top-p sampling')
@click.option('-a', '--penalty-alpha', type=click.FloatRange(0, 1, min_open=True), default=0.0,
              show_default=True, help='Contrastive search penalty')
@click.option('-t', '--temperature', type=click.FloatRange(0, min_open=True), default=0.6,
              show_default=True, help='Model temperature')
@click.option('-f', '--flash-attn', is_flag=True,
              help='Use flash-attn 2 (must be installed separately)')
@click.option('-b', '--better-transformer', is_flag=True, help='Use BetterTransformer')
@click.option('-q', '--quantization', type=click.Choice(['4', '8']))
@click.option('--trust-remote-code', is_flag=True, help='Trust remote code')
def huggingface_chat(model_name, prompt_template, input_jsonl, output_dir, outdir_name, device, quantization, top_k,
                     top_p, penalty_alpha, decay_start, decay_factor, better_transformer, flash_attn, headlines_only,
                     trust_remote_code, **kwargs):

    model_name_out = model_name
    model_args = {
        'torch_dtype': torch.bfloat16
    }
    if flash_attn:
        model_args.update({'attn_implementation': 'flash_attention_2'})
    if quantization:
        model_args.update({
            'quantization_config': BitsAndBytesConfig(**{
                f'load_in_{quantization}bit': True,
                f'bnb_{quantization}bit_compute_dtype': torch.bfloat16
            })
        })
        model_name_out = model_name + f'-{quantization}bit'

    model_name_out = model_name_out.replace('\\', '/').rstrip('/')
    if '/' in model_name_out:
        model_name_out = '-'.join(model_name_out.split('/')[-2:])
    output_dir = Path(output_dir) / (outdir_name if outdir_name else model_name_out.lower())

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, trust_remote_code=trust_remote_code, **model_args)
        if better_transformer:
            model = model.to_bettertransformer()
    except Exception as e:
        raise click.UsageError('Failed to load model: ' + str(e))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_cache=False, padding_side='left', trust_remote_code=trust_remote_code)

    kwargs.update(dict(
        model=model,
        tokenizer=tokenizer,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p if penalty_alpha > 0 and top_k > 1 else None,
        penalty_alpha=penalty_alpha,
        exponential_decay_length_penalty=(decay_start, decay_factor)
    ))

    if headlines_only:
        del kwargs['min_length']
        del kwargs['exponential_decay_length_penalty']
        kwargs['max_new_tokens'] = 60
        prompt_template = 'headline_chat'

    fn = partial(_map_records_to_files, fn=_huggingface_chat_gen_article,
                 prompt_template=prompt_template, out_dir=output_dir, **kwargs)

    _generate_articles(input_jsonl, fn)


if __name__ == "__main__":
    main()
