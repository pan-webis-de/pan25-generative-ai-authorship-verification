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

import os
from pathlib import Path
import typing as t

import click
import nltk
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig, set_seed, T5Tokenizer, T5ForConditionalGeneration


@click.group(context_settings={'show_default': True})
def main():
    """Paraphrase LLM texts to evade detection."""


@main.command()
@click.argument('input-dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', help='Output directory', default=os.path.join(
    'data', 'text-llm-paraphrased'))
@click.option('-m', '--model', help='Model name or path', default='kalpeshk2011/dipper-paraphraser-xxl')
@click.option('-d', '--device-map', help='Model device map', default='auto')
@click.option('--lex-diversity', help='Lexical diversity of the output',
              type=click.Choice(['0', '20', '40', '60', '80', '100']), default='60')
@click.option('--order-diversity', help='Order diversity of the output',
              type=click.Choice(['0', '20', '40', '60', '80', '100']), default='0')
@click.option('--prefix', help='Prompt prefix', default='')
@click.option('-i', '--sent-interval', help='Sentence span / interval',
              type=click.IntRange(1), default=3)
@click.option('-m', '--min-length', type=click.IntRange(1), default=300, help='Minimum length in tokens')
@click.option('-x', '--max-length', type=click.IntRange(1), default=512, help='Maximum output length')
@click.option('-k', '--top-k', type=click.IntRange(0), default=0, help='Top-k sampling (0 to disable)')
@click.option('-p', '--top-p', type=click.FloatRange(0, 1), default=0.75, help='Top-p sampling')
@click.option('-t', '--temperature', type=click.FloatRange(0, min_open=True), default=1.0,
              help='Model temperature')
@click.option('-f', '--flash-attn', is_flag=True,
              help='Use flash-attn 2 (must be installed separately)')
@click.option('-b', '--better-transformer', is_flag=True, help='Use BetterTransformer')
@click.option('-q', '--quantization', type=click.Choice(['4', '8']))
@click.option('--seed', type=int, default=42, help='Random seed')
def dipper(input_dir, output_dir, model, device_map, lex_diversity, order_diversity, prefix, sent_interval, min_length,
           max_length, top_k, top_p, temperature, flash_attn, better_transformer, quantization, seed):
    """
    Paraphrase an LLM text using DIPPER to evade detection.

    References:
    ===========
        Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023). Paraphrasing evades detectors of
        AI-generated text, but retrieval is an effective defense. In A. Oh, T. Naumann, A. Globerson, K. Saenko,
        M. Hardt, & S. Levine (Eds.), Advances in Neural Information Processing Systems 36 (NeurIPS 2023)
        (Vol. 36, pp. 27469–27500).
    """
    set_seed(seed)

    model_args: t.Dict[str, t.Any] = {'torch_dtype': torch.bfloat16}
    if flash_attn:
        model_args.update({'attn_implementation': 'flash_attention_2'})
    if quantization:
        # noinspection PyTypeChecker
        model_args.update({
            'quantization_config': BitsAndBytesConfig(
                **{f'load_in_{quantization}bit': True},
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4'
            )
        })

    lex_code = int(100 - int(lex_diversity))
    order_code = int(100 - int(order_diversity))

    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
    model = T5ForConditionalGeneration.from_pretrained(model, device_map=device_map)
    model.eval()
    if better_transformer:
        model = model.to_bettertransformer()

    generation_args = dict(
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    nltk.download('punkt_tab')

    user_prefix = prefix
    input_dir = Path(input_dir)
    input_files = input_dir.rglob('**/*.txt')
    for infile in tqdm(input_files, desc='Paraphrasing texts', unit='t'):
        out_file = Path(output_dir) / infile.relative_to(input_dir.parent)
        if out_file.exists():
            continue

        input_text = infile.read_text()
        output_text = []
        prefix = ' '.join(user_prefix.split())

        for paragraph in input_text.strip().split('\n\n'):
            output_paragraph = []
            sentences = nltk.sent_tokenize(paragraph)

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = ' '.join(sentences[sent_idx:sent_idx + sent_interval])
                paragraph = f'lexical = {lex_code}, order = {order_code}'
                if prefix:
                    paragraph += f' {prefix}'
                paragraph += f' <sent> {curr_sent_window} </sent>'
                input_tokens = tokenizer([paragraph], return_tensors='pt', max_length=max_length).to(model.device)
                with torch.inference_mode():
                    outputs = model.generate(**input_tokens, **generation_args)
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += ' ' + outputs[0]
                output_paragraph.append(outputs[0])

            output_text.append(' '.join(output_paragraph))
            prefix = output_paragraph[-1]

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text('\n\n'.join(output_text))
