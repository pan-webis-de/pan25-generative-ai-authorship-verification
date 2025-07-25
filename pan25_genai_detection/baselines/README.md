# PAN'25 "Voight-Kampff" Generative AI Authorship Verification Baselines

LLM detection baselines for the "Voight-Kampff" Generative AI Authorship Verification Baselines at PAN 2025.

## Usage

If installed via ``pip``, you can just run the baselines with

```
pan25-baseline BASELINENAME INPUT_FILE OUTPUT_DIRECTORY
```

Use `--help` on any subcommand for more information:

```console
$ pan25-baseline --help
Usage: pan25-baseline [OPTIONS] COMMAND [ARGS]...

  PAN'25 Generative AI Authorship Verification baselines.

Options:
  --help  Show this message and exit.

Commands:
  binoculars  PAN'25 baseline: Binoculars.
  ppmd        PAN'25 baseline: Compression-based cosine.
  tfidf       PAN'25 baseline: TF-IDF SVM.
```

If you want to run the baselines via Docker, use:

```
docker run --rm --gpus=all -v INPUT_FILE:/val.jsonl -v OUTPUT_DIRECTORY:/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    BASELINENAME /val.jsonl /out
```

`INPUT_FILE` is the test / validation input data (JSONL format). `OUTPUT_DIRECTORY` is the output
directory for the predictions.

Concrete example:

```
docker run --rm --gpus=all -v $(pwd)/val.jsonl:/val.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    tfidf /val.jsonl /out
```

The option ``--gpus=all`` is needed only for Binoculars.
