# Mechanistic Journey Through Othello-GPT

**From Hidden Board States to Rule Circuits**

This repository is the source for a student-friendly technical book about a mechanistic interpretability investigation of Othello-GPT. The book is built with MkDocs Material and is intended to preserve both polished explanations and the research memory needed for future Codex/ChatGPT sessions to continue the work.

The project should distinguish established results, strong evidence, moderate evidence, hypotheses, and open questions. New scientific claims should be recorded in `docs/research/research_log.md` before they are turned into book prose.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.

## Build

```bash
mkdocs build
```

The static site is written to `site/`.

## Deployment

GitHub Pages deployment is configured in `.github/workflows/deploy-docs.yml`. On pushes to `main`, the workflow installs dependencies, builds the MkDocs site, and publishes it with GitHub Pages.

## Research Notebooks

Research notebooks should live in `notebooks/` or be linked from `notebooks/README.md`. When an experiment produces a new result, update:

- `docs/research/research_log.md`
- `docs/research/experiment_index.md`
- `docs/research/findings_snapshot.md` if the result changes the status of a claim

## Contribution Convention

Before changing scientific claims in chapters, contributors should first add or update an entry in `docs/research/research_log.md`. Book chapters should cite the relevant notebook section, figure, table, or research-log entry.

