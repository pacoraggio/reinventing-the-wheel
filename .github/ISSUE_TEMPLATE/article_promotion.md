---
name: Article promotion
about: Track the development and promotion of a new article
title: "Article: [title here]"
labels: article
assignees: pacoraggio
---

## Article details

| Field | Value |
|---|---|
| Internal ID | `NN_article-slug` |
| Title | |
| Subtitle | |
| Category | `probability-and-statistics` / `statistical-illusions` / `football-in-data` |
| Target date | |

## Development checklist

### Notebook
- [ ] Notebook runs top to bottom without errors
- [ ] All cells are self-contained (no hidden state dependencies)
- [ ] No hardcoded local paths in `src/`
- [ ] No Greek letters in code or prose
- [ ] Outputs cleared before committing

### Quarto front matter
- [ ] `title` present and matches intended article title
- [ ] `description` present (one sentence, shown in listing)
- [ ] `author` set to "Paolo Coraggio"
- [ ] `date` set (controls sort order on site)
- [ ] `categories` set using approved slugs

### Data and source files
- [ ] Raw data in `data/raw/`
- [ ] Processed data in `data/processed/`
- [ ] `src/` scripts reviewed, no hardcoded paths

### Review
- [ ] Prose reads at the right level (technically literate, non-specialist)
- [ ] The *why* is explained, not just the *how*
- [ ] Real-world anchor is clear and consistent throughout
- [ ] Tone is consistent with project voice (curious practitioner, dry humour)
- [ ] No em-dashes

## Promotion checklist

- [ ] Article folder copied to public repo root (without `NN_` prefix)
- [ ] Notebook path added to `_quarto.yml` render list
- [ ] Notebook path added to `articles/index.qmd` contents list
- [ ] Article inventory updated in root `CLAUDE.md`
- [ ] `quarto render` run locally — site looks correct
- [ ] `quarto publish gh-pages` run — live site verified
- [ ] This issue closed

## Notes

<!-- Any decisions made during development, known limitations,
     or ideas for follow-up articles -->