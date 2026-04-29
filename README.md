# Reinventing the Wheel

*Not because it is easy, but because we don't know how to do it.*

A series of articles deconstructing data science methods and building
them from first principles using Python. Each article reconstructs a
tool — probability, statistics, hypothesis testing, measurement —
before connecting it to library implementations.

Published at [pacoraggio.github.io/reinventing-the-wheel](https://pacoraggio.github.io/reinventing-the-wheel/).

## Articles

**Probability and Statistics**
- **Measuring the Measurement** — samples, variation, and the surprisingly honest art of not knowing everything

**Statistical Illusions**
- **The Coin Has No Memory, But You Do** — streaks, probability illusions, and why 4 heads in a row is less surprising than you think

**Football in Data**
- **The Foul, the Whistle and the Doubt** — how to turn a pub argument into a statistical test
- **The Card, the Crowd and the Referee** — when the data stops behaving, the tools have to change
- **The Foul, the Card and the Gap** — not all fouls are created equal

## Setup

Conda environment: `rtw` (Python 3.11)

```bash
conda env create -f environment.yml
conda activate rtw
```

Built with [Quarto](https://quarto.org). To render locally:

```bash
quarto render
```

## Structure

```
drafts/                 articles under development (private repo only)
archive/                legacy notebooks, kept for reference
articles/               Quarto listing page
img/                    site images
_quarto.yml             Quarto configuration
environment.yml         conda environment
```