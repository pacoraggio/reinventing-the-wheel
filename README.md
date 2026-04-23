# Reinventing the Wheel

*Not because it is easy, but because we don't know how to do it.*

A series of articles deconstructing data science methods and building them from first principles using Python. Each article reconstructs a tool — probability, statistics, hypothesis testing, measurement — before connecting it to library implementations.

Published at [github.io](https://pacoraggio.github.io/reinventing-the-wheel/).

## Articles

- **The Coin Has No Memory, But You Do** — streaks, probability illusions, and why 4 heads in a row is less surprising than you think
- **Measuring the Measurement** — samples, variation, and the surprisingly honest art of not knowing everything

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
articles/           article listing page
population_parameter_and_sample_statistic/
probability_illusions_flips_in_a_row/
index.qmd           site home page
_quarto.yml         Quarto configuration
environment.yml     conda environment
```
