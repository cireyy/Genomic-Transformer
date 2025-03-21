﻿# Genomic Transformer for Disease Prediction

This project implements a Transformer-based multi-task learning model for predicting diseases based on SNP genomic data and family history features. **The provided data is dummy data and should not be used for actual genomic analysis.**

## Data Source

This study is based on data from the **UK Biobank (UKBB)**, a large-scale biomedical database containing genetic and health information from over 500,000 participants. To access the original UKBB dataset, researchers must submit an application through the UK Biobank portal: [🔗 UK Biobank Access Application](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access)

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

The dataset consists of:

- **SNP Data: Synthetic SNP values per chromosome
- **Labels: Dummy binary disease indicators (6 diseases)
- **Family History: Dummy binary family history for each disease

## How to Run

### Train the Model

```bash
python main.py --config config.yaml
```

## Notes

- **The dataset provided here is synthetic and does not represent real biological data.**
- The original data used in this study is from **UK Biobank (UKBB)**.



