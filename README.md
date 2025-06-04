# Genomic Transformer for Disease Prediction

This project implements a Transformer-based multi-task learning model for predicting diseases based on SNP genomic data and family history features. **The provided data is dummy data and should not be used for actual genomic analysis.**

## Abstract

Current genome-wide association studies provide valuable insights into the genetic basis of ischaemic stroke (IS) risk. However, polygenic risk scores, the most widely used method for genetic risk prediction, have notable limitations due to their linear nature and inability to capture complex, non-linear interactions among genetic variants. While deep neural networks offer advantages in modelling these complex relationships, the multifactorial nature of IS and the influence of modifiable risk factors present additional challenges for genetic risk prediction.

To address these challenges, we propose a Chromosome-wise Multi-task Genomic (MetaGeno) framework that utilizes genetic data from IS and five related diseases. The framework includes a chromosome-based embedding layer to model local and global interactions among adjacent variants, enabling a biologically informed approach. Incorporating multi-disease learning further enhances predictive accuracy by leveraging shared genetic information.

Among various sequential models tested, the Transformer demonstrated superior performance and outperformed other machine learning models and PRS baselines, achieving an AUROC of 0.809 on the UK Biobank dataset. Risk stratification identified a 2-fold increased stroke risk (HR = 2.14; 95% CI: 1.81–2.46) in the top 1% risk group, with a nearly 5-fold increase in those with modifiable risk factors such as atrial fibrillation and hypertension. Finally, the model was validated on the diverse All of Us dataset (AUROC = 0.764), highlighting ancestry and population differences while demonstrating effective generalization.

## Data Source

This study is based on genomic data from the **UK Biobank (UKBB)**, a large-scale biomedical database containing genetic and health information from over 500,000 participants. To access the original UKBB dataset, researchers must submit an application through the UK Biobank portal: [🔗 UK Biobank Access Application](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access)

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



