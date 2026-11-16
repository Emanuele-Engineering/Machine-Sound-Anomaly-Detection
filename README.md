# Anomaly Detection on Machine Acoustic Data

Unsupervised anomaly detection on acoustic emissions from industrial machines,
comparing a classical **feature-based** approach with **one-class
classification** approaches. Every detection rule and model is fitted on
**healthy training data only** — anomalous samples are never used for training
or threshold calibration.

## Highlights

- **Feature-based detection** on real machine sounds (pump / valve):
  exploratory analysis of raw signals, FFT spectra and spectrograms;
  statistical and spectral features (mean, variance, skewness, kurtosis,
  max-abs, spectral skewness) with percentile thresholds fitted on healthy
  data; alarm overlap (Jaccard) analysis across features.
- **Classification-based detection**: MEL-spectrogram features, a dense
  autoencoder scored by reconstruction error, One-Class SVM and Isolation
  Forest on bottleneck features, and a PCA baseline.
- **Extension — feature fusion**: handcrafted features combined with learned
  (bottleneck / PCA) features under robust median/MAD scaling, benchmarked
  across detectors with threshold calibration on healthy data.

## Results (pump_04 machine, test set)

| Approach                                        | AUC    |
| ----------------------------------------------- | ------ |
| Autoencoder reconstruction error                | F1 = 0.616 / Acc = 0.695 |
| Bottleneck + One-Class SVM                      | 0.597  |
| Bottleneck + Isolation Forest                   | 0.608  |
| PCA (64 components) + One-Class SVM             | 0.895  |
| PCA (64 components) + Isolation Forest          | 0.872  |

Within the fusion study, the handcrafted + PCA combination reaches the best
AUC (0.882 with Isolation Forest), and PCA + One-Class SVM the best
threshold-calibrated F1 at the 95th-percentile operating point (0.50).
PCA-compressed spectrograms clearly dominate the bottleneck features for
this machine — a useful reminder that a learned low-dimensional space is not
automatically the most discriminative one.

## Repository structure

```
├── anomaly-detection-acoustic-data.ipynb   # full analysis (Part 1 / Part 2 + feature-fusion extension)
├── README.md
└── data/                                   # not committed — see below
```

## Data

The datasets are **not redistributed** with this repository. The notebook
expects the following local layout:

```
data/
├── toy/toy_data.npz                     # synthetic demonstration dataset
├── machines/pump/{train,test}/*.wav     # machine-sound recordings
├── machines/valve/{train,test}/*.wav
└── mimii/dev_data_pump_04/ ...          # labeled data, MIMII format
    dev_data_valve_00/ ...
```

Each machine folder holds `train/` (healthy recordings only) and `test/`
(healthy + abnormal recordings) sub-folders of WAV files. The labeled data
follows the public MIMII machine-sound benchmark format
(`dev_data_<machine>/<machine_type>/{train,test}/*.wav`).

## Dependencies

`numpy`, `pandas`, `scipy`, `librosa`, `torch`, `scikit-learn`, `seaborn`,
`matplotlib`, `matplotlib-venn`, `tqdm`.
