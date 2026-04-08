# Cerebrovascular Signal Processing Pipeline in Parkinson’s Disease

This repository provides a reproducible Python pipeline for the analysis of cerebrovascular and hemodynamic signals in Parkinson’s disease, focusing on temporal response dynamics and key physiological variables derived from transcranial Doppler and arterial blood pressure recordings.

## Features

- Signal segmentation based on event markers (START/END)
- Baseline extraction (20-second window prior to task onset)
- Extraction of physiological variables:
  - Cerebral blood flow velocity (MCAv)
  - Mean arterial pressure (MAP)
  - Heart rate (HR)
  - Cerebrovascular conductance index (CVCi = MCAv / MAP)
- Peak detection and Time-to-Peak (TTP) calculation relative to task onset
- Optional frequency-domain analysis (Transfer Function Analysis)

## Methods

The pipeline includes both time-domain and frequency-domain analyses. Transfer Function Analysis (TFA) is implemented as an optional module using SciPy-based signal processing methods, following established guidelines for cerebral autoregulation assessment.

## Author

Michelle Salabert  
PhD Candidate in Neurology and Neuroscience
