# Cerebrovascular Signal Processing Pipeline in Parkinson’s Disease

> ⚠️ **Note**  
> This pipeline is under active development as part of an ongoing PhD project in Neurology/Neuroscience.  
> Core steps (data import and baseline extraction) have been internally validated against LabChart DataPad outputs, while other components (e.g., peak detection and derived metrics) are still under evaluation and refinement.

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
- Frequency-domain analysis (e.g., Transfer Function Analysis) is planned for future implementation.

## Methods

### Data Acquisition

Physiological signals were acquired using transcranial Doppler ultrasound and non-invasive arterial blood pressure monitoring. Input data consist of LabChart-exported `.mat` files containing continuous recordings of cerebral blood flow velocity (MCAv), mean arterial pressure (MAP), and heart rate (HR).

The pipeline supports multiple `.mat` file structures (standard and block formats) and automatically identifies relevant channels based on flexible name matching.

### Preprocessing

Signals are extracted and temporally aligned using their respective sampling rates. Continuous signals are resampled to a 1 Hz representation by computing mean values per second, enabling consistent temporal analysis across modalities.

Time vectors are converted to a common reference frame relative to task onset (t = 0 s).

### Event Detection and Segmentation

Task timing is determined using event markers extracted from LabChart comment fields. The pipeline supports:

- Explicit START/END markers (preferred)
- Single-event detection with fixed-duration assumption (60 s)
- Fallback window (20–80 s) when markers are unavailable

All signals are segmented relative to task onset using the following convention:

- Baseline window: −20 to 0 s  
- Task window: 0 to 60 s  

Only recordings with a minimum valid response window (≥30 s) are included in the analysis.

### Physiological Variables

The following variables are computed on a beat-to-beat averaged (1 Hz) basis:

- Cerebral blood flow velocity (MCAv)
- Mean arterial pressure (MAP)
- Heart rate (HR)
- Cerebrovascular conductance index (CVCi = MCAv / MAP)

Baseline values are defined as the mean within the −20 to 0 s window.

### Response Characterization

Cerebrovascular responses are quantified using:

- **Peak MCAv**: maximum value during the task window  
- **Time-to-Peak (TTP)**: time from task onset to peak MCAv  
- **Relative changes (%)**:
  - ΔMCAv%
  - ΔMAP%
  - ΔCVCi%

Relative changes are computed with respect to baseline values.

### Data Quality Control

The pipeline includes several robustness features:

- Automatic detection of missing or misnamed channels  
- Handling of heterogeneous `.mat` file structures  
- Validation of minimum response window duration  
- Error logging for problematic files  

Baseline MCAv and MAP values have been internally validated against LabChart DataPad outputs, showing good agreement. Other derived metrics (e.g., peak detection and TTP) are currently under evaluation.

### Output and Reproducibility

The pipeline generates:

- Structured result tables (.xlsx)
- Individual subject figures (time-series and response panels)
- Group-level summary statistics
- Publication-style figures

All processing steps are fully automated and reproducible, enabling scalable analysis across participants and experimental conditions.

### Frequency-Domain Analysis

Frequency-domain analysis (e.g., Transfer Function Analysis) is not currently implemented in this version of the pipeline but is planned for future development following established guidelines for cerebral autoregulation assessment.

## Author

Michelle Salabert  
PhD Candidate in Neurology and Neuroscience
