# Adherence Analysis Project

This project analyzes patient adherence data from both active and passive monitoring in a clinical trial studying Multiple System Atrophy (MSA), Parkinson's Disease (PD), and Progressive Supranuclear Palsy (PSP).

## Project Overview

The project performs comprehensive adherence analysis including:
- **Active Monitoring**: Analysis of patient gait test completion
- **Passive Monitoring**: Analysis of wearable sensor data (worn time)
- **Correlation Analysis**: Relationships between active/passive adherence, SUS scores, and disease severity
- **Statistical Analysis**: Cross-disease comparisons and temporal changes

## Project Structure

```
Adherence Project/
├── data/                                    # Raw data files (create this folder)
│   ├── active_monitoring.xlsx              # Active monitoring test data
│   ├── passive_monitoring.xlsx             # Passive sensor wearing time data
│   └── RCT_Clinical_Data.xlsx              # Clinical data with multiple sheets
│
├── active_monitoring/                       # Active monitoring analysis
│   ├── active_monitoring.py                # Main active adherence analysis
│   ├── sus_adherence_correlation.py        # Active adherence vs SUS correlation
│   └── [output files generated here]
│
├── passive_monitoring/                      # Passive monitoring analysis
│   ├── passive_monitoring.py               # Main passive adherence analysis
│   ├── passive_sus_adherence_correlation.py # Passive adherence vs SUS correlation
│   ├── week1/                              # Week 1 outputs (auto-created)
│   └── week8/                              # Week 8 outputs (auto-created)
│
├── active_passive_adherence_correlation.py  # Cross-monitoring correlation analysis
├── adherence_severity_simple_average.py     # Adherence vs severity analysis
└── README.md                                # This file
```

## Required Data Files

### Create a `data/` folder and place these files in it:

#### 1. `active_monitoring.xlsx`
- **Sheet1**: Overview of active monitoring data with columns:
  - `PATID`: Patient ID
  - `Week`: Week identifier (week_1, week_8)
  - `report_test_1`, `report_test_2`, `report_test_3`: Test completion status
  - Test status codes:
    - `*` = Completed test
    - `**` = Incomplete test
    - `***` = No test performed
    
- **Correct_files**: Files marked as correct for analysis

#### 2. `passive_monitoring.xlsx`
- Contains sensor wearing time data with columns:
  - `PATID`: Patient ID
  - `week`: Week number (1, 8)
  - `worn_time_LF`: Left foot worn time (hours)
  - `worn_time_RF`: Right foot worn time (hours)
  - `measurement_time`: Total recording time (hours)

#### 3. `RCT_Clinical_Data.xlsx`
This file should contain multiple sheets:

- **Visit2**: SUS scores for week 1
  - `PATID`: Patient ID
  - `Label`: Disease label (MSA, PD, PSP)
  - `SUS_total`: System Usability Scale total score

- **Visit4**: Patient demographics and labels
  - `PATID`: Patient ID
  - `Label`: Disease label

- **Visit5**: SUS scores for week 8
  - `PATID`: Patient ID
  - `SUS_total`: System Usability Scale total score

- **Severity**: Disease severity scores
  - `PATID`: Patient ID
  - Disease-specific severity columns (e.g., UMSARS, UPDRS, PSPRS)

## How to Use

### Prerequisites

Install required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-posthocs openpyxl
```

### Running the Analysis

Follow this order for complete analysis:

#### Step 1: Active Monitoring Analysis
```bash
cd active_monitoring
python active_monitoring.py
```
**Outputs:**
- `active_test_behavior_summary.csv`: Adherence rates by patient
- Various statistical and visualization files
- Charts showing test completion patterns

#### Step 2: Active SUS Correlation
```bash
python sus_adherence_correlation.py
```
**Outputs:**
- `active_adherence_with_sus_scores.csv`: Enhanced dataset with SUS scores
- `active_sus_correlations_overall.csv`: Overall correlation results
- `active_sus_correlations_by_disease.csv`: Disease-specific correlations

#### Step 3: Passive Monitoring Analysis
```bash
cd ../passive_monitoring
python passive_monitoring.py
```
**Outputs:**
- `week1/week1_validity_rate.csv`: Week 1 adherence rates
- `week8/week8_validity_rate.csv`: Week 8 adherence rates
- `week1/week1_detailed_metrics.csv`: Detailed wearing metrics
- `week8/week8_detailed_metrics.csv`: Detailed wearing metrics
- Statistical analysis files and visualizations

#### Step 4: Passive SUS Correlation
```bash
python passive_sus_adherence_correlation.py
```
**Outputs:**
- `passive_adherence_with_sus_scores.csv`: Enhanced dataset with SUS scores
- `passive_sus_correlations_overall.csv`: Overall correlation results
- `passive_sus_correlations_by_disease.csv`: Disease-specific correlations

#### Step 5: Active-Passive Correlation Analysis
```bash
cd ..
python active_passive_adherence_correlation.py
```
**Outputs:**
- `active_passive_correlations_overall.csv`: Cross-monitoring correlations
- `active_passive_correlations_by_disease.csv`: Disease-specific correlations
- `active_passive_adherence_combined.csv`: Combined dataset
- Visualization plots

#### Step 6: Adherence-Severity Analysis
```bash
python adherence_severity_simple_average.py
```
**Outputs:**
- `adherence_severity_simple_average_correlations.csv`: Correlation results
- `simple_average_adherence_scores.csv`: Computed adherence scores
- Correlation plots

## Key Concepts

### Adherence Metrics

**Active Adherence**: Percentage of tests completed out of 21 total tests per week (3 tests/day × 7 days)

**Passive Adherence**: Percentage of valid days where sensors were worn ≥8 hours (out of 7 days)

**Simple Average Adherence**: Mean of active and passive adherence

### Excluded Patients

The following patients are excluded from analyses due to dropout or screening failure:
- pat001, pat005, pat014, pat117, pat120, pat124, pat127, pat131
- pat133, pat134, pat154, pat212, pat233, pat238, pat314
- PAT402, pat403, pat407, PAT408, PAT411, pat412, PAT414

### Disease Groups

- **MSA**: Multiple System Atrophy
- **PD**: Parkinson's Disease
- **PSP**: Progressive Supranuclear Palsy

## Analysis Features

- **Adherence Calculation**: Computes adherence rates for active and passive monitoring
- **Statistical Testing**: Kruskal-Wallis tests, Wilcoxon signed-rank tests, post-hoc analyses
- **Correlation Analysis**: Spearman and Pearson correlations
- **Patient Padding**: Handles missing data by padding with zeros
- **Visualization**: Generates plots for distributions, correlations, and comparisons
