# Low-Cost Motor Vibration Monitoring and Analysis System

**Undergraduate Mechanical Engineering Minor Project | Institute of Engineering, Tribhuvan University — Paschimanchal Campus | August 2026**

An experimental, low-cost platform for recording three-axis motor vibration using an **ADXL345 accelerometer**, **ESP32**, and **Python**. The project demonstrates an end-to-end embedded data acquisition and offline signal-analysis workflow: **sensor → microcontroller → serial CSV recording → preprocessing → time/frequency-domain features → visual reports**.

> **Project status — working prototype; preliminary experimental validation.** The hardware–software acquisition and analysis pipeline operated during the project and was functionally tested by the team. The accompanying college report documents tests under multiple motor operating conditions. **The experimental data and the report's quantitative interpretations have not been independently validated to research/industrial diagnostic standards.** A measurement-range configuration error also caused clipping in some high-vibration recordings. See [Data quality and limitations](#data-quality-and-limitations) before interpreting plots or numbers.

## What we built

- **Three-axis acquisition:** ADXL345 MEMS accelerometer attached to a DC motor housing, interfaced with an ESP32 over SPI.
- **Buffered sampling and transfer:** Sensor FIFO buffering and serial transfer of X/Y/Z readings to a computer.
- **Data recording:** Python-based data capture and storage in CSV format for offline processing.
- **Signal processing:** Unit conversion/calibration, axis-wise DC removal, FFT, Welch power spectral density (PSD), and spectrograms.
- **Feature extraction:** RMS, peak and peak-to-peak amplitude, standard deviation, crest factor, kurtosis, skewness, and frequency-domain descriptors.
- **Visualization/reporting:** Time traces, frequency spectra, comparative plots, and generated PDF analysis reports.

The project was designed as an **accessible experimental condition-monitoring platform**—not as a calibrated commercial vibration meter or an autonomous fault classifier.

## Architecture

```text
 DC motor ── mechanical coupling ── DC generator ── variable resistive load rig
     │
     └── Motor-housing vibration
              │
          ADXL345 (X, Y, Z)
              │ SPI
           ESP32 + FIFO
              │ Serial / USB
         Python CSV logger
              │
     Calibration / preprocessing
              │
    FFT • PSD • spectrogram • features
              │
     Plots and analysis PDF reports
```

### Main hardware

| Component | Purpose |
| --- | --- |
| ADXL345 | Three-axis acceleration measurements |
| ESP32 | Sensor configuration, SPI acquisition, buffering and serial data output |
| DC motor–generator test rig | Experimental rotating-machine platform |
| Adjustable resistive loading rig | Changes generator electrical load and, in turn, the opposing mechanical torque |
| Computer with Python | Recording, processing and visualization |

The submitted report describes a nominal **1600 Hz ADXL345 output data-rate configuration** and an **approximately 1559 Hz estimated acquisition rate**. The latter should be treated as an estimated effective recording rate until verified against sample timing and buffering behavior; CSV row counts alone do not establish an exact sensor sampling clock.

## Experiments

The prototype was used to study vibration under several deliberately varied conditions:

| Test | What was changed | Interpretation status |
| --- | --- | --- |
| Normal operation | Motor operated without an intentionally introduced test condition | Preliminary baseline; environmental variability possible |
| Generator loading | Resistive electrical load adjusted through the coupled generator | Comparative observation; electrical and speed measurements were not synchronized with vibration |
| Sudden load change | Loading-rig knob adjusted rapidly to produce a torque-demand transient | Transient response observed; exact adjustment count/timing not independently logged |
| Mechanical looseness | Mechanical condition deliberately altered | Measured vibration changes observed; degree of looseness not independently quantified |
| Imbalance / other tested conditions | See original college report for the experimental descriptions | Treat interpretations as preliminary until source recordings and conditions are reconciled |

A change in the measured vibration signature is **not, by itself, proof of a particular fault mechanism**. Speed, operating load, sensor mounting, nearby machinery and sampling artifacts can also affect the signal.

## Functional validation vs. measurement validation

We distinguish two different things:

**1. Functional system validation — completed during the project.** The team assembled and operated the ADXL345–ESP32 data collection system, transferred readings for logging, and generated signal-processing outputs and plots. This supports the claim that the **end-to-end prototype worked** for acquisition and analysis.

**2. Quantitative experimental validation — incomplete.** We did not establish a traceable acceleration calibration, independently verify per-sample timestamps, or establish rigorous fault-specific diagnostic accuracy. Some recordings were captured under less-than-ideal experimental conditions. Results in the original report should be read as **exploratory observations**, not guaranteed ground truth.

## Data quality and limitations

### Important configuration issue: ±2 g measurement range

The ADXL345 supports selectable **±2 g, ±4 g, ±8 g and ±16 g** measurement ranges. We configured the sensor to **±2 g** during acquisition, underestimating the peak acceleration that could occur in mechanical-looseness and sudden-load tests. Some high-vibration recordings reached the measurement limits, causing **clipping/saturation**.

This is a **measurement-range selection error**, not simply an error in numerical resolution. In the ADXL345's *full-resolution* mode, changing the range retains approximately **3.9 mg/LSB** scaling while providing additional headroom (subject to the device's bandwidth and noise limitations).

**Fix for future acquisitions:** Set the appropriate ADXL345 measurement range in firmware (for example ±4 g, ±8 g or up to ±16 g as warranted), verify the scaling and valid register configuration, and repeat measurements while checking for clipping. **Updating firmware cannot recover peaks already clipped in historical CSV files**. Reacquisition is necessary for accurate quantitative measurements. A wider range alone also does not solve timestamp uncertainty, vibration-bandwidth limitations or mounting problems.

### Additional qualifications

- **Sampling time:** Original CSVs do not provide an independent timestamp for every sensor sample. A nominal/estimated rate was used for spectral calculations, so reported frequency positions and derived RPM estimates require confirmation.
- **Clipped transients:** Saturation affects peaks, crest factors, distributions, and potentially spectra; high-vibration results are not precision measurements of the true acceleration.
- **Environmental and setup effects:** Vibration from other equipment, sensor attachment, motor speed and electrical load may have varied across runs.
- **Test labels and repeatability:** Recording labels and trial counts should be reconciled with original logs before reporting statistical conclusions; duplicated files must not be counted as independent trials.
- **Fault attribution:** No synchronized tachometer, torque or generator current measurements were captured to isolate mechanical causes conclusively.
- **Offline pipeline:** The documented analysis is offline; it is not an industrial real-time alarm system or a validated predictive-maintenance model.

**Data-sharing policy:** All shared recordings are retained as historical experimental records. Potentially affected recordings should be labeled *preliminary / possibly clipped*, rather than selectively removed or presented as validated reference datasets.

## Software and reproduction

The original submission contains the ESP32 and Python source in its appendices. For an easily reproducible repository, extract the **actual tested source files** from the original development environment rather than copying formatted snippets from Word/PDF. The published programs should be checked for compilation, hardware pin mapping and dependencies.

Suggested repository structure (adapt to the files you actually have):

```text
.
├── README.md
├── firmware/                 # Tested ESP32 / Arduino source
├── acquisition/              # Python serial CSV logger
├── analysis/                 # Calibration, FFT/PSD and feature scripts
├── data/                     # Raw records, with labels and caveats
├── figures/                  # Setup photos and selected outputs
├── reports/                  # Original minor-project report / analysis PDFs
└── requirements.txt          # Verified Python dependencies
```

The reported Python toolchain uses **NumPy, pandas, SciPy, Matplotlib**, and **ReportLab**, with the ESP32 developed in an Arduino-compatible environment. Exact package versions, board core and serial settings should be added once tested on a clean machine. **No one-command installation or reproduction procedure is claimed here until it has actually been verified.**

### Recommended reproduction checklist

1. Match ADXL345–ESP32 SPI wiring to the **tested firmware**, not merely the report's wiring table; resolve any GPIO/interrupt discrepancies.
2. Verify accelerometer range, full-resolution mode, output data rate, FIFO status and reading scale.
3. Record a stationary-sensor baseline and verify gravitational orientation and offset treatment.
4. Check received samples for missing records, duplicate/stale samples and sensor-range saturation.
5. Process recorded XYZ data with the analysis scripts and check units, FFT normalization, PSD units and known-frequency signals.
6. Repeat tests at a known shaft speed with a tachometer or other independent timing reference.

## Original project report

The college report is included for transparency and to document the team's original design, experimental setup, code appendices and analysis. It was prepared under a limited submission timeline and **has not been retrospectively rewritten to conceal uncertainties**. Some numerical comparisons, operating-condition labels or explanatory claims may require correction when the datasets are fully reconciled. In cases of inconsistency, treat the **original data, verified code and this README's limitations** as the appropriate starting points for further review—not the report as conclusive fault-diagnosis evidence.

## Future improvements

- Select measurement range based on observed acceleration and log clipping flags.
- Add sample sequencing/timestamps and confirm delivered vs. actual sensor sample rate.
- Use synchronized shaft-speed and generator voltage/current measurements.
- Improve sensor mounting repeatability and document motor/load settings for every run.
- Build a clean, reproducible processing pipeline with unit tests and sample input data.
- Re-run controlled experiments with adequate repetitions and confidence intervals.
- Evaluate diagnostic features only after the measurement chain is validated.

## Project team

**Aryan Parajuli · Aayush Bhandari · Kiran Bhusal · Sushil Nyaupane**  
Supervisor: **Er. Ghanshyam Aryal**  
Department of Mechanical and Automobile Engineering, IOE Paschimanchal Campus, Tribhuvan University, Nepal.

## Acknowledgment and reuse

This is an educational prototype and research-learning artifact. Please do not use its historical recordings or preliminary condition labels as a validated industrial fault-diagnosis benchmark. Add a license only after the team agrees on how to release the jointly developed code and documentation.
