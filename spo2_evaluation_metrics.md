# Evaluation Metrics for SpO2 Estimation

## 1. Per‑Sample Accuracy Metrics

**Mean Absolute Error (MAE, %SpO2)**  
Used in: 2022–2024 non‑contact facial SpO2 papers, including contactless face‑camera works, algorithmic benchmarks, and CL‑SPO2Net. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Use as the primary accuracy metric, since it is directly interpretable as average absolute deviation in %SpO2 and is consistently reported in recent SpO2 estimation literature.

**Root Mean Square Error (RMSE, %SpO2)**  
Used in: Camera‑based SpO2 and algorithmic benchmark studies, typically alongside MAE. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Use when you want to penalize larger errors more strongly and highlight safety‑critical outliers in performance.

**Mean Error / Bias (signed error, %SpO2)**  
Used in: Camera‑based SpO2 and spectroscopy ML papers comparing against clinical oximeters. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Report to quantify systematic over‑ or under‑estimation relative to reference, especially when discussing compliance with clinical accuracy limits.

**Standard Deviation (SD) of Error (%SpO2)**  
Used in: Same studies that report bias, often combined to form an RMS‑style accuracy measure. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Use with bias to describe dispersion of errors and to support computation of ISO‑style accuracy metrics.

**\(A_{RMS}\) (Root‑Mean‑Square Accuracy, %SpO2)**  
Used in: Non‑contact camera and SpO2 validation works referencing ISO 80601‑2‑61. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Use when explicitly positioning your method relative to pulse oximeter standards, since ISO expresses accuracy via RMS error in the 70–100% range.

## 2. Agreement and Distribution Metrics

**Pearson Correlation Coefficient (r)**  
Used in: 2023 facial‑analysis case study and other ML‑based SpO2 regression works. [fruct](https://fruct.org/publications/volume-33/fruct33/files/Ham.pdf)
When to choose: Include to show how well predicted SpO2 tracks the trend of ground‑truth values, beyond absolute error magnitude.

**Coefficient of Determination (R²)**  
Used in: ML‑on‑PPG and spectroscopy‑based SpO2 regression papers. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Use when framing SpO2 as a regression problem for an ML audience and you want a familiar “variance explained” statistic.

**Bland–Altman Bias and Limits of Agreement (LoA)**  
Used in: Camera‑based SpO2 and vital‑sign estimation studies targeting clinical readers. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Use for clinical‑style agreement analysis, to show bias and the spread of differences across the full SpO2 range.

**Error Distribution (Histogram/KDE)**  
Used in: 2023 facial‑video SpO2 works inspecting the “shape” of predicted samples. [fruct](https://fruct.org/publications/volume-33/fruct33/files/Ham.pdf)
When to choose: Use to demonstrate that errors are concentrated and roughly symmetric, or to reveal heavy‑tail behavior and outliers.

**Percentage of Predictions within ±k %SpO2**  
Used in: Benchmark and validation‑oriented contactless SpO2 papers. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Report for intuitive summaries such as “X% of readings within ±2% of reference”, which are easy to interpret for both technical and clinical audiences.

## 3. Threshold / Classification Metrics

**Accuracy (thresholded SpO2)**  
Used in: Works that cast SpO2 or hypoxemia detection as a binary task around a clinical cutoff. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Use when you define a threshold (e.g., SpO2 < 90%) and need an overall proportion of correctly classified normal vs hypoxemic samples.

**Sensitivity (Recall for Hypoxemia)**  
Used in: Hypoxemia‑oriented camera and ML vital‑sign studies. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Emphasize when false negatives are critical and you must reliably detect low‑SpO2 events.

**Specificity**  
Used in: Same threshold‑based evaluations as sensitivity. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Report alongside sensitivity to quantify how well you avoid false alarms for hypoxemia.

**Area Under ROC Curve (AUROC)**  
Used in: Algorithmic benchmark and ML SpO2 works that explore multiple thresholds. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Use when you want threshold‑independent classification performance for hypoxemia detection or early‑warning tasks.

**Area Under Precision–Recall Curve (AUPRC)**  
Used in: ML‑style vital‑sign and spectroscopy papers with class‑imbalanced low‑SpO2 events. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Prefer over AUROC when hypoxemia episodes are rare, to better characterize performance on the positive class.

## 4. Range‑ and Condition‑Specific Metrics

**MAE / RMSE per SpO2 Range**  
Used in: 2022+ systematic/benchmark works noting degradation at low SpO2. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774849/)
When to choose: Report when you need to show reliability in clinically critical low ranges (e.g., 70–90%) separately from high ranges.

**MAE per Condition (Lighting, Motion, Head Rotation, etc.)**  
Used in: CL‑SPO2Net and recent deep facial‑video methods. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/38391599/)
When to choose: Use to quantify robustness across realistic nuisance conditions such as illumination change, motion, and pose variation.

**Per‑Subject MAE / Error Distribution**  
Used in: Deep facial‑video benchmark studies on contactless SpO2. [behealth](https://www.behealth.hk/wp-content/uploads/2023/12/An-Algorithmic-Benchmark-for-Contactless-SPO2-Measurement-from-Facial-Videos.pdf)
When to choose: Include when generalization across individuals matters and you want to expose subject‑specific failures.

**Temporal Stability (Variance Under Steady State)**  
Used in: Real‑time SpO2 monitoring and facial‑video estimation works. [ieda.ust](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2024-SPO2.pdf)
When to choose: Use when targeting continuous monitoring and you want to show that predictions are not excessively jittery during physiologically stable periods.

## 5. Model‑Level Metrics

**Inference Time / Computational Complexity**  
Used in: Algorithmic benchmark and deep learning SpO2 facial‑video papers. [ieda.ust](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2024-SPO2.pdf)
When to choose: Report when comparing architectures or arguing that your approach is deployable on mobile or edge devices.

**Robustness to Missing Frames / Occlusion**  
Used in: Ablation and robustness analyses for video‑based SpO2 estimation. [ieda.ust](https://www.ieda.ust.hk/dfaculty/so/pdf/Cheng-et-al-2024-SPO2.pdf)
When to choose: Evaluate when your deployment environment may suffer from dropped frames, partial face visibility, or compression artifacts.
