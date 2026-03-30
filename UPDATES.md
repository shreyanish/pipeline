# Codebase Update Log

Weekly improvements to the rPPG pipeline for SpO2 estimation.

---

## January (Pre-Jan Work Included)
*Commits from Nov 21 – Nov 28 crunched here*

- **Initial pipeline built** — `pipeline.py` established as the core script: MediaPipe face detection, RGB channel extraction from facial ROIs, Green channel-based rPPG signal, and SpO2 feature extraction (R-ratio, AC/DC)
- **CHROM & ICA algorithms added** — Integrated two new rPPG algorithms (CHROM and ICA) alongside the existing Green channel method
- **31 facial ROI support** — Expanded the pipeline from a single ROI to 31+ distinct facial landmark regions for comprehensive regional comparison
- **Test scripts added** — `test_pipeline_quick.py` and `test_validation.py` added for sanity checking pipeline outputs
- **Tasks roadmap established** — `tasks.txt` seeded with upcoming goals: additional algorithms, deep learning models, more evaluation metrics, and modular refactoring

---

## Week of Jan 12
*Commits: `352567a`, `14228a5`*

- **Major modular refactor** — Monolithic `pipeline.py` (290+ lines) broken out into focused, reusable modules:
  - `config.py` — Central configuration and constants
  - `roi_definitions.py` — All 31+ facial region landmark definitions
  - `rppg_algorithms.py` — Algorithm implementations (Green, CHROM, ICA)
  - `signal_extraction.py` — Frame-by-frame signal extraction logic
  - `feature_extraction.py` — SpO2 feature calculation (R-ratio, AC/DC, etc.)
- **Tasks updated** — Roadmap refined to reflect completed items and new targets (10–15 total algorithms, DL models, expanded metrics)

---

## Week of Feb 16
*Commit: `a009dd7`*

- **GPU-accelerated algorithms via PyTorch** — `rppg_pytorch.py` added (~498 lines); all rPPG algorithms reimplemented using PyTorch tensors for GPU compatibility, cutting per-video processing time dramatically
- **Blind evaluation metrics** — `evaluation.py` added (~230 lines) with signal quality metrics: SNR, correlation, peak detection, and frequency-domain analysis
- **Batch processing pipeline** — `batch_process.py` added (~446 lines); CLI tool to run all algorithms across all videos and all 31 ROIs in a single pass, outputting combined `.npy` result files
- **Evaluation runner** — `run_evaluation.py` added; orchestrates evaluation across the full batch output
- **Algorithm expansion** — `rppg_algorithms.py` grew significantly (+314 lines) to support the full suite of 11 algorithms
- **ROI definitions expanded** — `roi_definitions.py` updated (+141 lines) with refined landmark groupings
- **Test suite updated** — `test_ensemble.py` and `test_evaluation.py` added for validating the new components

---

## Week of Feb 20
*Commits: `85dbffc`, `ce434f9`, `b237ead`*

- **Per-algorithm `.npy` files** — `batch_process.py` now saves individual `.npy` result files for each of the 11 algorithms (in addition to the combined file), making it easy to load and analyze results per-method
- **Per-algorithm signal plots** — Script auto-generates 11 plots (one per algorithm) showing rPPG signals across all 31 facial ROIs for a randomly selected video; saved automatically on each batch run
- **Workflow documentation** — `batch_process_workflow.md` written (~157 lines) covering: CLI usage, execution flow, algorithm list, and output file descriptions
- **Large file cleanup** — `.gitignore` updated to exclude bulky output files (`.npy` arrays, generated plots) from version control
