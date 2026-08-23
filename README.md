# Automated Proofreading of Digitally Reconstructed Neural Morphology Enhances Accuracy, Scalability, and Standardization

This repository contains the source code, machine-learning workflows, validation data, and supplementary resources associated with the study:
The project provides an automated framework for quality control (QC), standardization, structural correction, morphometric analysis, and dendritic relabeling of neuronal morphology reconstructions represented in SWC format.
The framework combines deterministic morphology-processing procedures with machine-learning-based dendritic classification and is implemented within a containerized, cloud-deployable architecture.

---

## Overview

Digital neuronal reconstructions are increasingly generated at scales that make manual proofreading and correction difficult to sustain. This project provides an integrated computational workflow for processing SWC-formatted neuronal
reconstructions with minimal manual intervention.

The framework supports:
- SWC structural validation and standardization
- Detection and correction of overlapping or duplicate points
- Detection and removal of spurious side branches
- Correction of non-positive radius values
- Detection of anomalously long parent-child connections
- Reconnection of disconnected neuronal components
- Morphometric analysis
- Automated PNG visualization of processed reconstructions
- Machine-learning-based apical/basal dendritic relabeling of pyramidal neurons
- Confidence scoring and uncertainty quantification
- Automated logging and traceability of processing operations

The application uses a React-based front end and Flask-based backend, with containerization through Docker and cloud deployment on Amazon Web Services (AWS).

---

## Processing Architecture

The framework contains three primary automated processing workflows.

### 1. SWC Standardization and Quality Control

The standardization workflow validates SWC structure and detects and corrects common reconstruction irregularities, including:

- overlapping points
- duplicate spatial points
- spurious included side branches
- non-positive radius values
- formatting inconsistencies
- invalid parent-child relationships

Processing operations and detected irregularities are recorded in log and summary files to provide traceability.

Morphometric measurements are generated using L-Measure, and processed SWC files can be rendered as PNG images for visual inspection.

### 2. Long-Connection Detection and Topology Repair

SWC reconstructions are represented as directed graphs using NetworkX.

For each parent-child connection, Euclidean distance is calculated. Connection lengths are evaluated relative to the distribution within each reconstruction, and anomalously long connections are detected using a user-defined statistical
threshold (six standard deviations by default).

When an anomalous connection is removed, the resulting disconnected components are evaluated relative to the soma-containing main tree. Candidate reconnection points are identified using spatial proximity and graph topology,
and eligible detached components are iteratively reconnected.

The workflow records processing statistics including connection-length distributions, thresholds, detected long connections, removed edges, and reconnected components.

### 3. Automated Pyramidal Dendritic Relabeling

The dendritic-labeling workflow identifies apical and basal dendritic components in pyramidal neurons.

Each SWC reconstruction is represented as a morphology graph containing spatial coordinates, radius values, and parent-child relationships. The soma serves as the anatomical root for graph traversal and dendritic-tree extraction.

The original soma representation is preserved; automated correction and standardization procedures operate on the branched neuronal arbor rather than modifying detailed soma geometry.

Morphological descriptors used to characterize candidate dendritic trees include:

- node count
- number of bifurcations
- maximum Euclidean distance from the soma
- maximum root-to-tip path length
- total arbor length
- Sholl-derived radial complexity
- principal spatial-axis orientation

A neural classifier implemented in PyTorch is used to classify dendritic
identity.

---

## Neuron Selection and Dataset Construction

Neural reconstructions used for development of the dendritic relabeling model
were retrieved programmatically from NeuroMorpho.Org through its public
application programming interface (API).

The initial candidate population was obtained by querying pyramidal neurons
using the NeuroMorpho.Org API endpoint:

https://neuromorpho.org/api/neuron/select?q=cell_type:pyramidal

The API response was used to systematically retrieve available pyramidal-cell
records and associated reconstruction metadata.

Neurons were selected according to the following protocol:

1. **Cell-type selection**  
   Reconstructions returned by the NeuroMorpho.Org API query for    `cell_type:pyramidal` were identified as the initial     candidate population.

2. **SWC reconstruction availability**  
   Candidate neuronal reconstructions were required to be available in the SWC representation used by the processing and     machine-learning pipeline.

3. **Soma identification**  
   Each SWC reconstruction was parsed to identify the presence of a soma (SWC type 1), which served as the anatomical        root for subsequent processing. Reconstructions lacking an identifiable soma were excluded from the dataset.

4. **Apical-dendrite selection**  
   Pyramidal neuron reconstructions containing a single identifiable apical dendritic tree were retained for supervised      model development. Reconstructions not meeting this criterion were excluded from the model-development dataset.

5. **Final dataset construction**  
   Application of these selection criteria yielded 20,500 pyramidal neuron reconstructions** for model development.

6. **Dataset partitioning**  
   Prior to partitioning, the complete selected file list was randomly shuffled using: np.random.shuffle(all_files) to       reduce ordering bias. The resulting dataset was divided into:
   - **80% training**
   - **10% validation**
   - **10% test**

7. **Reproducibility**  
   Reconstruction identifiers/file lists and associated labeling information are provided with the publicly available        study resources to facilitate independent examination and reproduction of the dataset-selection
   procedure.

---

## Machine-Learning Model

The dendritic relabeling model was developed to distinguish apical, basal, and other neuronal components based on morphology-derived features.
Training was performed using the Adam optimizer with adaptive learning-rate scheduling based on validation loss.
Message Passing Interface (MPI) was used to distribute computational tasks,
including:
- SWC parsing
- feature extraction
- model training and evaluation

The model-development workflow was evaluated over **ten independent runs** to assess performance stability and generalization.

---

## Model Evaluation

Model performance was evaluated using multiple complementary metrics, including:
- accuracy
- weighted precision
- weighted recall
- weighted F1-score
- cross-entropy loss
- confidence score
- uncertainty score

Across the independent runs, validation and test accuracies remained approximately **99.5%**.

Reported performance included:
| Metric | Value |
| --- | ---: |
| Mean accuracy | ~99.5% |
| Weighted precision | 0.978 |
| Weighted recall | 0.977 |
| Weighted F1-score | 0.977 |
| Mean confidence score | 0.8680 |
| Mean uncertainty score | 0.1320 |

The model incorporates the biological constraint of a single apical dendritic tree for the pyramidal neurons represented in the model-development dataset.
All ten training runs were completed in approximately 25 hours using high-performance computing (HPC) resources provided by the Office of Research Computing at George Mason University.

Per-neuron predictions, node sequences, and evaluation metrics are provided with the supplementary study resources.

---

## Independent Cross-Dataset Validation

An independent blinded cross-dataset evaluation was performed to assess mode generalization beyond the data available during model development.

Following the original manuscript submission, NeuroMorpho.Org was queried for newly released pyramidal neurons containing an apical dendrite. These reconstructions were unavailable during model development.

The independent dataset contained:
- **341 previously unseen neuronal reconstructions**
- reconstructions contributed by **17 independent laboratories**
- no reconstructions represented in the original training, validation, or test datasets

For blinded evaluation, the existing apical dendrite annotations were removed before inference. The previously trained model was then tasked with identifying the apical dendrite in each reconstruction without retraining.

Results:

- **341 neurons evaluated**
- **336 correctly classified**
- **98.53% independent cross-dataset accuracy**

This evaluation provides an additional assessment of model performance on newly acquired data originating from independent sources.

---

## Quantitative Morphometric Validation

Structural preservation following automated correction was quantitatively evaluated using **500 randomly sampled NeuroMorpho.Org reconstructions** representing a broad diversity of independent laboratories, animal species,
brain regions, cell types, experimental methods, and reconstruction systems.

Original and standardized reconstructions were compared using multiple morphometric measurements and complementary statistical analyses.

Representative results include:

| Morphometric feature | Pearson r | ICC | Mean % difference |
| --- | ---: | ---: | ---: |
| Volume | 0.9999 | 0.9999 | 0.57% |
| Bifurcation count | 0.9982 | 0.9982 | 1.12% |
| Branch count | 0.9984 | 0.9984 | 0.85% |
| Depth | 0.9954 | 0.9739 | — |

Paired statistical testing demonstrated no statistically significant differences between the original and standardized measurements for the representative morphometric parameters reported in the manuscript.

These analyses assess preservation of key morphometric characteristics while structural inconsistencies are corrected.

---

## Input Format

The primary input format is **SWC**, the widely used representation for digitally reconstructed neuronal morphology.

Each SWC record contains standard fields describing:
- node identifier
- morphological type
- x coordinate
- y coordinate
- z coordinate
- radius
- parent node identifier

Other neuronal morphology formats may be incorporated after conversion to SWC, for example through the `xyz2swc` converter.

---

## Output

Depending on the selected processing workflow, outputs may include:
- standardized SWC reconstructions
- structurally corrected SWC reconstructions
- dendritically relabeled SWC reconstructions
- processing logs
- quality-control summaries
- morphometric measurements
- model predictions
- confidence and uncertainty measurements
- PNG morphology visualizations
- validation datasets and statistical analyses

---

## Core Technologies

The framework uses several open-source and cloud technologies, including:
- Python
- PyTorch
- NetworkX
- MPI
- Flask
- React
- Docker
- Amazon Web Services (AWS)
- L-Measure
- StdSwc
- NeuroMorpho.Org processing utilities

---

## Web Application

The executable web-based pipeline is available at:

https://swcstandardization.computational-neuromorpho.org

The web interface allows SWC files to be submitted to the automated processing workflows and provides processing-status information and generated outputs.

---

## Repository Organization

The repository contains resources associated with the automated SWC standardization and dendritic-relabeling framework.

### Machine Learning

[`ML/`](ML/)

Contains machine-learning-related code, model-development resources, evaluation outputs, and supplementary documentation.

### Supplementary Methods

[`ML/Supplementary_Methods.pdf`](ML/Supplementary_Methods.pdf)

Contains additional methodological details for the machine-learning workflow.

### Morphometric Validation

[`groundtruth_vs_standardized/`](groundtruth_vs_standardized/)

Contains the original-versus-standardized morphometric comparison datasets and associated statistical analyses used for quantitative validation.

Additional directories contain processing scripts and resources associated with SWC standardization, structural correction, dendritic relabeling, and visualization.

---

## Public Datasets

### GCN Training and Evaluation Data

The training and evaluation datasets used for development and assessment of the dendritic relabeling model are publicly available through Zenodo:
https://doi.org/10.5281/zenodo.20534897

The deposited resources support independent examination of model development and evaluation.

### Morphometric Validation Data

The datasets used to compare original and standardized neuronal reconstructions are available in:

[`groundtruth_vs_standardized/`](groundtruth_vs_standardized/)

### Source Neural Reconstructions

Source neuronal reconstructions were obtained from the publicly available NeuroMorpho.Org repository:

https://neuromorpho.org

---

## Documentation and Supplementary Materials

- [Supplementary Methods for the machine-learning component](ML/Supplementary_Methods.pdf)
- [Training and evaluation datasets (Zenodo)](https://doi.org/10.5281/zenodo.20534897)
- [Morphometric validation datasets and statistical analyses](groundtruth_vs_standardized/)

---

## Reproducibility

The project was designed to support reproducible large-scale processing through:
- standardized SWC inputs and outputs
- programmatic NeuroMorpho.Org data retrieval
- explicit neuron-selection criteria
- documented dataset partitioning
- containerized execution
- controlled software dependencies
- centralized logging
- processing summaries
- publicly available source code
- publicly available training and evaluation data
- publicly available validation datasets
- repeated model training and evaluation
- independent cross-dataset testing

Docker containerization provides a consistent runtime environment across supported computing platforms and deployment environments.

---

## Limitations

The current implementation has several important limitations.

1. The processing framework operates on SWC-formatted neuronal reconstructions. Other formats require conversion to SWC       before processing.

2. The soma is used as the anatomical root for graph traversal and dendritic tree extraction. The framework does not         reconstruct or modify the detailed three-dimensional geometry of the soma.

3. The dendritic relabeling model was developed using pyramidal neurons containing a single apical dendrite. Pyramidal       neurons containing multiple apical dendrites will require additional specialized training data and
   potentially modification of the machine-learning module.

4. Although quantitative morphometric validation demonstrated strong agreement between original and standardized             reconstructions, the automated corrections have not yet been systematically benchmarked against independent expert
   manual proofreading or an independently curated biological ground-truth dataset.

5. The present work primarily focuses on datasets and processing requirements associated with NeuroMorpho.Org. Additional    validation across other morphology repositories, experimental methods, imaging modalities, species,
   and reconstruction systems will further establish generalizability.

Future work may include direct comparison with expert manual proofreading, benchmarking against alternative machine-learning classifiers and existing QC pipelines, extension to additional neuronal classes and structural domains,
and evaluation using broader independently curated datasets.

---

## Acknowledgments

This research was supported by resources provided by the [Office of Research Computing at George Mason University](https://orc.gmu.edu) and funded in part by grants from the **National Science Foundation (Award No. 2018631)** and the **National Institutes of Health (Award No. R37NS39600)**.

The authors are grateful to Drs. Duncan Donohue, Sridevi Polavaram, and Sumit Nanda for developing various components of the NeuroMorpho.Org data processing codebase used in this work.
