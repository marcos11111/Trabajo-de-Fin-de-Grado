# Automated Micromagnetic Domain Segmentation & Multiregion Anisotropy Inference Pipeline

**Code repository for the Bachelor's Thesis (Trabajo Fin de Grado) in Physics and Mathematics.** *Author: Marcos Cuervo Santos | University of Oviedo*

## Abstract

This repository contains an advanced computational framework developed to orchestrate high-performance micromagnetic simulations (via MuMax3), perform automated magnetic domain segmentation using unsupervised machine learning, and execute physics-informed multiregion anisotropy inference. 

The pipeline bridges data-driven clustering techniques with underlying physical constraints—such as exchange length and spatial boundary conditions—to systematically recover spatial anisotropy maps. Furthermore, it includes automated cross-verification routines utilizing remote GPU infrastructure to validate the inferred macroscopic hysteresis behavior against ground-truth simulations.

## Methodology & Workflow

The computational pipeline operates through four synchronized phases to ensure mathematical rigor and structural validation:

1. **Distributed Micromagnetic Simulations:** Automated batch generation and transmission of script variants (.mx3) to a high-performance remote GPU server via parallel SSH/SCP worker threads.
2. **Feature Extraction & Dimensionality Reduction:** Vector extraction of local physical quantities (e.g., mx, my, mz, temporal angular velocity \partial\phi/\partial t, and field dynamics), followed by Principal Component Analysis (PCA) to isolate relevant variance.
3. **Unsupervised Domain Segmentation:** Spatial partitioning using K-Means clustering, optimized via Silhouette validation scoring, to isolate independent magnetic regions without prior ground-truth dependency.
4. **Physics-Informed Inference & Validation:** Analytical execution of a multiregion anisotropy inference engine that accounts for neighbor drag contributions (\kappa). The inferred Object Vector Field (OVF) map is automatically re-injected into the remote server for macroscopic hysteresis loop cross-verification.

## Repository Architecture

The codebase is modularized to separate data handling, machine learning, and physical simulation concerns:

├── modules/
│   ├── base.mx3          # MuMax3 base script template (Voronoi/Injected modes)
│   ├── branches.py       # Multi-branch data management and project handlers
│   ├── cluster.py        # ML clustering core & Multiregion Inference Engine
│   ├── data_core.py      # I/O Parquet engine and numerical derivative computations
│   ├── simulator.py      # Paramiko/SCP remote multi-threaded GPU executor
│   └── visualize.py      # Scientific plotting engine (Polar histograms, Hysteresis, Grids)
├── main.py               # Main execution script and hyperparameter configuration
├── requirements.txt      # Python environment dependencies
└── .gitignore            # Exclusion filters for local caches and binary arrays

## Reproducibility & Setup

### 1. Environment Preparation
To ensure reproducibility, clone the repository and instantiate a virtual environment:

git clone https://github.com/marcos11111/Trabajo-de-Fin-de-Grado.git
cd Trabajo-de-Fin-de-Grado
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

### 2. Infrastructure Configuration (.env)
To protect remote server credentials and decouple local paths from the codebase, create a .env file in the root directory. This file is excluded from version control by default.

# Remote GPU Server Credentials
REMOTE_HOST=156.35.97.43
REMOTE_USER=GPU
REMOTE_PASSWORD=your_secure_password

# Execution Environment Paths
MUMAX_PATH=F:/SciProg/Mumax/3.10f/mumax3.exe
REMOTE_BASE_DIR=F:/Users_Data/Your_User/regions/tests_remotos/

### 3. Execution
The full analytical pipeline is triggered via the main script. Physical parameters, grid definitions, and ML hyperparameters can be configured within the CONFIG dictionary inside main.py.

python main.py

## Analytical Capabilities

* **Ablation Studies:** Automated extraction of marginal performance metrics when modifying the physical dimensions included in the clustering feature vector.
* **Spatial Mismatch Analysis:** Hungarian matching algorithm alignment between unsupervised clusters and ground-truth regions to compute and visualize domain wall divergence.
* **Macroscopic Validation:** High-contrast, publication-ready visualizations mapping the original global hysteresis loops against the behavior governed by the newly inferred spatial anisotropy grids.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
