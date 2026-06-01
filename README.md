# Automated Micromagnetic Domain Segmentation & Multiregion Anisotropy Inference Pipeline

An advanced computational framework designed to orchestrate high-performance micromagnetic simulations in **MuMax3**, perform automated magnetic domain segmentation via unsupervised machine learning, and execute physics-informed multiregion anisotropy inference.

This pipeline bridges data-driven clustering techniques with physical constraints (such as exchange length and spatial boundary conditions) to recover spatial anisotropy maps and perform automated cross-verification using remote GPU infrastructure.

## 🚀 System Architecture & Workflow

The pipeline operates in 4 synchronized phases to guarantee mathematical rigor and structural validation:

1. **Phase 1: Remote Simulation Sampling** – Automated batch generation and transmission of script variants (`.mx3`) to a high-performance remote GPU server via parallel SSH/SCP worker threads.
2. **Phase 2: Temporal & Structural Feature Engineering** – Vector extraction of local physical quantities ($\mathbf{m}_x, \mathbf{m}_y, \mathbf{m}_z$, temporal angular velocity $\partial\theta/\partial t$, and field dynamics) combined with PCA dimensionality reduction.
3. **Phase 3: Domain Segmentation & Blind Calibration** – Unsupervised partition using K-Means optimized via Silhouette validation scoring to isolate independent magnetic regions without ground-truth dependency.
4. **Phase 4: Physics-Informed Axis Inference & Cross-Verification** – Analytical execution of a multiregion anisotropy inference engine considering neighbor drag contributions ($\kappa$). The inferred OVF map is automatically re-injected into the remote server for macroscopic hysteresis loop cross-verification.

## 📁 Repository Structure

```text
├── modules/
│   ├── base.mx3          # MuMax3 base script template (Voronoi/Injected modes)
│   ├── branches.py       # Multi-branch data management and MumaxProject handlers
│   ├── cluster.py        # ML clustering core & Multiregion Inference Engine
│   ├── data_core.py      # I/O Parquet engine, numerical derivatives, and BenchmarkAnalyzer
│   ├── simulator.py      # Paramiko/SCP remote multi-threaded GPU executor
│   └── visualize.py      # Production-grade plotting engine (Polar histograms & Hysteresis)
├── main.py               # Dynamic entry point (Decoupled environment)
├── requirements.txt      # Python dependencies
└── .gitignore            # Security filters for local caches and data arrays

🛠️ Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/micromagnetic-segmentation-pipeline.git](https://github.com/your-username/micromagnetic-segmentation-pipeline.git)
cd micromagnetic-segmentation-pipeline
2. Install Dependencies
It is highly recommended to use a virtual environment (venv or conda):

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
3. Configure Infrastructure Secrets (.env)
To protect remote server credentials and separate local disk paths from the codebase, create a .env file in the root directory of the project:

Ini, TOML
# Remote GPU Server Access
REMOTE_HOST=156.35.97.43
REMOTE_USER=GPU
REMOTE_PASSWORD=your_secure_password

# Remote Executable Environment
MUMAX_PATH=F:/SciProg/Mumax/3.10f/mumax3.exe
REMOTE_BASE_DIR=F:/Users_Data/Your_User/regions/tests_remotos/
Note: The .env file is explicitly ignored by Git via .gitignore to prevent data and credential leaks.

💻 Usage
To trigger the complete multi-phase pipeline, configure your search space or material parameters in main.py and run:

Bash
python main.py
Analytical Capabilities Included:
Ablation Studies: Automated extraction of marginal performance gains when adding or removing specific physical dimensions from the clustering feature vector.

Spatial Mismatch Verification: Hungarian matching algorithm alignment between K-Means clusters and Ground Truth zones to generate strict domain wall contour divergence maps.

Macroscopic Validation Plots: High-contrast comparisons mapping the global original hysteresis loops against the newly inferred spatial anisotropy grid loops.

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.