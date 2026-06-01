import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.branches import ClusterConfig, BatchProcessor
from modules.data_core import Quantity, MaterialProps
from modules.simulator import RemoteSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

CONFIG = {
    "aesthetics": {
        "use_scienceplots": True,
        "font.family": "serif", 
        "font.serif": ["DejaVu Serif"], 
        "font.size": 10,
        "axes.titlesize": 10, 
        "axes.labelsize": 10, 
        "xtick.labelsize": 8, 
        "ytick.labelsize": 8,
        "mathtext.fontset": "cm", 
        "figure.dpi": 300, 
        "savefig.format": "pdf"
    },
    "material": MaterialProps(A=1.3e-11, Ms=8e5, Ku1=2.5e5, dz=20e-9),
    "grid": {"n_grid": 128, "b_max": 500.0e-3},
    "ml": {
        "n_clusters": (2,8), "exact_pca": 8, "exact_kappa": 0.3,    
        "exact_selection": [(Quantity.NORM, 'diff_fields'), (Quantity.ANGLE, 'diff_fields'), 
                            (Quantity.D_ANGLE, 'yes'), (Quantity.D_NORM, 'diff_fields')]
    },
    "workflow": {
        "sample_prefix": "v4",
        "seeds": [71,72,73,74],
        "b_step": 100.0e-3,
        "skip_sims": True,
        "animations": True
    }
}

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    local_data_root = base_dir / "data_storage"
    
    simulator = RemoteSimulator.from_env(base_dir=base_dir, local_base_results=local_data_root / "mumax_remoto", max_concurrent=4)
    processor = BatchProcessor(
        parent_folder=local_data_root / "mumax_remoto" / "v3",
        verification_parent_folder=local_data_root / "mumax_remoto" / "v3_verificacion",
        base_output_dir=local_data_root / "v3_analisis",
        material=CONFIG["material"], simulator=simulator,
        cluster_cfg=ClusterConfig(n_clusters=CONFIG["ml"]["n_clusters"], plot=True),
        use_scienceplots=CONFIG["aesthetics"].get("use_scienceplots", False), # ⚡ NUEVO
        **CONFIG["ml"], **CONFIG["grid"]   
    )
    
    processor.execute_pass1_inference(
        sample_prefix=CONFIG["workflow"]["sample_prefix"], 
        seeds=CONFIG["workflow"]["seeds"], 
        b_step=CONFIG["workflow"]["b_step"],
        skip_sims=CONFIG["workflow"]["skip_sims"]
    )

    processor.execute_pass2_verification(
        b_step=CONFIG["workflow"]["b_step"],
        skip_sims=CONFIG["workflow"]["skip_sims"]
    )