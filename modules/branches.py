import gc
import json
import logging
import shutil
import sys
import traceback
import itertools
from dataclasses import asdict
from pathlib import Path

import discretisedfield as dfield
import numpy as np
import pandas as pd

from modules.cluster import ClusterConfig, Clusterer
from modules.data_core import DataCore, MaterialProps, Quantity, load_mumax_table
from modules.simulator import RemoteSimulator
from modules.visualize import Visualizer

logger = logging.getLogger(__name__)

class BranchAnalyzer:
    def __init__(self, in_folder: Path, project_data_dir: Path, project_plot_dir: Path,
                 anis_path: str = "anisU000000.ovf", max_sample_size: int = 10000, 
                 facecolor: str = '#ffffff', use_scienceplots: bool = False): # ⚡ NUEVO ARGUMENTO
        self.data = DataCore(in_folder, project_data_dir, project_plot_dir, anis_path)
        self.vis = Visualizer(self.data, facecolor=facecolor, use_scienceplots=use_scienceplots) # ⚡ PASAMOS EL ARGUMENTO
        self.ml = Clusterer(self.data, self.vis, self.data.data_dir, 
                            self.data.in_folder / self.data.anis_path, max_sample_size)


class MumaxProject:
    """Gestiona entornos de ejecución, transformaciones y análisis multirama."""
    
    def __init__(self, base_folder: str | Path, out_folder: str | Path | None = None, 
                 ref_branch: str = "yes", diff_branch: str = "diff_fields", **kwargs):
        self.base_folder = Path(base_folder)
        self.ref_branch = ref_branch
        self.diff_branch = diff_branch
        
        self._setup_paths(out_folder)
        self._setup_settings(kwargs)
        self._setup_benchmark_defaults(kwargs)
        
        self.analyzers: dict[str, BranchAnalyzer] = {}
        self.global_data: dict[Quantity, pd.DataFrame] = {}
        self.global_gt_df: pd.DataFrame | None = None
        self.has_gt = False
        self.use_scienceplots = kwargs.get('use_scienceplots', False) # ⚡ NUEVO

        self._setup_logging()
        if kwargs.get('auto_prepare', True):
            self._auto_prepare()
        self._save_metadata()

    def _auto_prepare(self):
        logger.info("Inicializando estructuras de datos automatizadas...")
        self.get_analyzer("yes").data.load_data([Quantity.ANGLE])
        self.get_analyzer("no").data.load_data([Quantity.ANGLE])
        
        self.subtract_angles("yes", "no", "diff_angles")
        self.subtract_fields("yes", "no", "diff_fields")
        self._prepare_globals()
        logger.info("Contexto cargado para ejecución matemática.")

    def _get_physics_context(self) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Path]:
        diff_analyzer = self.get_analyzer(self.diff_branch)
        diff_analyzer.data.load_data([Quantity.ANGLE])
        diff_df = diff_analyzer.data.get_df(Quantity.ANGLE)
        
        table_path = self.base_folder / self.ref_branch / "table.txt"
        table_df = load_mumax_table(table_path)
        b_ext = table_df['B_extx (T)'].values
        
        df_coords = self.global_data[Quantity.NORM][['x (m)', 'y (m)']]
        return diff_df, b_ext, df_coords, table_path

    def _prepare_selection_data(self, selection: list[tuple[Quantity, str]]) -> tuple[dict, list[str]]:
        hybrid_dfs = {}
        synthetic_keys = []
        for qty, branch in selection:
            key = f"{qty}_{branch}"
            analyzer = self.get_analyzer(branch)
            qty_str = str(qty)
            if qty_str.startswith('d_'):
                analyzer.data.load_data([Quantity(qty_str[2:])])
            analyzer.data.load_data([qty])
            hybrid_dfs[key] = analyzer.data.get_df(qty)
            synthetic_keys.append(key)
        return hybrid_dfs, synthetic_keys

    def _setup_paths(self, out_folder: str | Path | None):
        if not out_folder:
            out_folder = self.base_folder.parent / f"{self.base_folder.name}_output"
        self.out_folder = Path(out_folder)
        self.project_plot_dir = self.out_folder
        self.project_data_dir = self.out_folder / "data"
        for p in [self.project_plot_dir, self.project_data_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def _setup_settings(self, kwargs):
        self.animations = kwargs.get('animations', False)
        self.start_step = kwargs.get('start_step', 0)
        self.end_step = kwargs.get('end_step', None)
        self.stride = kwargs.get('stride', 1)
        self.quantities = kwargs.get('quantities', [Quantity.ANGLE, Quantity.D_ANGLE])
        self.cluster_cfg = kwargs.get('cluster_cfg') or ClusterConfig()

    def _setup_benchmark_defaults(self, kwargs):
        self.benchmark_step_configs = kwargs.get('benchmark_step_configs', [{'name': 'Full Range', 'start': 0, 'end': None}])
        self.benchmark_quantity_options = kwargs.get('benchmark_quantity_options', [[Quantity.ANGLE, Quantity.D_ANGLE, Quantity.NORM, Quantity.D_NORM, Quantity.D_NORM_XY]])

    def _setup_logging(self):
        log_path = self.out_folder / "pipeline.log"
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[file_handler, stream_handler])

    def _save_metadata(self):
        meta = {
            "source": str(self.base_folder.absolute()),
            "range": [self.start_step, self.end_step],
            "cluster_cfg": asdict(self.cluster_cfg)
        }
        with open(self.project_data_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    def _prepare_globals(self):
        if Quantity.NORM in self.global_data: return
        logger.info("Parseando matrices de normalización y GT...")
        base_analyzer = self.get_analyzer("yes")
        base_analyzer.data.load_data([Quantity.ANGLE, Quantity.NORM, Quantity.NORM_XY, Quantity.D_NORM, Quantity.D_NORM_XY])
        
        for qty in [Quantity.NORM, Quantity.NORM_XY, Quantity.D_NORM, Quantity.D_NORM_XY]:
            self.global_data[qty] = base_analyzer.data.get_df(qty)

        anis_file = self.base_folder / "yes" / "anisU000000.ovf"
        if anis_file.exists():
            logger.info("Ground Truth espacial detectado.")
            self.has_gt = True
            self.global_gt_df = base_analyzer.ml.extract_anisotropy_ground_truth(base_analyzer.data.get_df(Quantity.ANGLE))
        else:
            self.has_gt = False

    def get_analyzer(self, branch_name: str) -> BranchAnalyzer:
        if branch_name not in self.analyzers:
            read_path = self.base_folder / branch_name
            self.analyzers[branch_name] = BranchAnalyzer(
                read_path, self.project_data_dir, self.project_plot_dir, 
                use_scienceplots=self.use_scienceplots # ⚡ PASAMOS EL ARGUMENTO
            )
        return self.analyzers[branch_name]

    def subtract_fields(self, source: str, target: str, out: str):
        out_dir = self.base_folder / out
        if out_dir.exists() and any(out_dir.iterdir()):
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        path_a, path_b = self.base_folder / source, self.base_folder / target
        
        for file_a in sorted([f for f in path_a.iterdir() if f.suffix in ('.ovf', '.omf')]):
            file_b = path_b / file_a.name
            if file_b.exists() and file_a.name != "anisU000000.ovf":
                (dfield.Field.from_file(file_a) - dfield.Field.from_file(file_b)).to_file(str(out_dir / file_a.name), representation="bin8")
            elif file_a.name == "anisU000000.ovf":
                shutil.copy2(file_a, out_dir / file_a.name)

    def subtract_angles(self, source: str, target: str, out: str):
        branch_dir = self.base_folder / out
        data_out = branch_dir / "data" 
        pqt_path = data_out / "angle.parquet"
        
        if pqt_path.exists(): return

        data_out.mkdir(parents=True, exist_ok=True)
        df_a = pd.read_parquet(self.project_data_dir / source / "angle.parquet")
        df_b = pd.read_parquet(self.project_data_dir / target / "angle.parquet")
        
        df_diff = df_a[['x (m)', 'y (m)']].copy()
        t_cols = [c for c in df_a.columns if c not in ['x (m)', 'y (m)']]
        
        df_diff[t_cols] = ((df_a[t_cols] - df_b[t_cols]) + np.pi) % (2 * np.pi) - np.pi
        df_diff.to_parquet(pqt_path, index=False)

    def process_selection(self, selection: list[tuple[Quantity, str]] = None, 
                          mag_branch: str = "yes", force_recompute: bool = False, 
                          material: 'MaterialProps' = None, verification_table_path: Path = None):
        if material is None: raise ValueError("MaterialProps es requerido.")
        self._prepare_globals()
        
        hybrid_dfs, synthetic_keys = self._prepare_selection_data(selection)
        host_analyzer = self.get_analyzer(self.ref_branch)
        
        res, _ = host_analyzer.ml.compare_clustering_methods(
            dfs=hybrid_dfs, df_norm=self.global_data[Quantity.NORM], 
            gt_labels=self.global_gt_df['Cluster'].values if self.has_gt else None, 
            quantities=synthetic_keys, start_step=self.start_step, end_step=self.end_step,
            force_recompute=force_recompute, cfg=self.cluster_cfg,
            plot_filename="clustering_dashboard.pdf" 
        )

        winning = res.get('K-Means')
        if winning is not None:
            ref_analyzer = self.get_analyzer(mag_branch)
            h_data_plot = host_analyzer.ml.get_cluster_hysteresis_data(winning, override_df=ref_analyzer.data.get_df(Quantity.ANGLE))
            
            _, b_ext, df_coords, table_path = self._get_physics_context()
            inferred = host_analyzer.ml.infer_anisotropy_multiregion(df_coords, winning, h_data_plot, b_ext, material)
            
            # ⚡ FIX: Usamos "filename=" explícitamente para evitar archivos ocultos
            host_analyzer.ml.plot_combined_anisotropy_maps(
                df_coords, winning, inferred, gt_df=self.global_gt_df, filename="combined_anisotropy_map.pdf"
            )
            
            inferred_map = np.full(len(winning), np.nan)
            for c, data in inferred.items():
                inferred_map[winning == c] = np.radians(data['theta_deg'])
                
            gt_angles = self.global_gt_df['Angle_Rad'].values if self.global_gt_df is not None else np.full_like(inferred_map, np.nan)
            
            host_analyzer.vis.plot_global_anisotropy_histograms(
                gt_angles, inferred_map, filename="global_histograms.pdf"
            )
            
            host_analyzer.vis.plot_cluster_anisotropy_polar(
                host_analyzer.ml.get_cluster_anisotropy_data(winning), inferred, filename="cluster_polar.pdf"
            )
            
            host_analyzer.vis.plot_cluster_hysteresis(
                h_data_plot, table_path, inferred, filename="hysteresis_regional.pdf"
            )
            
            host_analyzer.vis.plot_global_verification_hysteresis(
                table_path, verification_table_path, filename="hysteresis_global_compare.pdf"
            )
            
            # ⚡ INYECTAMOS LAS GRÁFICAS INDIVIDUALES Y PURAS
            if self.global_gt_df is not None:
                host_analyzer.ml.plot_original_anisotropy_map(df_coords, self.global_gt_df, filename="original_anisotropy_map.pdf")
                
            host_analyzer.vis.plot_original_hysteresis_clean(table_path, filename="hysteresis_original_clean.pdf")
            
            try:
                analyzer_yes = self.get_analyzer("yes")
                analyzer_no = self.get_analyzer("no")
                analyzer_diff = self.get_analyzer("diff_angles")
                
                # Forzamos la carga en RAM de las magnitudes espaciales
                analyzer_yes.data.load_data([Quantity.ANGLE])
                analyzer_no.data.load_data([Quantity.ANGLE])
                analyzer_diff.data.load_data([Quantity.ANGLE])
                
                # Construimos el diccionario ordenado para el grid
                dfs_comp = {
                    r"$\mathbf{m}_\text{no}$": analyzer_no.data.get_df(Quantity.ANGLE),
                    r"$\mathbf{m}$": analyzer_yes.data.get_df(Quantity.ANGLE),
                    r"$\mathbf{m}_\text{dif}$": analyzer_diff.data.get_df(Quantity.ANGLE)
                }
                
                # ⚡ FIX: Inyectamos explícitamente el campo magnético recuperado de la tabla OMF
                host_analyzer.vis.plot_magnetization_comparison(
                    dfs_comp, 
                    b_ext=b_ext, 
                    filename="magnetization_frames_compare.pdf", 
                    num_frames=4
                )
            except Exception as e:
                logger.warning(f"No se pudo generar el grid comparativo de magnetización: {e}")
            
            try:
                # hybrid_dfs contiene los dataframes exactos y ordenados de la configuración de entrada
                host_analyzer.vis.plot_clustering_features_grid(
                    dfs=hybrid_dfs, 
                    b_ext=b_ext, 
                    filename="clustering_features_B0.pdf"
                )
            except Exception as e:
                logger.warning(f"No se pudo generar el grid de variables de clustering: {e}")
            
            # -------------------------------------------------------------
            host_analyzer.ml.export_inferred_to_ovf(df_coords, winning, inferred, "inferred_anisotropy.ovf")
            
        # Generación de vídeos MP4 (Ángulo y Velocidad)
        if getattr(self, 'animations', False):
            logger.info(f"Generando animaciones MP4 (Ángulo y Velocidad) para las ramas de {self.base_folder.name}...")
            for branch_name in ["yes", "no"]:
                try:
                    analyzer = self.get_analyzer(branch_name)
                    analyzer.data.load_data([Quantity.ANGLE, Quantity.D_ANGLE])
                    
                    # ⚡ REDUCIMOS VELOCIDAD a fps=2 para un análisis visual sosegado
                    analyzer.vis.animation(quantity=str(Quantity.ANGLE), fps=2)
                    analyzer.vis.animation(quantity=str(Quantity.D_ANGLE), fps=2)
                except Exception as e:
                    logger.warning(f"Omitiendo animaciones para la rama '{branch_name}': {e}")

class BatchProcessor:
    """Orquestador maestro de la ejecución por lotes."""
    
    def __init__(self, parent_folder: Path, base_output_dir: Path, verification_parent_folder: Path,
                 material: MaterialProps, simulator: RemoteSimulator, cluster_cfg: ClusterConfig, **kwargs):
        self.parent_folder = Path(parent_folder)
        self.base_output_dir = Path(base_output_dir)
        self.verification_parent_folder = Path(verification_parent_folder)
        self.material = material
        self.simulator = simulator
        self.cluster_cfg = cluster_cfg
        
        self.exact_selection = kwargs.get("exact_selection")
        self.exact_pca = kwargs.get("exact_pca", 8)
        self.exact_kappa = kwargs.get("exact_kappa", 0.3)
        self.n_grid = kwargs.get("n_grid", 128)
        self.b_max = kwargs.get("b_max", 500.0e-3)
        self.animations = kwargs.get("animations", False)
        self.use_scienceplots = kwargs.get("use_scienceplots", False) # ⚡ NUEVO

    # Y en run_analysis_pass (línea ~205):
                

    def _get_mumax_parameters(self, bstep: float, **extra_variables) -> dict:
        params = {"Aex": self.material.A, "Msat": self.material.Ms, "myKu1": self.material.Ku1, "N": self.n_grid, "Bmax": self.b_max, "Bstep": bstep}
        params.update(extra_variables)
        return params

    def run_analysis_pass(self, label: str):
        subfolders = [f for f in self.parent_folder.iterdir() if f.is_dir() and (f / "yes").exists()]
        if not subfolders: return
            
        logger.info(f"[{label}] Desplegando pipeline sobre {len(subfolders)} muestras.")

        for idx, subfolder in enumerate(subfolders, start=1):
            sample_name = subfolder.name
            out_folder = self.base_output_dir / f"{sample_name}_analysis"
            
            try:
                current_cfg = ClusterConfig(**asdict(self.cluster_cfg))
                current_cfg.use_pca = (self.exact_pca is not None and self.exact_pca > 0)
                current_cfg.n_components = self.exact_pca
                self.material.kappa = self.exact_kappa

                project = MumaxProject(
                    base_folder=subfolder, out_folder=out_folder, 
                    animations=self.animations, cluster_cfg=current_cfg,
                    use_scienceplots=self.use_scienceplots # ⚡ PASAMOS EL ARGUMENTO
                )
                project.start_step, project.end_step = 0.0, 1.0
                
                verif_table_path = None
                if self.verification_parent_folder:
                    possible_paths = [
                        self.verification_parent_folder / sample_name / "yes" / "table.txt",
                        self.verification_parent_folder / f"{sample_name}_verificacion" / "yes" / "table.txt"
                    ]
                    for path in possible_paths:
                        if path.exists():
                            verif_table_path = path
                            break
                
                project.process_selection(selection=self.exact_selection, mag_branch='yes', force_recompute=True, material=self.material, verification_table_path=verif_table_path)
                            
            except Exception as e:
                logger.error(f"Fallo en la muestra '{sample_name}': {e}")
                logger.error(traceback.format_exc())
            finally:
                if 'project' in locals() and hasattr(project, 'analyzers'):
                    for analyzer in project.analyzers.values(): analyzer.data.free_memory()
                    del project
                gc.collect()

    def execute_pass1_inference(self, sample_prefix: str, seeds: list[int], b_step: float, skip_sims: bool = False):
        logger.info("="*60)
        logger.info("🚀 PASADA 1: SIMULACIÓN BASE E INFERENCIA IA")
        logger.info("="*60)
        
        if not skip_sims:
            initial_variants = [(f"{sample_prefix}_Seed{seed}", self._get_mumax_parameters(b_step, randomSeed=seed)) for seed in seeds]
            self.simulator.run_batch(parent_folder_name=self.parent_folder.name, variants=initial_variants)

        self.run_analysis_pass(label="Generación de Mapas Inferidos")
        logger.info("✅ PASADA 1 COMPLETADA.")

    def execute_pass2_verification(self, b_step: float, skip_sims: bool = False):
        logger.info("="*60)
        logger.info("🔍 PASADA 2: VERIFICACIÓN CRUZADA DE HISTÉRESIS")
        logger.info("="*60)
        
        if not skip_sims:
            verification_variants = []
            if self.parent_folder.exists():
                for subfolder in self.parent_folder.iterdir():
                    if subfolder.is_dir() and (subfolder / "yes").exists():
                        inferred_ovf_path = self.base_output_dir / f"{subfolder.name}_analysis" / "data" / "inferred_anisotropy.ovf"
                        if inferred_ovf_path.exists():
                            params_verif = self._get_mumax_parameters(b_step, AnisMapFile=str(inferred_ovf_path.resolve()))
                            verification_variants.append((f"{subfolder.name}_verificacion", params_verif))
            if verification_variants:
                self.simulator.run_batch(parent_folder_name=self.verification_parent_folder.name, variants=verification_variants, just_yes=True)
            else: return

        self.run_analysis_pass(label="Cálculo de Desviación RMSE/MAE")
        logger.info("✅ PASADA 2 COMPLETADA.")