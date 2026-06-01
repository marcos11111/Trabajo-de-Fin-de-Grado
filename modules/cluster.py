import logging
from dataclasses import dataclass
from pathlib import Path

import discretisedfield as dfield
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler

from modules.data_core import MaterialProps, Quantity

logger = logging.getLogger(__name__)

@dataclass
class ClusterConfig:
    n_clusters: tuple[int, int] | list[int] | int = (2, 8)
    use_pca: bool = True
    n_components: int = 18
    stride: int = 1
    save_details: bool = True
    plot: bool = True

class Clusterer:
    """Clustering management framework."""
    
    LABEL_EMPTY = -2       
    LABEL_NOISE = -1       
    MAG_THRESHOLD = 1e-6   

    def __init__(self, data_core, visualizer, data_dir: Path, anis_file_path: Path, max_sample_size: int = 10000):
        self.core = data_core
        self.vis = visualizer
        self.data_dir = data_dir
        self.anis_file_path = anis_file_path
        self.max_sample_size = max_sample_size
        self.cluster_results = {}
        self.ground_truth_clusters = None

    def _prepare_data_pipeline(self, dfs: dict, df_norm: pd.DataFrame, quantities: list, start_step: int, end_step: int | None, cfg: ClusterConfig):
        all_features = [self._prepare_clustering_features(dfs[qty], start_step, end_step, qty) for qty in quantities]
        X_features_full = np.hstack(all_features)
        X_full_rows = X_features_full.shape[0]

        time_cols_norm = df_norm.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        active_mask = (df_norm[time_cols_norm] > self.MAG_THRESHOLD).any(axis=1).values
        if not np.any(active_mask): active_mask = np.ones(X_full_rows, dtype=bool)

        X_active = X_features_full[active_mask]
        X_active_scaled = StandardScaler().fit_transform(X_active)

        if cfg.use_pca:
            pca = PCA(n_components=cfg.n_components, random_state=42)
            X_active = pca.fit_transform(X_active_scaled)
        else:
            X_active = X_active_scaled
        return X_active, active_mask, X_full_rows

    def _prepare_clustering_features(self, df: pd.DataFrame, start_step: int | float, end_step: int | float | None, quantity_key: str) -> np.ndarray:
        time_cols = df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        total_steps = len(time_cols)
        s = int(start_step * total_steps) if isinstance(start_step, float) else start_step
        e = int(end_step * total_steps) if isinstance(end_step, float) else end_step
            
        selected_cols = time_cols[s:e]
        features_array = df[selected_cols].values
        if 'angle' in str(quantity_key).lower():
            return np.hstack([np.cos(features_array), np.sin(features_array)])
        return features_array

    def extract_anisotropy_ground_truth(self, df_angle: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
        gt_path = self.data_dir.parent / "global_ground_truth_clusters.csv"
        if not force_recompute and gt_path.exists():
            self.ground_truth_clusters = pd.read_csv(gt_path)
            return self.ground_truth_clusters
        
        anis_field = dfield.Field.from_file(self.anis_file_path)
        s_anis = anis_field.sel(z=0)
        ux, uy = s_anis.x.array.squeeze().flatten(), s_anis.y.array.squeeze().flatten()
        
        norm = np.sqrt(ux**2 + uy**2)
        true_angles_rad = np.mod(np.arctan2(uy, ux), np.pi)
        true_angles_rad[norm < 1e-3] = np.nan 
        
        clean_angles = np.nan_to_num(true_angles_rad, nan=-99.0)
        features = np.column_stack((np.round(clean_angles, 3), np.round(norm, 3)))
        _, cluster_ids = np.unique(features, axis=0, return_inverse=True)
        cluster_ids[np.isnan(true_angles_rad)] = self.LABEL_EMPTY

        df_gt = df_angle[['x (m)', 'y (m)']].copy()
        df_gt['Cluster'] = cluster_ids
        df_gt['Angle_Rad'] = true_angles_rad
        self.ground_truth_clusters = df_gt
        self.ground_truth_clusters.to_csv(gt_path, index=False)
        return self.ground_truth_clusters

    def evaluate_physics_accuracy(self, labels: np.ndarray, inferred_results: dict, tolerance_deg: float = 5.0) -> tuple[float, float]:
        if not self.anis_file_path.exists(): return 0.0, 0.0
        anis_field = dfield.Field.from_file(self.anis_file_path)
        s_anis = anis_field.sel(z=0)
        ux, uy = s_anis.x.array.squeeze().flatten(), s_anis.y.array.squeeze().flatten()
        
        norm = np.sqrt(ux**2 + uy**2)
        true_angles_deg = np.mod(np.degrees(np.arctan2(uy, ux)), 180)
        true_angles_deg[norm < 1e-3] = np.nan 
        
        predicted_angles_deg = np.full_like(true_angles_deg, np.nan)
        for c, data in inferred_results.items(): predicted_angles_deg[labels == c] = np.mod(data['theta_deg'], 180)
            
        eval_mask = (labels >= 0) & (~np.isnan(true_angles_deg))
        if not np.any(eval_mask): return 0.0, 0.0
        
        true_v, pred_v = true_angles_deg[eval_mask], predicted_angles_deg[eval_mask]
        diff = np.abs(true_v - pred_v) % 180
        error_deg = np.minimum(diff, 180 - diff)
        return np.mean(error_deg <= tolerance_deg) * 100.0, np.mean(error_deg)

    def compare_clustering_methods(self, dfs: dict, df_norm: pd.DataFrame, gt_labels: np.ndarray | None, 
                                   quantities: list, start_step=0, end_step=None, 
                                   force_recompute=False, cfg: ClusterConfig | None = None,
                                   plot_filename: str = None):
        cfg = cfg or ClusterConfig()
        range_tag = f"s{start_step}_e{end_step if end_step is not None else 'end'}"
        q_tag = f"{'_'.join([str(q) for q in quantities])}_{range_tag}"
        
        clusters_pqt_path = self.data_dir / f"cluster_results_{q_tag}.parquet"
        metrics_csv_path = self.data_dir / f"cluster_metrics_{q_tag}.csv"

        if not force_recompute and clusters_pqt_path.exists():
            df_res = pd.read_parquet(clusters_pqt_path)
            metrics = pd.read_csv(metrics_csv_path).to_dict(orient='list')
            results_full = {col: df_res[col].values for col in df_res.columns if col not in ['x (m)', 'y (m)', 'Cluster']}
            if cfg.plot: self._plot_clustering_dashboard(df_norm, results_full, metrics, cfg.stride, plot_filename or f"cluster_dashboard.pdf", has_gt=(gt_labels is not None))
            return results_full, metrics

        X_active, active_mask, X_full_shape = self._prepare_data_pipeline(dfs, df_norm, quantities, start_step, end_step, cfg)
        s_size = min(self.max_sample_size, len(X_active))
        k_range = range(cfg.n_clusters[0], cfg.n_clusters[1] + 1) if isinstance(cfg.n_clusters, tuple) else (cfg.n_clusters if isinstance(cfg.n_clusters, list) else [cfg.n_clusters])

        best_k, best_sil, best_labels_active = -1, -1.0, None

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_active = kmeans.fit_predict(X_active)
            sil = silhouette_score(X_active, labels_active, sample_size=s_size, random_state=42) if len(np.unique(labels_active)) > 1 else -1.0
            if sil > best_sil:
                best_sil, best_k, best_labels_active = sil, k, labels_active

        full_labels = np.full(X_full_shape, self.LABEL_EMPTY)
        full_labels[active_mask] = best_labels_active

        results_full = {'K-Means': full_labels}
        metrics = {'Silhouette (↑)': [best_sil]}

        if cfg.save_details:
            df_res = df_norm[['x (m)', 'y (m)']].copy()
            if gt_labels is not None: df_res['Cluster'] = gt_labels
            for name, lbls in results_full.items(): df_res[name] = lbls
            df_res.to_parquet(clusters_pqt_path, index=False)
            pd.DataFrame(metrics).to_csv(metrics_csv_path, index=False)

        if cfg.plot: self._plot_clustering_dashboard(df_norm, results_full, metrics, cfg.stride, plot_filename or f"cluster_dashboard.pdf", has_gt=(gt_labels is not None))
        return results_full, metrics

    def _plot_clustering_dashboard(self, df: pd.DataFrame, results: dict, metrics: dict, stride: int, filename: str, has_gt: bool = False) -> None:
        """Dashboard con alineación geométrica blindada mediante GridSpec explícito."""
        vis_maps = results
        n_cols = len(vis_maps) + (1 if has_gt and self.ground_truth_clusters is not None else 0)

        # Dimensiones fijas para evitar el auto-scaling destructivo de Matplotlib
        fig = plt.figure(figsize=((6.5 * n_cols)/2.54, 6.5/2.54), facecolor='#ffffff')
        
        # Estructuramos la rejilla de subplots de forma explícita
        gs = fig.add_gridspec(1, n_cols, wspace=0.15, hspace=0.0)

        x_coords = df['x (m)'].unique()
        y_coords = df['y (m)'].unique()
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

        import string
        labels_letters = list(string.ascii_lowercase)

        for i, (name, labels) in enumerate(vis_maps.items()):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor('#ffffff')

            grid_labels = df.assign(L=labels).pivot(index='y (m)', columns='x (m)', values='L').values
            mapped_grid = np.where(grid_labels == self.LABEL_EMPTY, np.nan, grid_labels)

            unique_valid = sorted([L for L in np.unique(labels) if L >= 0])
            colors = [self.vis.master_cmap(L % 10) for L in unique_valid]
            if not colors: colors = ['#000000']
            
            cmap = mcolors.ListedColormap(colors)
            cmap.set_bad(color='white', alpha=0.0) 

            display_grid = np.full_like(mapped_grid, np.nan)
            for idx, L in enumerate(unique_valid):
                display_grid[mapped_grid == L] = idx

            ax.imshow(display_grid, cmap=cmap, origin='lower', extent=extent, interpolation='nearest', zorder=1)

            if has_gt and self.ground_truth_clusters is not None:
                grid_gt = self.ground_truth_clusters.pivot(index='y (m)', columns='x (m)', values='Cluster').values
                unique_vals = np.unique(grid_gt[~np.isnan(grid_gt)])
                if len(unique_vals) > 1:
                    levels = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                    ax.contour(x_coords, y_coords, grid_gt, levels=levels, colors='black', linewidths=1.2, alpha=0.9, zorder=3)

            # Anclaje de la letra del panel (a)
            self.vis._add_panel_letter(ax, labels_letters[i], is_polar=False)
            
            # ⚡ FIJADO MATEMÁTICO: Anclamos el texto en el centro real de coordenadas del disco (0.5, 1.05)
            ax.text(0.5, 1.05, r"$\mathrm{Comparación}$", transform=ax.transAxes, 
                    fontsize=10, weight='bold', ha='center', va='bottom', zorder=5)
            
            ax.set_aspect('equal')
            ax.axis('off')

        if has_gt and self.ground_truth_clusters is not None:
            ax_err = fig.add_subplot(gs[0, n_cols - 1])
            ax_err.set_facecolor('#ffffff')

            grid_labels = df.assign(L=labels).pivot(index='y (m)', columns='x (m)', values='L').values
            grid_gt = self.ground_truth_clusters.pivot(index='y (m)', columns='x (m)', values='Cluster').values

            mask_valid = (grid_labels >= 0) & (grid_gt >= 0)
            mismatch_mask = np.zeros_like(grid_labels, dtype=bool)

            if np.any(mask_valid):
                cm = confusion_matrix(grid_gt[mask_valid], grid_labels[mask_valid])
                row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
                lookup = dict(zip(col_ind, row_ind))
                aligned_grid = np.vectorize(lambda x: lookup.get(x, x) if x >= 0 else x)(grid_labels)
                mismatch_mask = (aligned_grid != grid_gt) & mask_valid

            error_grid = np.full_like(grid_labels, np.nan, dtype=float)
            error_grid[grid_labels >= 0] = 0.0 
            error_grid[mismatch_mask] = 1.0    

            cmap_err = mcolors.ListedColormap(['#ffffff', '#d62728'])
            cmap_err.set_bad(color='white', alpha=0.0)

            ax_err.imshow(error_grid, cmap=cmap_err, origin='lower', extent=extent, interpolation='nearest', zorder=1)

            unique_vals = np.unique(grid_gt[~np.isnan(grid_gt)])
            if len(unique_vals) > 1:
                levels = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                ax_err.contour(x_coords, y_coords, grid_gt, levels=levels, colors='black', linewidths=1.2, alpha=0.8, zorder=3)

            # Anclaje de la letra del panel (b)
            self.vis._add_panel_letter(ax_err, labels_letters[n_cols - 1], is_polar=False)
            
            # ⚡ FIJADO MATEMÁTICO: Anclamos el texto exactamente en la misma coordenada
            ax_err.text(0.5, 1.05, r"$\mathrm{Desajuste}$", transform=ax_err.transAxes, 
                        fontsize=10, weight='bold', ha='center', va='bottom', zorder=5)
            
            ax_err.set_aspect('equal')
            ax_err.axis('off')

        self.vis.save_and_show(fig, filename)

    def plot_combined_anisotropy_maps(self, df_coords: pd.DataFrame, labels: np.ndarray, 
                                      inferred_results: dict, gt_df: pd.DataFrame = None, 
                                      subtitle: str = "", filename: str = "combined_anisotropy_map.pdf"):
                                      
        inferred_map = np.full(len(labels), np.nan)
        for c, data in inferred_results.items():
            inferred_map[labels == c] = np.mod(np.radians(data['theta_deg']), np.pi)
            
        grid_infer = pd.DataFrame({'x': df_coords['x (m)'], 'y': df_coords['y (m)'], 'A': inferred_map}).pivot(index='y', columns='x', values='A').values
        grid_gt = gt_df.pivot(index='y (m)', columns='x (m)', values='Angle_Rad').values if gt_df is not None else np.full_like(grid_infer, np.nan)
            
        # ⚡ SOLUCIÓN GEOMÉTRICA BLINDADA: 5 columnas. Las posiciones 2 y 4 son exclusivas para las colorbars.
        fig = plt.figure(figsize=(19.5/2.54, 5.5/2.54), facecolor='#ffffff')
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.15, hspace=0.0)
        
        extent = [df_coords['x (m)'].min(), df_coords['x (m)'].max(), df_coords['y (m)'].min(), df_coords['y (m)'].max()]
        
        # Panel (a) Ground Truth
        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(grid_gt, cmap='twilight_shifted', origin='lower', extent=extent, vmin=0, vmax=np.pi)
        self.vis._add_panel_letter(ax0, 'a', is_polar=False)
        ax0.text(0.5, 1.05, r"$\mathrm{Original}$", transform=ax0.transAxes, fontsize=10, weight='bold', ha='center', va='bottom')
        ax0.set_aspect('equal')
        ax0.axis('off')
        
        # Panel (b) Inferencia
        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(grid_infer, cmap='twilight_shifted', origin='lower', extent=extent, vmin=0, vmax=np.pi)
        self.vis._add_panel_letter(ax1, 'b', is_polar=False)
        ax1.text(0.5, 1.05, r"$\mathrm{Inferido}$", transform=ax1.transAxes, fontsize=10, weight='bold', ha='center', va='bottom')
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # ⚡ Colorbar para paneles (a) y (b) en la columna de índice 2
        cbar_ax1 = fig.add_subplot(gs[0, 2])
        cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='vertical')
        cbar1.set_ticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        cbar1.ax.set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        cbar1.ax.tick_params(labelsize=8)

        # Panel (c) Error
        ax2 = fig.add_subplot(gs[0, 3])
        if gt_df is not None:
            diff = np.abs(grid_gt - grid_infer)
            error_map = np.minimum(diff, np.pi - diff)
            im_err = ax2.imshow(error_map, cmap='inferno', origin='lower', extent=extent, vmin=0, vmax=np.pi/2)
            
            self.vis._add_panel_letter(ax2, 'c', is_polar=False)
            ax2.text(0.5, 1.05, r"$\mathrm{Error\ Angular}$", transform=ax2.transAxes, fontsize=10, weight='bold', ha='center', va='bottom')
            ax2.set_aspect('equal')
            ax2.axis('off')
            
            # ⚡ Colorbar exclusiva para el error en la columna de índice 4
            cbar_ax2 = fig.add_subplot(gs[0, 4])
            cbar_err = fig.colorbar(im_err, cax=cbar_ax2, orientation='vertical')
            cbar_err.set_ticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
            cbar_err.ax.set_yticklabels([r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
            cbar_err.ax.tick_params(labelsize=8)
        else:
            ax2.text(0.5, 0.5, 'GT no disponible', ha='center', va='center', fontsize=10)
            self.vis._add_panel_letter(ax2, 'c', is_polar=False)
            ax2.text(0.5, 1.05, r"$\mathrm{Error\ Angular}$", transform=ax2.transAxes, fontsize=10, weight='bold', ha='center', va='bottom')
            ax2.set_aspect('equal')
            ax2.axis('off')
            
            # Dejamos la última columna vacía si no hay GT
            ax_empty = fig.add_subplot(gs[0, 4])
            ax_empty.axis('off')
        
        self.vis.save_and_show(fig, filename)

    def get_cluster_anisotropy_data(self, labels: np.ndarray) -> dict:
        if not self.anis_file_path.exists(): return {}
        anis_field = dfield.Field.from_file(self.anis_file_path)
        s_anis = anis_field.sel(z=0)
        ux, uy = s_anis.x.array.squeeze().flatten(), s_anis.y.array.squeeze().flatten()
        anis_angles = np.arctan2(uy, ux)
        
        cluster_anis = {}
        unique_clusters = [c for c in np.unique(labels) if c >= 0]
        for cluster_id in unique_clusters: cluster_anis[cluster_id] = anis_angles[labels == cluster_id]
        return cluster_anis

    def get_cluster_hysteresis_data(self, labels: np.ndarray, override_df: pd.DataFrame = None) -> dict:
        hysteresis_data = {}
        unique_clusters = [c for c in np.unique(labels) if c >= 0]
        df_angle = override_df if override_df is not None else self.core.get_df(Quantity.ANGLE)
        time_cols = df_angle.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        
        for cluster_id in unique_clusters:
            mx_evolution = np.cos(df_angle.loc[labels == cluster_id, time_cols].values).mean(axis=0)
            hysteresis_data[cluster_id] = mx_evolution
        return hysteresis_data

    def infer_anisotropy_multiregion(self, df_coords: pd.DataFrame, labels: np.ndarray, hysteresis_data: dict, b_ext: np.ndarray, props: MaterialProps) -> dict:
        mu_0 = 4 * np.pi * 1e-7
        inferred_results = {}
        
        x_unique = np.sort(df_coords['x (m)'].unique())
        y_unique = np.sort(df_coords['y (m)'].unique())
        dx = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1e-9
        dy = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1e-9
        cell_area = dx * dy

        df_grid = df_coords.copy()
        df_grid['L'] = labels
        grid_labels = df_grid.pivot(index='y (m)', columns='x (m)', values='L').values

        unique_clusters = [c for c in np.unique(labels) if c >= 0]
        areas, perimeters = {}, {c: 0.0 for c in unique_clusters}
        interfaces = {c: {n: 0.0 for n in unique_clusters} for c in unique_clusters}
        shifts = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for c in unique_clusters:
            mask_c = (grid_labels == c)
            areas[c] = np.sum(mask_c) * cell_area
            for dy_s, dx_s in shifts:
                shifted_labels = np.roll(grid_labels, shift=(dy_s, dx_s), axis=(0, 1))
                edge_mask = mask_c & (shifted_labels != c)
                perimeters[c] += np.sum(edge_mask) * dx 
                for neighbor in unique_clusters:
                    if neighbor != c: interfaces[c][neighbor] += np.sum(edge_mask & (shifted_labels == neighbor)) * dx

        l_ex = np.sqrt((2 * props.A) / (mu_0 * props.Ms**2))
        df_angle = self.core.get_df(Quantity.ANGLE)
        time_cols = df_angle.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        common_len = min(len(b_ext), len(time_cols))
        
        grid_angles = df_angle.pivot(index='y (m)', columns='x (m)', values=time_cols[-common_len:][np.argmin(np.abs(b_ext[-common_len:]))]).values
        theta_raw_dict = {}
        
        for c in unique_clusters:
            mask_2d = (grid_labels == c)
            dist_map = distance_transform_edt(mask_2d)
            core_mask_2d = mask_2d & (dist_map >= np.percentile(dist_map[mask_2d], 60)) if np.any(mask_2d) else mask_2d
            angles_c = grid_angles[core_mask_2d] if np.any(core_mask_2d) else grid_angles[mask_2d]
            
            if len(angles_c) == 0: 
                theta_raw_dict[c] = 0.0
                continue
            theta_raw_dict[c] = np.mod(np.arctan2(np.mean(np.sin(angles_c)), np.mean(np.cos(angles_c))), np.pi)

        for c in unique_clusters:
            total_interface = sum(interfaces[c].values())
            theta_vecinos = sum(interfaces[c][n] * theta_raw_dict[n] for n in unique_clusters if n != c) / total_interface if total_interface > 0 else theta_raw_dict[c]
            gamma = props.kappa * (perimeters[c] / areas[c]) * l_ex if areas[c] > 0 else 0.0
            
            theta_raw = theta_raw_dict[c]
            delta_theta = 0.5 * np.arctan2(np.sin(2 * (theta_vecinos - theta_raw)), np.cos(2 * (theta_vecinos - theta_raw)))
            theta_final = theta_raw - gamma * delta_theta
            
            inferred_results[c] = {
                'phi_i_deg': np.degrees(theta_raw),
                'theta_deg': np.degrees(theta_final),
                'drag_deg': np.degrees(-gamma * delta_theta), 
                'correction_deg': np.degrees(theta_final - theta_raw)
            }
        return inferred_results
    
    def export_inferred_to_ovf(self, df_coords: pd.DataFrame, labels: np.ndarray, inferred_results: dict, filename: str = "inferred_anis.ovf"):
        if not self.anis_file_path.exists(): raise FileNotFoundError("Template mesh OVF missing.")
        template_field = dfield.Field.from_file(self.anis_file_path)
        mesh = template_field.mesh
        vector_data = np.zeros((*mesh.n, 3))
        
        grid_labels = df_coords.copy().assign(L=labels).pivot(index='x (m)', columns='y (m)', values='L').values
        for cluster_id, data in inferred_results.items():
            mask = (grid_labels == cluster_id)
            vector_data[mask, 0, 0] = np.cos(np.radians(data['theta_deg']))
            vector_data[mask, 0, 1] = np.sin(np.radians(data['theta_deg']))
            
        new_field = dfield.Field(mesh, nvdim=3, value=vector_data)
        out_path = self.data_dir.parent / filename
        new_field.to_file(str(out_path), representation="bin8")
        return out_path
    
    def plot_original_anisotropy_map(self, df_coords: pd.DataFrame, gt_df: pd.DataFrame, filename: str = "original_anisotropy_map.pdf"):
        if gt_df is None: return
        grid_gt = gt_df.pivot(index='y (m)', columns='x (m)', values='Angle_Rad').values
        
        # Ajustamos el ancho ligeramente (de 6.5 a 7.8) para dar cabida a la barra de color sin comprimir el disco
        fig, ax = plt.subplots(figsize=(7.8/2.54, 6.5/2.54), facecolor='#ffffff')
        extent = [df_coords['x (m)'].min(), df_coords['x (m)'].max(), df_coords['y (m)'].min(), df_coords['y (m)'].max()]
        
        im = ax.imshow(grid_gt, cmap='twilight_shifted', origin='lower', extent=extent, vmin=0, vmax=np.pi)
        
        # ⚡ NUEVO: Inyección de la barra de color con proporciones idénticas a las del mapa combinado
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_ticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        cbar.ax.set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        cbar.ax.tick_params(labelsize=8)
        
        ax.set_aspect('equal')
        ax.axis('off') # Mantiene el recuadro cartesiano y ticks ocultos para un acabado limpio
        
        self.vis.save_and_show(fig, filename)
