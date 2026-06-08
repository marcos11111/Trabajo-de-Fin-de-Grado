import logging
from pathlib import Path
import string

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import matplotlib.cm as mcm

from modules.data_core import Quantity, load_mumax_table

try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

logger = logging.getLogger(__name__)

# Centralized font scaling constants
FONTS = {
    'label': 26,
    'title': 28,
    'tick': 22,
    'legend': 20,
    'letter': 30
}

class Visualizer:
    def __init__(self, data_core, facecolor: str = '#ffffff', plotting: bool = True, use_scienceplots: bool = False):
        self.core = data_core
        self.facecolor = facecolor 
        self.plotting = plotting
        self.use_scienceplots = use_scienceplots and HAS_SCIENCEPLOTS
        
        if use_scienceplots and not HAS_SCIENCEPLOTS:
            logger.warning("SciencePlots requested but not installed. Run 'pip install SciencePlots'. Using fallback.")
            
        if self.use_scienceplots:
            plt.style.use(['science', 'ieee', 'no-latex'])
            logger.info("SciencePlots loaded correctly with 'science' and 'ieee' profiles.")
            
        self.master_cmap = plt.get_cmap('tab10')

    # =========================================================================
    # INTERNAL HELPERS (DRY Enforcement)
    # =========================================================================

    def save_and_show(self, fig: plt.Figure, filename: str) -> None:
        if not self.plotting:
            plt.close(fig)
            return
            
        filename = filename if filename.endswith('.png') else filename.replace('.pdf', '.png')
        fig.savefig(self.core.plot_dir / filename, dpi=300, bbox_inches='tight', facecolor=self.facecolor)
        plt.close(fig)

    def apply_paper_style(self, ax: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = ""):
        if not self.use_scienceplots:
            ax.set_facecolor(self.facecolor)
            ax.tick_params(axis='both', which='major', labelsize=FONTS['tick'], direction='in', top=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
            ax.grid(True, linestyle='-', alpha=0.15, color='gray', linewidth=0.5)
            
        if xlabel: ax.set_xlabel(xlabel, fontsize=FONTS['label'], weight='bold', labelpad=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=FONTS['label'], weight='bold', labelpad=8)
        if title: ax.set_title(title, fontsize=FONTS['title'], weight='bold', pad=15)

    def _add_panel_letter(self, ax: plt.Axes, letter: str, is_polar: bool = False):
        x_pos, y_pos = (-0.15, 1.15) if is_polar else (-0.12, 1.05)
        ax.text(x_pos, y_pos, f'({letter})', transform=ax.transAxes, fontweight='bold', fontsize=FONTS['letter'])

    def _add_angle_colorbar(self, fig, im, ax=None, cax=None, orientation='horizontal', label=r'Ángulo $\phi$ (rad)', pad=0.04):
        cbar = fig.colorbar(im, ax=ax, cax=cax, orientation=orientation, pad=pad)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']) if orientation == 'horizontal' else cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar.ax.tick_params(labelsize=FONTS['tick']) 
        cbar.set_label(label, weight='bold', fontsize=FONTS['label'], labelpad=15)
        return cbar

    def _get_symmetric_counts(self, angles, bins):
        valid = angles[~np.isnan(angles)]
        if len(valid) == 0: return np.zeros(len(bins)-1), bins
        symmetric = np.concatenate([np.mod(valid, np.pi), np.mod(valid, np.pi) + np.pi])
        counts, b_edges = np.histogram(symmetric, bins=bins)
        return counts / 2.0, b_edges

    def _save_animation(self, anim, filepath, fps):
        """Unified fallback logic for FFmpeg -> GIF rendering"""
        try:
            anim.save(filepath, writer=animation.FFMpegWriter(fps=fps, bitrate=3000), facecolor=self.facecolor)
            logger.info(f"✅ MP4 generated: {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ FFmpeg missing or failed. Rendering native GIF...")
            gif_path = filepath.with_suffix('.gif')
            anim.save(gif_path, writer=animation.PillowWriter(fps=fps), facecolor=self.facecolor)
            logger.info(f"✅ GIF generated: {gif_path}")

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_cluster_anisotropy_polar(self, cluster_anis: dict, inferred_angles: dict = None, filename: str = "cluster_polar.pdf", subtitle: str = "") -> None:
        if not cluster_anis: return
        
        bins = np.linspace(0, 2 * np.pi, 41)
        cluster_data = {cid: self._get_symmetric_counts(angs, bins) for cid, angs in cluster_anis.items()}
        max_r = max([np.max(c[0]) for c in cluster_data.values()] + [1.0])
        
        n_clusters = len(cluster_anis)
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        fig = plt.figure(figsize=((8.0 * cols)/1.41, (8.5 * rows + 2.0)/1.41), facecolor=self.facecolor)
        gs = fig.add_gridspec(rows, cols, wspace=0.25, hspace=0.45)
        
        legend_lines = []
        for i, (cluster_id, (counts, b_edges)) in enumerate(cluster_data.items()):
            ax = fig.add_subplot(gs[i // cols, i % cols], projection='polar')
            ax.set_theta_zero_location('S')
            ax.grid(True, alpha=0.4, linestyle=':', color='gray') 
            
            ax.bar((b_edges[:-1] + b_edges[1:]) / 2.0, counts, width=(2 * np.pi)/40, color=self.master_cmap(cluster_id % 10), alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_ylim(0, max_r)
            ax.set_yticklabels([])
            ax.tick_params(axis='x', labelsize=FONTS['tick']-2, pad=4) 
            
            self._add_panel_letter(ax, string.ascii_lowercase[i], is_polar=True)
            
            if inferred_angles and cluster_id in inferred_angles:
                data = inferred_angles[cluster_id]
                rad_orig, rad_corr = np.radians(data['phi_i_deg']), np.radians(data['theta_deg'])
                
                l_bruto = ax.axvline(np.mod(rad_orig, np.pi), color='#1f77b4', linewidth=1.5, linestyle=':')
                ax.axvline(np.mod(rad_orig, np.pi) + np.pi, color='#1f77b4', linewidth=1.5, linestyle=':')
                l_final = ax.axvline(np.mod(rad_corr, np.pi), color='#d62728', linewidth=2.0, linestyle='--')
                ax.axvline(np.mod(rad_corr, np.pi) + np.pi, color='#d62728', linewidth=2.0, linestyle='--')
                
                if not legend_lines: legend_lines = [l_bruto, l_final]
                ax.set_title(rf"$\mathrm{{Región}}\ {cluster_id}$" + "\\n" + rf"$\hat \theta = {data['phi_i_deg']:.1f}^\circ\ |\ \tilde \theta = {data['theta_deg']:.1f}^\circ$", fontsize=FONTS['title']-4, pad=20) 
            else:
                ax.set_title(rf"$\mathrm{{Región}}\ {cluster_id}$", fontsize=FONTS['label'], weight='bold', pad=20) 
                
        if legend_lines:
            fig.legend(handles=legend_lines, labels=[r'Inferencia original ($\hat \theta$)', r'Inferencia corregida ($\tilde \theta$)'],
                       loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=FONTS['legend'], frameon=True, facecolor=self.facecolor)
            
        fig.subplots_adjust(bottom=0.15) 
        self.save_and_show(fig, filename)

    def plot_global_anisotropy_histograms(self, gt_angles: np.ndarray, inferred_angles: np.ndarray, filename: str = "global_histograms.pdf", subtitle: str = "") -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16/1.41, 8/1.41), facecolor=self.facecolor, subplot_kw={'projection': 'polar'})
        bins = np.linspace(0, 2 * np.pi, 41)

        gt_c, b_edges = self._get_symmetric_counts(gt_angles, bins)
        inf_c, _ = self._get_symmetric_counts(inferred_angles, bins)
        max_r = max(1.0, np.max(gt_c), np.max(inf_c))
        
        for idx, (ax, data_counts, title, color) in enumerate(zip(axes, [gt_c, inf_c], [r"$\mathrm{Original}$", r"$\mathrm{Infererido}$"], ['#4c72b0', '#dd8452'])):
            ax.grid(True, alpha=0.4, linestyle=':', color='gray') 
            ax.bar((b_edges[:-1] + b_edges[1:]) / 2.0, data_counts, width=(2 * np.pi)/40, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_ylim(0, max_r)
            ax.set_yticklabels([])
            ax.tick_params(axis='x', labelsize=FONTS['tick'], pad=4)
            self._add_panel_letter(ax, ['a', 'b'][idx], is_polar=True)
            self.apply_paper_style(ax, title=title)
            
        fig.tight_layout()
        self.save_and_show(fig, filename)

    def plot_cluster_hysteresis(self, hysteresis_data: dict, table_path: Path = None, inferred_angles: dict = None, filename: str = "hysteresis_regional.pdf", subtitle: str = "") -> None:
        file_path = table_path or (self.core.in_folder / "table.txt")
        if not file_path.exists(): return
        
        df = load_mumax_table(file_path)
        b_ext, m_global = df['B_extx (T)'].values, df['mx ()'].values
        
        fig, ax = plt.subplots(figsize=(10/1.41, 7.5/1.41), facecolor=self.facecolor)
        self.apply_paper_style(ax, xlabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", ylabel=r"Imanación $m_x$")
        
        common_len_g = min(len(b_ext), len(m_global))
        ax.plot(b_ext[-common_len_g:], m_global[-common_len_g:], color='black', linestyle='-', linewidth=2.0, alpha=0.3, label='Global (Ref)', zorder=1)
        
        styles = itertools.cycle([('-', 'o'), ('--', 's'), ('-.', '^'), (':', 'D')])
        
        for c, mx_vals in hysteresis_data.items():
            common_len = min(len(b_ext), len(mx_vals))
            ls, marker = next(styles)
            ax.plot(b_ext[-common_len:], mx_vals[-common_len:], label=rf"$\mathrm{{Región}}\ {c}$", 
                    color=self.master_cmap(c % 10), linestyle=ls, linewidth=1.5, marker=marker, 
                    markersize=4, markevery=max(1, common_len // 20), alpha=0.9, zorder=2)
            
        ax.legend(frameon=False, fontsize=FONTS['legend'], loc='best', ncol=2)
        self.save_and_show(fig, filename)

    def plot_global_verification_hysteresis(self, table_path: Path, verification_table_path: Path, filename: str = "hysteresis_global_compare.pdf", subtitle: str = "") -> None:
        if not table_path.exists() or not (verification_table_path and verification_table_path.exists()): return
        
        df, v_df = load_mumax_table(table_path), load_mumax_table(verification_table_path)
        b_ext, m_global = df['B_extx (T)'].values, df['mx ()'].values
        v_b, v_m = v_df['B_extx (T)'].values, v_df['mx ()'].values
        
        fig = plt.figure(figsize=(10/1.41, 9/1.41), facecolor=self.facecolor)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.18)
        ax_main, ax_res = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        
        cl_o, cl_v = min(len(b_ext), len(m_global)), min(len(v_b), len(v_m))
        ax_main.plot(b_ext[-cl_o:], m_global[-cl_o:], color='black', linestyle='-', marker='o', markersize=3.5, linewidth=1.2, label='Original', zorder=3)
        ax_main.plot(v_b[-cl_v:], v_m[-cl_v:], color='#d62728', linestyle='--', marker='s', markersize=3.0, linewidth=1.2, label='Inferencia', zorder=2)
        
        self.apply_paper_style(ax_main, ylabel=r"Imanación $m_x$")
        ax_main.legend(frameon=False, fontsize=FONTS['legend'], loc='upper left')
        ax_main.tick_params(labelbottom=False)
        
        idx_o, idx_v = np.argsort(b_ext[-cl_o:]), np.argsort(v_b[-cl_v:])
        b_target, m_target = b_ext[-cl_o:][idx_o], m_global[-cl_o:][idx_o]
        v_m_aligned = np.interp(b_target, v_b[-cl_v:][idx_v], v_m[-cl_v:][idx_v])
        
        residual = v_m_aligned - m_target
        nrmse = np.sqrt(np.mean(residual ** 2)) / (np.max(m_target)-np.min(m_target))
        
        ax_res.axhline(0, color='black', linewidth=1.0, alpha=0.5)
        ax_res.plot(b_target, residual, color='#1f77b4', linestyle='-', marker='d', markersize=3.0, linewidth=1.0)
        ax_res.fill_between(b_target, 0, residual, color='#1f77b4', alpha=0.15)
        
        self.apply_paper_style(ax_res, xlabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", ylabel=r"$\Delta m_x$")
        ax_res.text(0.97, 0.05, f"NRMSE: {nrmse:.1e}", transform=ax_res.transAxes, ha='right', va='bottom', fontsize=FONTS['tick'], family='monospace', bbox=dict(facecolor=self.facecolor, edgecolor='none', alpha=0.8, pad=0))
        
        max_res = max(np.max(np.abs(residual)) * 1.5, 0.01)
        ax_res.set_ylim(-max_res, max_res)
        
        fig.align_ylabels([ax_main, ax_res])
        fig.subplots_adjust(bottom=0.18) 
        self.save_and_show(fig, filename)

    def plot_original_hysteresis_clean(self, table_path: Path, filename: str = "hysteresis_original_clean.pdf") -> None:
        if not table_path.exists(): return
        df = load_mumax_table(table_path)
        
        fig, ax = plt.subplots(figsize=(8.5/1.41, 6.5/1.41), facecolor=self.facecolor)
        self.apply_paper_style(ax, xlabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", ylabel=r"Imanación $m_x$")
        ax.plot(df['B_extx (T)'].values, df['mx ()'].values, color='black', linestyle='-', marker='o', markersize=3.5, linewidth=1.2)
        fig.subplots_adjust(bottom=0.18) 
        self.save_and_show(fig, filename)

    def plot_magnetization_comparison(self, dfs: dict, b_ext: np.ndarray = None, filename: str = "magnetization_frames_compare.pdf", num_frames: int = 4) -> None:
        if not dfs: return
        first_df = list(dfs.values())[0]
        time_cols = first_df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        if len(time_cols) == 0: return
        
        indices = np.linspace(1, len(time_cols) - 1, num_frames, dtype=int)
        extent = [first_df['x (m)'].min(), first_df['x (m)'].max(), first_df['y (m)'].min(), first_df['y (m)'].max()]
        
        n_rows = len(dfs)
        fig, axes = plt.subplots(n_rows, num_frames, figsize=(24, 7.5 * n_rows), facecolor=self.facecolor)
        if n_rows == 1: axes = np.array([axes])
        
        copy_cmap = mcm.get_cmap('twilight_shifted').copy()
        copy_cmap.set_bad(color=self.facecolor, alpha=0.0)
        
        cl = min(len(b_ext), len(time_cols)) if b_ext is not None else 0
        b_offset, t_offset = (len(b_ext) - cl, len(time_cols) - cl) if cl > 0 else (0, 0)
        
        for r_idx, (row_name, df) in enumerate(dfs.items()):
            for c_idx, frame_idx in enumerate(indices):
                ax = axes[r_idx][c_idx] if n_rows > 1 else axes[c_idx]
                ax.set_facecolor('#e8e5d1') 
                
                grid = df.pivot(index='y (m)', columns='x (m)', values=time_cols[frame_idx]).values
                im = ax.imshow(grid, cmap=copy_cmap, vmin=-np.pi, vmax=np.pi, origin='lower', extent=extent)
                
                self._add_panel_letter(ax, f"{string.ascii_lowercase[r_idx]}{c_idx + 1}")
                
                if r_idx == 0:
                    title_str = rf"$\mathbf{{H}}^\text{{ext}} = {b_ext[b_offset + (frame_idx - t_offset)]:.2f}\ \mathrm{{T}}$" if (cl > 0 and frame_idx >= t_offset) else rf"Fotograma {c_idx + 1}"
                    ax.text(0.5, 1.06, title_str, transform=ax.transAxes, fontsize=FONTS['title'], weight='bold', ha='center', va='bottom')
                
                if c_idx == 0:
                    ax.text(-0.25, 0.5, row_name, transform=ax.transAxes, rotation=90, va='center', ha='center', weight='bold', fontsize=FONTS['title'])
                
                ax.set_aspect('equal')
                ax.axis('off')
                
        fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.15, wspace=0.05, hspace=0.10)
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
        self._add_angle_colorbar(fig, im, cax=cbar_ax, orientation='horizontal')
        
        self.save_and_show(fig, filename)

    def plot_clustering_features_grid(self, dfs: dict, b_ext: np.ndarray, filename: str = "clustering_features_B0.pdf") -> None:
        if not dfs: return
        b_val, zero_idx = b_ext[np.argmin(np.abs(b_ext))], np.argmin(np.abs(b_ext))
        
        fig, axes = plt.subplots(2, 2, figsize=(16/1.41, 15/1.41), facecolor=self.facecolor)
        axes = axes.flatten()
        
        for idx, (name, df) in enumerate(list(dfs.items())[:4]):
            ax = axes[idx]
            ax.set_facecolor('#e8e5d1') 
            
            t_cols = df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
            if len(t_cols) == 0: continue
            
            cl = min(len(b_ext), len(t_cols))
            target_idx = max(0, min((len(t_cols) - cl) + (zero_idx - (len(b_ext) - cl)), len(t_cols) - 1))
            
            grid_data = df.pivot(index='y (m)', columns='x (m)', values=t_cols[target_idx]).values
            extent = [df['x (m)'].min(), df['x (m)'].max(), df['y (m)'].min(), df['y (m)'].max()]
            
            name_low = name.lower()
            if 'angle' in name_low and not name_low.startswith('d_'):
                cmap, vmin, vmax = mcm.get_cmap('twilight_shifted').copy(), -np.pi, np.pi
                c_ticks, c_lbls = [-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$']
                title_str = r"$\phi$"
            elif 'd_angle' in name_low or 'd_norm' in name_low:
                cmap = mcm.get_cmap('coolwarm').copy()
                vmin, vmax = -np.nanmax(np.abs(grid_data)), np.nanmax(np.abs(grid_data))
                c_ticks, c_lbls = None, None
                title_str = r"$\dot{\phi}$" if "d_angle" in name_low else r"$\dot{m}_\text{dif}$"
            else:
                cmap, vmin, vmax = mcm.get_cmap('viridis').copy(), np.nanmin(grid_data), np.nanmax(grid_data)
                c_ticks, c_lbls = None, None
                title_str = r"$m_\text{dif}$" if "norm" in name_low else rf"$\mathrm{{{name}}}$"
                
            cmap.set_bad(color=self.facecolor, alpha=0.0)
            im = ax.imshow(grid_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
            
            self._add_panel_letter(ax, string.ascii_lowercase[idx], is_polar=False)
            ax.text(0.5, 1.05, title_str, transform=ax.transAxes, fontsize=FONTS['title'], weight='bold', ha='center', va='bottom')
            ax.set_aspect('equal')
            ax.axis('off')
            
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=FONTS['tick'])
            if c_ticks: cbar.set_ticks(c_ticks); cbar.ax.set_yticklabels(c_lbls)
                
        for i in range(len(dfs), 4): axes[i].axis('off')
            
        fig.suptitle(rf"$\mathrm{{Parametros\ de\ agrupamiento\ en\ }}\mathbf{{H}}^\text{{ext}} = {b_val:.2f}\ \mathrm{{T}}$", fontsize=FONTS['letter'], weight='bold', y=1.02)
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, wspace=0.3, hspace=0.3)
        self.save_and_show(fig, filename)

    def plot_spatiotemporal_map(self, df: pd.DataFrame, b_ext: np.ndarray, filename: str = "spatiotemporal_map.pdf") -> None:
        time_cols = df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        if len(time_cols) == 0: return
        
        y_center = df['y (m)'].unique()[len(df['y (m)'].unique())//2]
        df_slice = df[df['y (m)'] == y_center].sort_values('x (m)')
        
        cl = min(len(b_ext), len(time_cols))
        mapa_2d = df_slice[time_cols].values.T[-cl:]
        
        fig, ax = plt.subplots(figsize=(14, 9), facecolor=self.facecolor)
        extent = [df_slice['x (m)'].min()*1e9, df_slice['x (m)'].max()*1e9, b_ext[-cl:].min(), b_ext[-cl:].max()]
        
        im = ax.imshow(mapa_2d, aspect='auto', origin='lower', cmap='twilight_shifted', extent=extent, vmin=-np.pi, vmax=np.pi)
        self.apply_paper_style(ax, xlabel="Posición en $x$ (nm)", ylabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", title="Evolución Espaciotemporal (Corte Central)")
        self._add_angle_colorbar(fig, im, ax=ax, orientation='vertical', pad=0.03)
        
        fig.tight_layout()
        self.save_and_show(fig, filename)

    # =========================================================================
    # UNIFIED ANIMATION PIPELINE
    # =========================================================================

    def animation(self, target=None, aux_df=None, filename=None, fps=2, quantity=None, **kwargs):
        tgt = target or quantity or str(Quantity.ANGLE)
        if isinstance(tgt, Path):
            self._animation_direct_sequential(tgt, aux_df, filename or "animacion_directa.mp4", fps)
        else:
            self._animation_variable(tgt, fps=fps, override_df=aux_df)

    def _animation_variable(self, quantity: str, fps: int = 2, override_df: pd.DataFrame | None = None) -> None:
        if not self.plotting: return
        df = override_df if override_df is not None else self.core.get_df(quantity)
        time_cols = df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        if len(time_cols) == 0:
            logger.warning(f"❌ No se encontraron columnas de tiempo válidas para {quantity}. Animación cancelada.")
            return

        initial = df.pivot(index='y (m)', columns='x (m)', values=time_cols[0]).values
        max_abs = max(np.abs(initial).max(), 1e-6)
        
        cmap, vmin, vmax, label = {
            str(Quantity.ANGLE): ('twilight_shifted', -np.pi, np.pi, r'Ángulo XY $\phi$ (rad)'),
            str(Quantity.D_ANGLE): ('coolwarm', -max_abs, max_abs, r'Velocidad Angular $\partial\phi/\partial t$')
        }.get(quantity, ('magma', 0.0, max(1.0, max_abs), f'Magnitud ({quantity})'))
        
        fig, ax = plt.subplots(figsize=(12/1.41, 10/1.41), facecolor=self.facecolor)
        extent = [df['x (m)'].min(), df['x (m)'].max(), df['y (m)'].min(), df['y (m)'].max()]
        im = ax.imshow(initial, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', extent=extent, aspect='equal')
        
        if quantity == str(Quantity.ANGLE): self._add_angle_colorbar(fig, im, ax=ax, orientation='vertical', label=label)
        else:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=FONTS['tick'])
            cbar.set_label(label, fontsize=FONTS['label'], weight='bold')
            
        ax.axis('off') 

        def update(frame_idx):
            z_data = df.pivot(index='y (m)', columns='x (m)', values=time_cols[frame_idx]).values
            im.set_data(z_data)
            if quantity == str(Quantity.D_ANGLE): im.set_clim(vmax=max(np.abs(z_data).max(), 1e-6), vmin=-max(np.abs(z_data).max(), 1e-6))
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=len(time_cols), interval=1000/fps, blit=True)
        self._save_animation(anim, self.core.plot_dir / f"{self.core.in_folder.name}_{quantity}.mp4", fps)

    def _animation_direct_sequential(self, yes_folder: Path, auxiliary_df: pd.DataFrame, output_filename: str, fps: int):
        import discretisedfield as dfield
        if not yes_folder.exists(): return logger.error(f"❌ '{yes_folder}' missing.")

        files = sorted([f for f in yes_folder.iterdir() if f.suffix in ('.ovf', '.omf')])
        if not files: return

        region = dfield.Field.from_file(files[0]).mesh.region
        extent = [region.pmin[0]*1e9, region.pmax[0]*1e9, region.pmin[1]*1e9, region.pmax[1]*1e9]
        frames_angles = [np.arctan2(dfield.Field.from_file(f).y.array.squeeze(), dfield.Field.from_file(f).x.array.squeeze()) for f in files]

        fig, ax = plt.subplots(figsize=(11, 9), facecolor=self.facecolor)
        ax.set_facecolor('#e8e5d1') 

        im = ax.imshow(frames_angles[0], cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi, origin='lower', extent=extent, aspect='equal')
        self.apply_paper_style(ax, xlabel="Posición $x$ (nm)", ylabel="Posición $y$ (nm)", title="Evolución de la Imanación")
        self._add_angle_colorbar(fig, im, ax=ax, orientation='vertical', pad=0.04)

        fig.tight_layout()
        anim = animation.FuncAnimation(fig, lambda i: [im.set_data(frames_angles[i]) or im], frames=len(files), blit=True)
        self._save_animation(anim, self.core.plot_dir / output_filename, fps)