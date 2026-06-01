import logging
from pathlib import Path
import string

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.data_core import Quantity

# ⚡ NUEVO: Carga opcional de SciencePlots
try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

logger = logging.getLogger(__name__)

class Visualizer:
    """Motor unificado y estandarizado para gráficas científicas (Paper-Ready)."""

    def __init__(self, data_core, facecolor: str = '#ffffff', plotting: bool = True, use_scienceplots: bool = False):
        self.core = data_core
        self.facecolor = facecolor 
        self.plotting = plotting
        
        # ⚡ NUEVO: Lógica de activación de SciencePlots
        self.use_scienceplots = use_scienceplots and HAS_SCIENCEPLOTS
        
        if use_scienceplots and not HAS_SCIENCEPLOTS:
            logger.warning("SciencePlots solicitado en CONFIG pero no instalado. Ejecuta 'pip install SciencePlots'. Usando fallback manual.")
            
        if self.use_scienceplots:
            # Usamos 'ieee' para el estándar estricto de ingeniería y 'no-latex' 
            # como salvavidas por si tu PC de Windows no tiene TeX Live nativo instalado.
            plt.style.use(['science', 'ieee', 'no-latex'])
            logger.info("SciencePlots cargado correctamente con perfiles 'science' e 'ieee'.")
            
        self.master_cmap = plt.get_cmap('tab10')

    def save_and_show(self, fig: plt.Figure, filename: str) -> None:
        if not self.plotting:
            plt.close(fig)
            return
            
        if not filename.endswith('.pdf'):
            filename = filename.replace('.png', '.pdf')
            
        fig.savefig(self.core.plot_dir / filename, dpi=300, bbox_inches='tight', facecolor=self.facecolor)
        plt.close(fig)

    def apply_paper_style(self, ax: plt.Axes, xlabel: str = "", ylabel: str = ""):
        """Estilo cartesiano riguroso: sin título interior, marcas hacia adentro."""
        if not self.use_scienceplots:
            ax.set_facecolor('white')
            
        # ⚡ SOLUCIÓN DE SOLAPAMIENTO: Añadimos 'labelpad' para empujar los títulos
        # hacia afuera y evitar que LaTeX pise los números del eje.
        if xlabel: ax.set_xlabel(xlabel, fontsize=10, weight='bold', labelpad=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10, weight='bold', labelpad=8)
            
        if not self.use_scienceplots:
            ax.tick_params(axis='both', which='major', labelsize=8, direction='in', top=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
            ax.grid(True, linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    def _add_panel_letter(self, ax: plt.Axes, letter: str, is_polar: bool = False):
        """Etiquetas universales (a), (b), (c) para maquetación multipanel."""
        x_pos = -0.15 if is_polar else -0.12
        y_pos = 1.15 if is_polar else 1.05
        ax.text(x_pos, y_pos, f'({letter})', transform=ax.transAxes, fontweight='bold', fontsize=11)

    def plot_cluster_anisotropy_polar(self, cluster_anis: dict, inferred_angles: dict = None, filename: str = "cluster_polar.pdf", subtitle: str = "") -> None:
        n_clusters = len(cluster_anis)
        if n_clusters == 0: return
        
        bins = np.linspace(0, 2 * np.pi, 41)
        cluster_data = {}
        all_counts = []
        
        for cluster_id, angles in cluster_anis.items():
            valid = angles[~np.isnan(angles)]
            if len(valid) == 0: continue
            symmetric = np.concatenate([np.mod(valid, np.pi), np.mod(valid, np.pi) + np.pi])
            counts, b_edges = np.histogram(symmetric, bins=bins)
            cluster_data[cluster_id] = (counts / 2.0, b_edges)
            all_counts.append(np.max(counts / 2.0))
        
        max_r = max(all_counts) if all_counts else 1.0
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        # Ampliamos ligeramente la altura del lienzo (figsize) para acomodar las filas holgadamente
        fig = plt.figure(figsize=((6.5 * cols)/2.54, (7.0 * rows + 2.0)/2.54), facecolor=self.facecolor)
        
        # ⚡ CRÍTICO: Controlamos el espaciado con GridSpec fijando hspace=0.45 para separar las filas
        gs = fig.add_gridspec(rows, cols, wspace=0.25, hspace=0.45)
        
        labels_letters = list(string.ascii_lowercase)
        legend_lines = []
        
        for i, (cluster_id, (counts, b_edges)) in enumerate(cluster_data.items()):
            row_idx = i // cols
            col_idx = i % cols
            
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='polar')
            
            # Rotamos la orientación para despejar la zona inferior de conflictos
            ax.set_theta_zero_location('S')
            ax.grid(True, alpha=0.4, linestyle=':', color='gray') 
            
            ax.bar((b_edges[:-1] + b_edges[1:]) / 2.0, counts, width=(2 * np.pi)/40, bottom=0.0, color=self.master_cmap(cluster_id % 10), alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_ylim(0, max_r)
            ax.set_yticklabels([])
            ax.tick_params(axis='x', labelsize=7, pad=4)
            
            self._add_panel_letter(ax, labels_letters[i], is_polar=True)
            
            if inferred_angles and cluster_id in inferred_angles:
                data = inferred_angles[cluster_id]
                
                l_bruto = ax.axvline(np.mod(np.radians(data['phi_i_deg']), np.pi), color='#1f77b4', linewidth=1.5, linestyle=':')
                ax.axvline(np.mod(np.radians(data['phi_i_deg']), np.pi) + np.pi, color='#1f77b4', linewidth=1.5, linestyle=':')
                l_final = ax.axvline(np.mod(np.radians(data['theta_deg']), np.pi), color='#d62728', linewidth=2.0, linestyle='--')
                ax.axvline(np.mod(np.radians(data['theta_deg']), np.pi) + np.pi, color='#d62728', linewidth=2.0, linestyle='--')
                
                if not legend_lines: legend_lines = [l_bruto, l_final]
                
                panel_txt = rf"$\mathrm{{Región}}\ {cluster_id}$" + "\n" + rf"$\hat \theta = {data['phi_i_deg']:.1f}^\circ\ |\ \tilde \theta = {data['theta_deg']:.1f}^\circ$"
                
                # Al no haber tight_layout destructivo, este pad de 15 se respetará de forma estricta
                ax.set_title(panel_txt, fontsize=8, weight='normal', pad=15)
            else:
                ax.set_title(rf"$\mathrm{{Región}}\ {cluster_id}$", fontsize=9, weight='bold', pad=15)
                
        if legend_lines:
            fig.legend(
                handles=legend_lines, labels=[r'Inferencia original ($\hat \theta$)', r'Inferencia corregida ($\tilde \theta$)'],
                loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=9,
                frameon=True, facecolor='white', edgecolor='#cccccc', fancybox=False
            )
            
        # ⚡ CRÍTICO: Eliminamos fig.tight_layout() para que no destruya la maquetación polar
        fig.subplots_adjust(bottom=0.15) 
        self.save_and_show(fig, filename)

    def plot_global_anisotropy_histograms(self, gt_angles: np.ndarray, inferred_angles: np.ndarray, filename: str = "global_histograms.pdf", subtitle: str = "") -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15/2.54, 7/2.54), facecolor=self.facecolor, subplot_kw={'projection': 'polar'})
        bins = np.linspace(0, 2 * np.pi, 41)

        def get_sym_counts(angles):
            v = angles[~np.isnan(angles)]
            if len(v) == 0: return np.zeros(40), bins
            return np.histogram(np.concatenate([np.mod(v, np.pi), np.mod(v, np.pi) + np.pi]), bins=bins)[0] / 2.0, bins
            
        gt_c, b_edges = get_sym_counts(gt_angles)
        inf_c, _ = get_sym_counts(inferred_angles)
        max_r = max(1.0, max(np.max(gt_c), np.max(inf_c)))
        
        titles = [r"$\mathrm{Original}$", r"$\mathrm{Infererido}$"]
        colors = ['#4c72b0', '#dd8452']
        data_counts = [gt_c, inf_c]
        
        for idx, ax in enumerate(axes):
            ax.grid(True, alpha=0.4, linestyle=':', color='gray') 
            ax.bar((b_edges[:-1] + b_edges[1:]) / 2.0, data_counts[idx], width=(2 * np.pi)/40, bottom=0.0, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_ylim(0, max_r)
            ax.set_yticklabels([])
            ax.tick_params(axis='x', labelsize=8, pad=2)
            
            self._add_panel_letter(ax, ['a', 'b'][idx], is_polar=True)
            ax.set_title(titles[idx], fontsize=9, weight='bold', pad=12)
            
        fig.tight_layout()
        self.save_and_show(fig, filename)

    def plot_cluster_hysteresis(self, hysteresis_data: dict, table_path: Path = None, inferred_angles: dict = None, filename: str = "hysteresis_regional.pdf", subtitle: str = "") -> None:
        file_path = table_path if table_path is not None else (self.core.in_folder / "table.txt")
        if not file_path.exists(): return
        
        df = pd.read_csv(file_path, sep='\t').rename(columns=lambda x: x.strip().replace('# ', ''))
        b_ext, m_global = df['B_extx (T)'].values, df['mx ()'].values
        
        fig, ax = plt.subplots(figsize=(8.5/2.54, 6.5/2.54), facecolor=self.facecolor)
        self.apply_paper_style(ax, xlabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", ylabel=r"Imanación $m_x$")
        
        common_len_g = min(len(b_ext), len(m_global))
        ax.plot(b_ext[-common_len_g:], m_global[-common_len_g:], color='black', linestyle='-', linewidth=2.0, alpha=0.3, label='Global (Ref)', zorder=1)
        
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']
        
        for idx, (c, mx_vals) in enumerate(hysteresis_data.items()):
            common_len = min(len(b_ext), len(mx_vals))
            current_b = b_ext[-common_len:]
            current_mx = mx_vals[-common_len:]
            
            ax.plot(current_b, current_mx, label=rf"$\mathrm{{Región}}\ {c}$", 
                    color=self.master_cmap(c % 10), linestyle=linestyles[idx % len(linestyles)], 
                    linewidth=1.5, marker=markers[idx % len(markers)], markersize=4, 
                    markevery=max(1, len(current_b) // 20), alpha=0.9, zorder=2)
            
        ax.legend(frameon=False, fontsize=8, loc='best', ncol=2)
        self.save_and_show(fig, filename)

    def plot_global_verification_hysteresis(self, table_path: Path, verification_table_path: Path, filename: str = "hysteresis_global_compare.pdf", subtitle: str = "") -> None:
        if not table_path.exists() or not (verification_table_path and verification_table_path.exists()): return
        
        df = pd.read_csv(table_path, sep='\t').rename(columns=lambda x: x.strip().replace('# ', ''))
        b_ext, m_global = df['B_extx (T)'].values, df['mx ()'].values

        v_df = pd.read_csv(verification_table_path, sep='\t').rename(columns=lambda x: x.strip().replace('# ', ''))
        v_b, v_m = v_df['B_extx (T)'].values, v_df['mx ()'].values
        
        fig = plt.figure(figsize=(8.5/2.54, 8/2.54), facecolor=self.facecolor)
        
        # ⚡ SOLUCIÓN DE MARGEN: Incrementamos hspace de 0.1 a 0.18 para distanciar los dos paneles verticales
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.18)
        
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)
        
        common_len_o, common_len_v = min(len(b_ext), len(m_global)), min(len(v_b), len(v_m))
        
        # ⚡ MARCADORES: Modificado a puntos discretos conectados por círculos y cuadrados
        ax_main.plot(b_ext[-common_len_o:], m_global[-common_len_o:], color='black', linestyle='-', marker='o', markersize=3.5, linewidth=1.2, label='Original', zorder=3)
        ax_main.plot(v_b[-common_len_v:], v_m[-common_len_v:], color='#d62728', linestyle='--', marker='s', markersize=3.0, linewidth=1.2, label='Inferencia', zorder=2)
        
        self.apply_paper_style(ax_main, ylabel=r"Imanación $m_x$")
        ax_main.legend(frameon=False, fontsize=8, loc='upper left')
        ax_main.tick_params(labelbottom=False)
        
        idx_o, idx_v = np.argsort(b_ext[-common_len_o:]), np.argsort(v_b[-common_len_v:])
        b_target, m_target = b_ext[-common_len_o:][idx_o], m_global[-common_len_o:][idx_o]
        v_m_aligned = np.interp(b_target, v_b[-common_len_v:][idx_v], v_m[-common_len_v:][idx_v])
        
        residual = v_m_aligned - m_target
        rmse = np.sqrt(np.mean(residual ** 2))
        nrmse = rmse / (np.max(m_target)-np.min(m_target))
        
        ax_res.axhline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.5)
        
        # ⚡ MARCADORES: Puntos discretos también en el panel inferior de error residual
        ax_res.plot(b_target, residual, color='#1f77b4', linestyle='-', marker='d', markersize=3.0, linewidth=1.0)
        ax_res.fill_between(b_target, 0, residual, color='#1f77b4', alpha=0.15)
        
        self.apply_paper_style(ax_res, xlabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", ylabel=r"$\Delta m_x$")
        
        ax_res.text(0.97, 0.05, f"NRMSE: {nrmse:.1e}", transform=ax_res.transAxes, ha='right', va='bottom', fontsize=8, family='monospace', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0))
        
        max_res = np.max(np.abs(residual)) * 1.5 if np.max(np.abs(residual)) > 0 else 0.01
        ax_res.set_ylim(-max_res, max_res)
        
        fig.align_ylabels([ax_main, ax_res])
        
        # ⚡ SOLUCIÓN DE MARGEN: Holgura inferior extra para que el labelpad y el título quepan perfectamente
        fig.subplots_adjust(bottom=0.18) 
        self.save_and_show(fig, filename)

    def animation(self, quantity: str = 'angle', fps: int = 2, override_df: pd.DataFrame | None = None) -> None:
        if not self.plotting: return
        
        df = override_df if override_df is not None else self.core.get_df(quantity)
        
        # ⚡ FIX CRÍTICO: Filtramos estrictamente los archivos de magnetización (mXXXXXX.ovf),
        # excluyendo el mapa de anisotropía inicial que se colaba como primer frame.
        time_cols = df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        if len(time_cols) == 0: return

        initial_frame = df.pivot(index='y (m)', columns='x (m)', values=time_cols[0]).values
        max_abs = max(np.abs(initial_frame).max(), 1e-6)
        
        if quantity == str(Quantity.ANGLE): 
            cmap, vmin, vmax, label = 'twilight_shifted', -np.pi, np.pi, r'Ángulo XY $\phi$ (rad)'
        elif quantity == str(Quantity.D_ANGLE): 
            cmap, vmin, vmax, label = 'coolwarm', -max_abs, max_abs, r'Velocidad Angular $\partial\phi/\partial t$'
        else: 
            cmap, vmin, vmax, label = 'magma', 0.0, max(1.0, max_abs), f'Magnitud ({quantity})'
        
        # ⚡ AUMENTADO: Lienzo mucho más grande para el renderizado del vídeo MP4
        fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54), facecolor=self.facecolor)
        ax.set_facecolor('white')
        
        extent = [df['x (m)'].min(), df['x (m)'].max(), df['y (m)'].min(), df['y (m)'].max()]
        im = ax.imshow(initial_frame, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', extent=extent, aspect='equal')
        
        cbar = fig.colorbar(im, ax=ax)
        if quantity == str(Quantity.ANGLE):
            cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
            
        cbar.set_label(label, fontsize=9, weight='bold')
        ax.axis('off') 

        def update(frame_idx):
            col_name = time_cols[frame_idx]
            z_data = df.pivot(index='y (m)', columns='x (m)', values=col_name).values
            im.set_data(z_data)
            if quantity == str(Quantity.D_ANGLE): 
                im.set_clim(vmax=max(np.abs(z_data).max(), 1e-6), vmin=-max(np.abs(z_data).max(), 1e-6))
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=len(time_cols), interval=1000/fps, blit=True)
        
        out_video_path = self.core.plot_dir / f"{self.core.in_folder.name}_{quantity}.mp4"
        anim.save(out_video_path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000)) # Bitrate alto para nitidez
        plt.close(fig)

    # ⚡ NUEVO: Generador individual de la Histéresis Limpia (Sin Inferencia IA ni Residuales)
    def plot_original_hysteresis_clean(self, table_path: Path, filename: str = "hysteresis_original_clean.pdf") -> None:
        if not table_path.exists(): return
        
        df = pd.read_csv(table_path, sep='\t').rename(columns=lambda x: x.strip().replace('# ', ''))
        b_ext, m_global = df['B_extx (T)'].values, df['mx ()'].values
        
        fig, ax = plt.subplots(figsize=(8.5/2.54, 6.5/2.54), facecolor=self.facecolor)
        self.apply_paper_style(ax, xlabel=r"Campo Externo $\mathbf{H}^\text{ext}$ (T)", ylabel=r"Imanación $m_x$")
        
        # ⚡ MARCADORES: Representación realista de datos discretos conectados
        ax.plot(b_ext, m_global, color='black', linestyle='-', marker='o', markersize=3.5, linewidth=1.2)
        
        # ⚡ AJUSTE: Mismo margen inferior para que LaTeX respire
        fig.subplots_adjust(bottom=0.18) 
        self.save_and_show(fig, filename)

    def plot_magnetization_comparison(self, dfs: dict, b_ext: np.ndarray = None, filename: str = "magnetization_frames_compare.pdf", num_frames: int = 4) -> None:
        """Genera una cuadrícula NxFrames comparando la evolución espacial de la magnetización."""
        if not dfs: return
        
        first_df = list(dfs.values())[0]
        time_cols = first_df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
        if len(time_cols) == 0: return
        
        indices = np.linspace(1, len(time_cols) - 1, num_frames, dtype=int)
        selected_cols = time_cols[indices]
        
        n_rows = len(dfs)
        n_cols = num_frames
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(17.5/2.54, (4.5 * n_rows + 1.5)/2.54), facecolor='#ffffff')
        if n_rows == 1: axes = [axes]
        
        extent = [first_df['x (m)'].min(), first_df['x (m)'].max(), first_df['y (m)'].min(), first_df['y (m)'].max()]
        labels_letters = list(string.ascii_lowercase)
        
        import matplotlib.cm as mcm
        base_cmap = mcm.get_cmap('twilight_shifted')
        copy_cmap = base_cmap.copy()
        copy_cmap.set_bad(color='white', alpha=0.0)
        
        vmin, vmax = -np.pi, np.pi
        
        # ⚡ LÓGICA DE ALINEACIÓN: Sincroniza los tensores de tiempo con el array del campo externo
        common_len = 0
        if b_ext is not None:
            common_len = min(len(b_ext), len(time_cols))
            time_offset = len(time_cols) - common_len
            b_offset = len(b_ext) - common_len
        
        for row_idx, (row_name, df) in enumerate(dfs.items()):
            for col_idx, (frame_idx, col_name) in enumerate(zip(indices, selected_cols)):
                ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]
                ax.set_facecolor('#f5f5f5')
                
                grid_data = df.pivot(index='y (m)', columns='x (m)', values=col_name).values
                mapped_data = np.where(grid_data == self.core.LABEL_EMPTY if hasattr(self.core, 'LABEL_EMPTY') else np.isnan(grid_data), np.nan, grid_data)
                
                im = ax.imshow(grid_data, cmap=copy_cmap, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
                
                # ⚡ FIX 1: Notación matricial para paneles (a1, a2, b1, b2...)
                letter_str = f"{labels_letters[row_idx]}{col_idx + 1}"
                self._add_panel_letter(ax, letter_str, is_polar=False)
                
                if row_idx == 0:
                    # ⚡ FIX 2: Título físico cruzando el frame temporal con el barrido del campo B_ext
                    if b_ext is not None and common_len > 0 and frame_idx >= time_offset:
                        b_idx = b_offset + (frame_idx - time_offset)
                        b_val = b_ext[b_idx]
                        title_str = rf"$\mathbf{{H}}^\text{{ext}} = {b_val:.2f}\ \mathrm{{T}}$"
                    else:
                        frame_num = int(col_name.replace('m', '').replace('.ovf', '').replace('.omf', ''))
                        title_str = rf"$\mathbf{{H}}^\text{{ext}} = {0.5}\ \mathrm{{T}}$"

                    ax.text(0.5, 1.06, title_str, transform=ax.transAxes,
                            fontsize=10, weight='bold', ha='center', va='bottom')
                
                if col_idx == 0:
                    ax.text(-0.25, 0.5, row_name, transform=ax.transAxes, rotation=90, va='center', ha='center', weight='bold', fontsize=10)
                
                ax.set_aspect('equal')
                ax.axis('off')
                
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15, wspace=0.05, hspace=0.15)
        
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r'Ángulo de imanación ($\phi$)', weight='bold', fontsize=9)
        
        self.save_and_show(fig, filename)

    # =========================================================================
    # ⚡ NUEVO: GRID 2x2 DE CARACTERÍSTICAS DE CLUSTERING EN CAMPO CERO
    # =========================================================================
    def plot_clustering_features_grid(self, dfs: dict, b_ext: np.ndarray, filename: str = "clustering_features_B0.pdf") -> None:
        """Genera un grid 2x2 con las 4 características alimentadas a la IA en el cruce por B=0."""
        if not dfs: return
        
        # Encontramos el índice donde el campo magnético es más cercano a 0 Teslas
        zero_idx = np.argmin(np.abs(b_ext))
        b_val = b_ext[zero_idx]
        
        # Preparamos un lienzo de doble columna cuadrada
        fig, axes = plt.subplots(2, 2, figsize=(16/2.54, 15/2.54), facecolor='#ffffff')
        axes = axes.flatten()
        
        labels_letters = list(string.ascii_lowercase)
        import matplotlib.cm as mcm
        
        for idx, (name, df) in enumerate(list(dfs.items())[:4]):
            ax = axes[idx]
            ax.set_facecolor('#f5f5f5') # Fondo sutil para dar contraste a la geometría
            
            time_cols = df.filter(regex=r'^(d_)?m\d{6}\.(ovf|omf)$').columns
            if len(time_cols) == 0: continue
            
            # Sincronización temporal con el barrido magnético
            common_len = min(len(b_ext), len(time_cols))
            time_offset = len(time_cols) - common_len
            b_offset = len(b_ext) - common_len
            
            target_time_idx = time_offset + (zero_idx - b_offset)
            target_time_idx = max(0, min(target_time_idx, len(time_cols) - 1))
            col_name = time_cols[target_time_idx]
            
            # Extracción y aislamiento geométrico
            grid_data = df.pivot(index='y (m)', columns='x (m)', values=col_name).values
            extent = [df['x (m)'].min(), df['x (m)'].max(), df['y (m)'].min(), df['y (m)'].max()]
            
            # Lógica inteligente de colormaps según la variable física
            if 'angle' in name.lower() and not name.lower().startswith('d_'):
                cmap = mcm.get_cmap('twilight_shifted').copy()
                vmin, vmax = -np.pi, np.pi
                cbar_ticks = [-np.pi, 0, np.pi]
                cbar_labels = [r'$-\pi$', r'$0$', r'$\pi$']
            elif 'd_angle' in name.lower() or 'd_norm' in name.lower():
                cmap = mcm.get_cmap('coolwarm').copy()
                max_abs = np.nanmax(np.abs(grid_data))
                vmin, vmax = -max_abs, max_abs
                cbar_ticks = None
                cbar_labels = None
            else:
                cmap = mcm.get_cmap('viridis').copy()
                vmin, vmax = np.nanmin(grid_data), np.nanmax(grid_data)
                cbar_ticks = None
                cbar_labels = None
                
            cmap.set_bad(color='white', alpha=0.0)
            im = ax.imshow(grid_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
            
            self._add_panel_letter(ax, labels_letters[idx], is_polar=False)
            
            # ⚡ MAPEO SEGURO: Mapeamos explícitamente cada clave a un entorno de LaTeX blindado y limpio
            name_lower = name.lower()
            if "d_angle" in name_lower:
                title_str = r"$\dot{\phi}$"
            elif "d_norm" in name_lower:
                title_str = r"$\dot{m}_\text{dif}$"
            elif "angle" in name_lower:
                title_str = r"$\phi$"
            elif "norm" in name_lower:
                title_str = r"$m_\text{dif}$"
            else:
                title_str = rf"$\mathrm{{{name}}}$"
                
            ax.text(0.5, 1.05, title_str, transform=ax.transAxes, fontsize=10, weight='bold', ha='center', va='bottom')
            
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Barra de color individual ajustada a la altura del disco
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
                cbar.ax.set_yticklabels(cbar_labels)
                
        # Apagamos los ejes sobrantes si se hubieran metido menos de 4 cantidades
        for idx in range(len(dfs), 4):
            axes[idx].axis('off')
            
        fig.suptitle(rf"$\mathrm{{Parametros\ de\ agrupamiento\ en\ }}\mathbf{{H}}^\text{{ext}} = {b_val:.2f}\ \mathrm{{T}}$", fontsize=11, weight='bold', y=0.98)
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, wspace=0.3, hspace=0.3)
        self.save_and_show(fig, filename)