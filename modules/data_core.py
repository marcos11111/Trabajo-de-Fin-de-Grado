import gc
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import itertools

import discretisedfield as dfield
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class MaterialProps:
    """Dataclass holding static intrinsic materials parameter settings."""
    A: float = 1.3e-11        
    Ms: float = 8e5           
    Ku1: float = 2.5e5        
    dz: float = 20e-9         
    kappa: float = 1.0        

def load_mumax_table(table_path: Path) -> pd.DataFrame:
    """Parses text log arrays out of standard tabular MuMax3 output layouts."""
    if not table_path.exists():
        raise FileNotFoundError(f"Simulation matrix missing at: {table_path}")
    df = pd.read_csv(table_path, sep='\t')
    df.columns = df.columns.str.strip().str.replace('# ', '')
    return df

def generate_hybrid_selections(variables, fixed=None):
    """Generates all combinations for experimental feature configuration mapping."""
    if fixed is None: fixed = []
    if variables is None: return [fixed]
    s = list(variables)
    variable_combinations = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
    return [list(fixed) + list(comb) for comb in variable_combinations]

class Quantity(str, Enum):
    ANGLE = 'angle'
    D_ANGLE = 'd_angle'
    NORM = 'norm'
    D_NORM = 'd_norm'
    NORM_XY = 'norm_xy'
    D_NORM_XY = 'd_norm_xy'

    def __str__(self):
        return self.value

def _extract_single_ovf(file_path: Path, quantity: Quantity) -> tuple[str, np.ndarray]:
    field = dfield.Field.from_file(file_path)
    mx = field.x.array.squeeze().flatten()
    my = field.y.array.squeeze().flatten()

    if quantity == Quantity.ANGLE:
        return file_path.name, (np.arctan2(my, mx) + np.pi) % (2 * np.pi) - np.pi
    if quantity == Quantity.NORM_XY:
        return file_path.name, np.sqrt(mx**2 + my**2)
    if quantity == Quantity.NORM:
        mz = field.z.array.squeeze().flatten()
        return file_path.name, np.sqrt(mx**2 + my**2 + mz**2)
    return file_path.name, np.zeros_like(mx)

class DataCore:
    """I/O high-performance extraction layer translating OVF tensors to PyArrow columns."""
    
    def __init__(self, in_folder: Path, project_data_dir: Path, project_plot_dir: Path, anis_path: str = "anisU000000.ovf"):
        self.in_folder = in_folder
        self.data_dir = project_data_dir / in_folder.name
        self.plot_dir = project_plot_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.anis_path = anis_path

    def free_memory(self, quantities: list[Quantity] | None = None) -> None:
        if quantities is None:
            self.dataframes.clear()
        else:
            for qty in quantities:
                self.dataframes.pop(str(qty), None)
        gc.collect()

    def check_loaded(self, quantity: Quantity = Quantity.ANGLE) -> None:
        if str(quantity) not in self.dataframes:
            self.load_data([quantity])

    def get_df(self, quantity: Quantity | str) -> pd.DataFrame:
        self.check_loaded(quantity)
        return self.dataframes[str(quantity)]

    def load_data(self, quantities: list[Quantity] | None = None) -> None:
        quantities = quantities or [Quantity.ANGLE]
        for qty in quantities:
            qty_str = str(qty)
            if qty_str in self.dataframes: continue
            
            pqt_out_path = self.data_dir / f"{qty_str}.parquet" 
            pqt_in_path = self.in_folder / "data" / f"{qty_str}.parquet"
            
            if pqt_out_path.exists():
                self.dataframes[qty_str] = pd.read_parquet(pqt_out_path, engine='pyarrow')
                continue
            elif pqt_in_path.exists():
                self.dataframes[qty_str] = pd.read_parquet(pqt_in_path, engine='pyarrow')
                continue

            if qty_str.startswith('d_'):
                self.compute_temporal_derivative(Quantity(qty_str[2:]))
            else:
                self.extract_data(quantity=qty)

    def extract_data(self, quantity: Quantity = Quantity.ANGLE, output_name: str | None = None) -> None:
        files = sorted(f for f in self.in_folder.iterdir() if f.suffix in ('.ovf', '.omf'))
        if not files: raise FileNotFoundError(f"Simulation files missing in directory: {self.in_folder}")

        initial_field = dfield.Field.from_file(files[0])
        region = initial_field.mesh.region
        X, Y = np.meshgrid(
            np.linspace(region.pmin[0], region.pmax[0], initial_field.mesh.n[0]), 
            np.linspace(region.pmin[1], region.pmax[1], initial_field.mesh.n[1]), 
            indexing='ij'
        )
        
        data = {'x (m)': X.flatten(), 'y (m)': Y.flatten()}
        for file in files:
            name, values = _extract_single_ovf(file, quantity)
            data[name] = values

        df = pd.DataFrame(data)
        df = df[['x (m)', 'y (m)'] + [f.name for f in files]]
        
        qty_str = str(quantity)
        out_path = self.data_dir / (output_name or f"{qty_str}.parquet")
        df.to_parquet(out_path, engine='pyarrow', index=False) 
        self.dataframes[qty_str] = df

    def compute_temporal_derivative(self, base_quantity: Quantity) -> None:
        self.check_loaded(base_quantity)
        base_str = str(base_quantity)
        target_qty = f"d_{base_str}"
        
        df = self.dataframes[base_str]
        time_cols = [c for c in df.columns if c not in ['x (m)', 'y (m)']]
        matrix = df[time_cols].values
        
        d_matrix = np.diff(matrix, axis=1)
        if base_quantity == Quantity.ANGLE:
            d_matrix = (d_matrix + np.pi) % (2 * np.pi) - np.pi
        
        d_cols = [f"d_{col}" for col in time_cols[:-1]]
        d_df = pd.concat([df[['x (m)', 'y (m)']].copy(), pd.DataFrame(d_matrix, columns=d_cols)], axis=1)
            
        self.dataframes[target_qty] = d_df
        out_path = self.data_dir / f"{target_qty}.parquet"
        d_df.to_parquet(out_path, engine='pyarrow', index=False)