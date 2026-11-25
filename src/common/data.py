"""Data loading shared by every stage. Only eligible predictors are exposed to models."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

DIS = ['CBPP', 'FMD', 'LSD']
ENV = ['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed']
EXCLUDED = ['Weight', 'Year_region', 'Year', 'Region']          # never predictors (leakage / sampling design)
AUDITED_MD5 = {'cbpp.xlsx': 'b1392ae82276e4ed4fc8284e9d2f4c5c',   # from stage1_v1 audit (stage1_results_v1.json)
               'fmd.xlsx': 'bb1b875cedd96732cd5116dfb8fdb858',
               'lsd.xlsx': '1102a0ed7affb13a27742594445fe82a'}
BLOCK_KM = 80.94049315824638   # median variogram practical range (run 1, stage3_variogram_v1.csv)
SEED = 20261119


def file_hashes(path):
    b = Path(path).read_bytes()
    return {'md5': hashlib.md5(b).hexdigest(), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}


def load(raw_dir='data/raw'):
    """Merged table: one row per household, the 3 disease labels, coordinates in km, and run-1 fold columns."""
    raw_dir = Path(raw_dir)
    raw = {d: pd.read_excel(raw_dir / f'{d.lower()}.xlsx', sheet_name=d.lower()).rename(columns={'wind': 'Wind_speed'})
           .sort_values('id').reset_index(drop=True) for d in DIS}
    shared = ['id'] + ENV + ['Year_region', 'x', 'y']
    for d in DIS[1:]:
        assert raw[d][shared].equals(raw['CBPP'][shared]), f'shared columns differ in {d}'
    df = raw['CBPP'][shared].copy()
    for d in DIS:
        df[d] = raw[d][d.lower()].astype(int).values
        df[f'Weight_{d}'] = raw[d]['Weight'].values           # kept only for the leakage check
    df = df.rename(columns={'id': 'ID', 'x': 'Longitude', 'y': 'Latitude'})
    df[['Year', 'Region']] = df.Year_region.str.split('.', n=1, expand=True)
    df['Year'] = df.Year.astype(int)
    folds = pd.read_csv(raw_dir / 'stage3_folds_v1.csv')
    df = df.merge(folds.drop(columns=['Region', 'Year']), on='ID', how='left', validate='one_to_one')
    assert len(df) == 3340 and df.Location_id.notna().all()
    df['loc'] = df.Location_id.astype(int) - 1
    lat0 = np.radians(df.Latitude.mean())                       # same projection as run 1
    df['x_km'] = df.Longitude * 111.32 * np.cos(lat0)
    df['y_km'] = df.Latitude * 110.57
    return df


def labels(df):
    return df[DIS].values.astype(int)
