"""Feature construction. Neighbourhood features use labels of FIT rows only and always exclude the
query location's own spatial block, for fit rows and prediction rows alike."""
import numpy as np
import pandas as pd
from .data import ENV

CLIMATE_NAMES = ENV + ['SL_temperature', 'DeMartonne', 'Temp_x_RH', 'Precip_x_RH', 'log_Precip', 'Wind_x_RH']
COORD_NAMES = ['Lat', 'Lon', 'LatPlusLon', 'LatMinusLon']


def climate_coords(df):
    """Label-free features: 5 original variables, 6 transforms/interactions, 4 coordinate terms."""
    c = pd.DataFrame({
        **{v: df[v] for v in ENV},
        'SL_temperature': df.Temperature + 0.0065 * df.Altitude,
        'DeMartonne': df.Precipitation / (df.Temperature + 10),
        'Temp_x_RH': df.Temperature * df.Relative_humidity / 100,
        'Precip_x_RH': np.log1p(df.Precipitation) * df.Relative_humidity / 100,
        'log_Precip': np.log1p(df.Precipitation),
        'Wind_x_RH': df.Wind_speed * df.Relative_humidity / 100,
        'Lat': df.Latitude, 'Lon': df.Longitude, 'LatPlusLon': df.Latitude + df.Longitude, 'LatMinusLon': df.Latitude - df.Longitude})
    return c[CLIMATE_NAMES + COORD_NAMES]


class Neighbourhood:
    """Pre-computes location distances; builds disease-rate features for a given fit set and block labelling."""

    def __init__(self, df, Y):
        self.loc = df['loc'].values
        L = df.groupby('loc')[['Latitude', 'Longitude']].first().sort_index()
        self.NL = len(L)
        la, lo = np.radians(L.Latitude.values), np.radians(L.Longitude.values)
        h = np.sin((la[None] - la[:, None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[None] - lo[:, None]) / 2) ** 2
        self.D = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        self.Y = Y

    def block_of_loc(self, row_blocks):
        s = pd.Series(np.asarray(row_blocks).astype(str)).groupby(self.loc).first().reindex(range(self.NL))
        return s.to_numpy(dtype=str)

    def rates(self, fit_idx, pred_idx, row_blocks, bandwidths=(50, 150, 400), ks=(5, 15, 40), prior=1.0,
              idw_power=None, density=True, diseases=(0, 1, 2)):
        """Return (F_fit, F_pred, names). Rates are (sum_j w_j y_j + prior * p0) / (sum_j w_j n_j + prior), where j runs over fit
        locations outside the query location's block and p0 is the fit-set rate outside that block."""
        bl = self.block_of_loc(row_blocks); same = bl[:, None] == bl[None, :]
        n = np.bincount(self.loc[fit_idx], minlength=self.NL).astype(float)
        pos = np.column_stack([np.bincount(self.loc[fit_idx], weights=self.Y[fit_idx, k], minlength=self.NL) for k in range(3)])
        # prior rate per location = fit-set rate EXCLUDING the location's own block (so own-block labels never enter)
        n_own = same.astype(float) @ n; pos_own = same.astype(float) @ pos
        p0 = (pos.sum(0)[None, :] - pos_own) / np.maximum(n.sum() - n_own, 1.0)[:, None]          # shape (NL, 3)
        avail = (~same) & (n[None, :] > 0)
        cols, names = [], []
        for b in bandwidths:
            K = np.where(avail, np.exp(-self.D ** 2 / (2.0 * b * b)), 0.0)
            r = (K @ pos + prior * p0) / ((K @ n)[:, None] + prior)
            cols += [r[:, k] for k in diseases]; names += [f'kern{int(b)}_{k}' for k in diseases]
            if density: cols.append(np.log1p(K @ n)); names.append(f'dens{int(b)}')
        if len(ks):
            Dm = np.where(avail, self.D, np.inf); order = np.argsort(Dm, axis=1, kind='stable')
            nav = np.isfinite(Dm).sum(1); rr = np.arange(self.NL)
            cn = np.cumsum(n[order], axis=1); cp = np.cumsum(pos[order], axis=1); Ds = np.take_along_axis(Dm, order, axis=1)
            for k_ in ks:
                j = np.clip(np.minimum(int(k_), nav) - 1, 0, self.NL - 1)
                r = np.where((nav > 0)[:, None], (cp[rr, j] + prior * p0) / (cn[rr, j][:, None] + prior), p0)
                cols += [r[:, k] for k in diseases]; names += [f'knn{int(k_)}_{k}' for k in diseases]
                if density: cols.append(np.log1p(np.where(nav > 0, Ds[rr, j], self.D.max()))); names.append(f'knndist{int(k_)}')
        if idw_power:
            W = np.where(avail & (self.D > 0), 1.0 / np.maximum(self.D, 1.0) ** idw_power, 0.0)
            r = (W @ pos + prior * p0) / ((W @ n)[:, None] + prior)
            cols += [r[:, k] for k in diseases]; names += [f'idw_{k}' for k in diseases]
        F = np.column_stack(cols)
        return F[self.loc[fit_idx]], F[self.loc[pred_idx]], names
