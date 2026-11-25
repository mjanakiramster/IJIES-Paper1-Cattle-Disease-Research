"""Exports the development out-of-fold predictions (dev_predictions_v2.pkl) to a portable long-format CSV.
One row per household x split x disease x model. Run from the project root: python src/paper1/export_predictions_v1.py"""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from paper1.core import setup, DIS
df, Y, _, _ = setup()
P = pd.read_pickle('results/paper1_v1/predictions/dev_predictions_v2.pkl'); out = []
for _, r in P.iterrows():
    i = np.asarray(r.test_idx); d = DIS.index(r.disease)
    out.append(pd.DataFrame({'rep': r.rep, 'fold': r.fold, 'disease': r.disease, 'model': r.model, 'household_ID': df.ID.values[i], 'location_id': df['loc'].values[i],
                             'survey_year': df.Year.values[i], 'observed': Y[i, d], 'score': r.score, 'calibrated_probability': r.prob, 'training_quantile_score': r.qscore,
                             'threshold': r.threshold, 'predicted_class': (np.asarray(r.prob) >= r.threshold).astype(int)}))
O = pd.concat(out, ignore_index=True); O.to_csv('results/paper1_v1/predictions/dev_oof_predictions_long_v1.csv', index=False, float_format='%.6f')
print(O.shape, O.groupby(['rep', 'disease', 'model']).size().unique())
