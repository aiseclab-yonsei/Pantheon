"""Apply the original FF++ median-centered quality normalization."""
import json
import numpy as np
from .pipeline import ROOT

def normalize_quality(crfiqa, qcn):
    params = json.loads((ROOT / 'data/standardized/normalization_params.json').read_text())
    result = []
    for name, values in [('cr_fiqa', crfiqa), ('qcn', qcn)]:
        p = params[name]
        clipped = np.clip(values, p['lower_bound'], p['upper_bound'])
        result.append(np.where(clipped <= p['median'], (clipped - p['min_inlier']) / p['denom_lower'] - 1, (clipped - p['median']) / p['denom_upper']))
    return result
