"""Table of the closed-loop runs found in */runs/*.json (written by the run scripts).

    python collect_results.py

Columns: residual wavefront variance [rad^2] in the first policy episode, in the worst one and averaged
over the last `n_last` episodes; Strehl (exp(-variance), piston removed) over the last episodes; median
policy inference time. All runs with the same seed see the same turbulence.
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(n_last=10):
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, '*', 'runs', '*.json')) + glob.glob(os.path.join(HERE, 'runs', '*.json'))):
        name = os.path.basename(path)[:-5]
        with open(path) as f:
            d = json.load(f)
        if 'history' not in d:                       # integrator gain scan: {gain: [variance per episode]}
            for gain, v in d.items():
                rows.append(('integrator gain %s' % gain, np.nan, np.max(v[10:]), np.mean(v[-n_last:]),
                             np.mean(np.exp(-np.array(v[-n_last:]))), np.nan, len(v) - 10))
            continue
        pol = [m for m in d['history'] if m['mode'] != 'warmup']
        v = [m['res_var'] for m in pol]
        lat = [m['latency_ms'] for m in pol if 'latency_ms' in m]
        rows.append((name, v[0], np.max(v), np.mean(v[-n_last:]), np.mean([m['strehl'] for m in pol[-n_last:]]),
                     np.median(lat) if lat else np.nan, len(v)))
    print('%-58s %8s %8s %8s %7s %8s %4s' % ('run', 'first', 'worst', 'last', 'Strehl', 'ms', 'ep.'))
    for r in rows:
        print('%-58s %8.3f %8.3f %8.3f %7.3f %8.2f %4d' % r)


if __name__ == '__main__':
    main()
