"""Build the humanoid28 core PHC bundle.

Same 172-clip manifest as humanoid21 (single source of truth:
mocap/humanoid21/gmr/build_core.py:clip_list), retargeted onto the
MimicKit humanoid morphology.

Run:
  mocap/gmr/.venv/bin/python mocap/humanoid28/gmr/build_core.py [--only SUBSTR] [--bundle core]
"""
import argparse, importlib.util, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# NOTE: h21's build_core.py does a bare `import export_phc` internally —
# its gmr dir must be on sys.path at exec time, and it lands in
# sys.modules['export_phc']. We therefore load BOTH files by explicit
# path and never `import export_phc` ourselves (it would bind h21's).
sys.path.insert(0, str(HERE.parents[1] / 'humanoid21' / 'gmr'))
_h21_bc = _load_module(
    'h21_build_core',
    HERE.parents[1] / 'humanoid21' / 'gmr' / 'build_core.py')
clip_list = _h21_bc.clip_list
sys.path.pop(0)

E = _load_module('h28_export_phc', HERE / 'export_phc.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='core')
    ap.add_argument('--only', default=None, help='substring filter on clip_id')
    ap.add_argument('--no-arm-bias', action='store_true')
    args = ap.parse_args()

    clips = _h21_bc.clip_list()
    ids = [c[1] for c in clips]
    assert len(ids) == len(set(ids)), 'duplicate clip_ids'
    cache = E.load_model()
    fails, done = [], 0
    for i, (npz, cid, labels, split) in enumerate(clips, 1):
        if args.only and args.only not in cid:
            continue
        try:
            E.export_clip(npz, cid, labels, split=split, dataset='core',
                          bundle=args.bundle, arm_bias=not args.no_arm_bias,
                          model_cache=cache)
            done += 1
        except Exception as e:
            fails.append((cid, str(e)))
            print(f'[FAIL] {cid}: {e}', flush=True)
    print(f'DONE: {done} exported, {len(fails)} failures', flush=True)
    for cid, err in fails:
        print(f'  FAIL {cid}: {err}')


if __name__ == '__main__':
    main()
