#!/usr/bin/env python3
"""Build the full core PHC bundle per DATASET.md §4/§4a.

Enumerates every core clip (ACCAD Male2 martial-arts sets + locomotion +
general, SSM strikes, CMU strikes, defense_recovery patch), retargets each
through export_phc.export_clip (GMR + arm-extension bias) into
combatbench_core_v1.pkl, then updates manifest.json.

Usage (repo root):
  mocap/gmr/.venv/bin/python mocap/humanoid21/gmr/build_core.py [--only SUBSTR] [--bundle core]
"""
import argparse, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_phc as E

SMPLH = E.AMASS_ROOTS[1]                     # smplx_from_smplh
NATIVE = E.AMASS_ROOTS[0]                    # smplx_gender_specific
ACCAD = SMPLH / 'ACCAD'

SKIP = re.compile(r'calibrat|subject cal|blanks', re.I)


def slug(stem):
    for suf in ('_poses', '_stageii', '_sync'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')


def accad_dir(dirname, prefix, default_label, label_rules=None):
    out = []
    for p in sorted((ACCAD / dirname).glob('*.npz')):
        if SKIP.search(p.stem):
            continue
        s = slug(p.stem)
        label = default_label
        for pat, lab in (label_rules or []):
            if re.search(pat, s):
                label = lab
                break
        out.append((f'ACCAD/{dirname}/{p.name}', prefix + s, [label], 'train'))
    return out


def clip_list():
    c = []
    c += accad_dir('Male2MartialArtsPunches_c3d', 'accad_m2ma_punches_', 'strike_punch')
    c += accad_dir('Male2MartialArtsKicks_c3d', 'accad_m2ma_kicks_', 'strike_kick')
    c += accad_dir('Male2MartialArtsStances_c3d', 'accad_m2ma_stances_', 'stance_guard')
    c += accad_dir('Male2MartialArtsExtended_c3d', 'accad_m2ma_extended_', 'stance_guard')
    c += accad_dir('MartialArtsWalksTurns_c3d', 'accad_mawt_', 'footwork',
                   label_rules=[(r'block', 'block'), (r'dodge|duck', 'dodge')])
    c += accad_dir('Male2Walking_c3d', 'accad_m2w_', 'walk')
    c += accad_dir('Male2Running_c3d', 'accad_m2r_', 'run')
    c += accad_dir('Male2General_c3d', 'accad_m2g_', 'general_misc')

    # SSM strike clips (native smplx tree)
    c += [
        ('SSM/20160330_03333/punching_stageii.npz', 'ssm_03333_punching', ['strike_punch'], 'train'),
        ('SSM/20160930_50032/punch_kick_sync_stageii.npz', 'ssm_50032_punch_kick', ['strike_punch', 'strike_kick'], 'train'),
        ('SSM/20161014_50033/punch_kick_sync_stageii.npz', 'ssm_50033_punch_kick', ['strike_punch', 'strike_kick'], 'train'),
        ('SSM/20160930_50032/dribble_kick_sync_stageii.npz', 'ssm_50032_dribble_kick', ['strike_kick'], 'train'),
        ('SSM/20161014_50033/dribble_kick_sync_stageii.npz', 'ssm_50033_dribble_kick', ['strike_kick'], 'train'),
        ('SSM/20160930_50032/special_move_sync_stageii.npz', 'ssm_50032_special_move', ['strike_kick'], 'train'),
    ]

    # CMU strikes
    c += [
        ('CMU/13/13_17_poses.npz', 'cmu_13_17', ['strike_punch'], 'train'),
        ('CMU/13/13_18_poses.npz', 'cmu_13_18', ['strike_punch'], 'val'),
        ('CMU/144/144_13_poses.npz', 'cmu_144_13', ['strike_punch'], 'train'),
        ('CMU/144/144_14_poses.npz', 'cmu_144_14', ['strike_punch'], 'train'),
        ('CMU/144/144_20_poses.npz', 'cmu_144_20', ['strike_punch'], 'train'),
        ('CMU/144/144_21_poses.npz', 'cmu_144_21', ['strike_punch'], 'train'),
        ('CMU/135/135_09_poses.npz', 'cmu_135_09', ['strike_punch'], 'train'),
        ('CMU/111/111_19_poses.npz', 'cmu_111_19', ['strike_punch', 'strike_kick'], 'train'),
    ]

    # defense_recovery patch (DATASET.md §4a)
    c += [
        ('CMU/144/144_07_poses.npz', 'cmu_144_07', ['block'], 'train'),
        ('CMU/144/144_26_poses.npz', 'cmu_144_26', ['block'], 'train'),
        ('CMU/135/135_05_poses.npz', 'cmu_135_05', ['block'], 'train'),
        ('CMU/135/135_10_poses.npz', 'cmu_135_10', ['block'], 'train'),
        ('CMU/77/77_09_poses.npz', 'cmu_77_09', ['dodge'], 'train'),
        ('CMU/139/139_09_poses.npz', 'cmu_139_09', ['dodge'], 'train'),
        ('CMU/76/76_03_poses.npz', 'cmu_76_03', ['dodge'], 'train'),
        ('CMU/127/127_29_poses.npz', 'cmu_127_29', ['dodge', 'has_ground_contact'], 'train'),
        ('CMU/128/128_04_poses.npz', 'cmu_128_04', ['dodge', 'has_ground_contact'], 'train'),
        ('CMU/140/140_01_poses.npz', 'cmu_140_01', ['getup', 'has_ground_contact', 'recovery'], 'train'),
        ('CMU/140/140_04_poses.npz', 'cmu_140_04', ['getup', 'has_ground_contact', 'recovery'], 'train'),
        ('CMU/140/140_08_poses.npz', 'cmu_140_08', ['getup', 'has_ground_contact', 'recovery'], 'train'),
        ('CMU/139/139_18_poses.npz', 'cmu_139_18', ['getup', 'has_ground_contact', 'recovery'], 'train'),
        ('CMU/111/111_06_poses.npz', 'cmu_111_06', ['getup', 'has_ground_contact', 'recovery'], 'train'),
        ('CMU/113/113_08_poses.npz', 'cmu_113_08', ['getup', 'has_ground_contact', 'recovery'], 'train'),
        ('CMU/90/90_16_poses.npz', 'cmu_90_16', ['fall', 'has_ground_contact'], 'train'),
        ('CMU/90/90_18_poses.npz', 'cmu_90_18', ['fall', 'has_ground_contact'], 'train'),
    ]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='core')
    ap.add_argument('--only', default=None, help='substring filter on clip_id')
    ap.add_argument('--no-arm-bias', action='store_true')
    args = ap.parse_args()

    clips = clip_list()
    ids = [c[1] for c in clips]
    assert len(ids) == len(set(ids)), 'duplicate clip_ids'
    cache = E.load_model()
    fails = []
    for i, (npz, cid, labels, split) in enumerate(clips, 1):
        if args.only and args.only not in cid:
            continue
        try:
            E.export_clip(npz, cid, labels, split=split, dataset='core',
                          bundle=args.bundle, arm_bias=not args.no_arm_bias,
                          model_cache=cache)
        except Exception as e:
            fails.append((cid, str(e)))
            print(f'[FAIL] {cid}: {e}', flush=True)
    print(f'DONE: {len(clips) - len(fails)}/{len(clips)} exported, '
          f'{len(fails)} failures', flush=True)
    for cid, err in fails:
        print(f'  FAIL {cid}: {err}')


if __name__ == '__main__':
    main()
