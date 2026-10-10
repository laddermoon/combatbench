# SAC Artifact Registry

> 类型：记录

All persisted/consumed artifacts in the SAC framework, their schema
versions, producers, consumers, and fail-loud rejection behavior.
Update this table whenever a schema is added or bumped.

## Versioned artifacts

| Artifact | Location | Schema marker | Producer | Consumer | Unknown-schema behavior |
|---|---|---|---|---|---|
| Checkpoint bundle | `runs/<run>/checkpoints/checkpoint_sXXXXXXXX/` | `sac_checkpoint_v1` (`manifest.json:schema`) | `checkpoint.save_checkpoint_bundle` | `checkpoint.load_checkpoint_bundle` | `SACCheckpointError`; per-file sha256 + size verified on load |
| Trainer state | inside checkpoint: `trainer.pt` | `sac_trainer_v1` (`payload["schema"]`) | `trainer.trainer_state_dict` | `load_trainer_state` / `load_model_state` | `SACTrainerError` |
| Replay buffer | inside checkpoint: `replay.pt` | `sac_replay_v2` (`state["schema"]`) | `SACReplayBuffer.save` | `SACReplayBuffer.load` | `SACReplayError` |
| Metrics event stream | `runs/<run>/metrics/events.jsonl` | `sac_metrics_v1` (`schema_version` per event) | `SACMetricsWriter` | `metrics.load_events` | `SACMetricsError`-family rejection |
| Debug dump | `runs/<run>/debug_dumps/critic_tick_NNNNNN/` | `sac_dump_v3` (`manifest.json:schema_version`; `sac_dump_v2` readable) | `debugkit` dump capture | `debugkit.load_dump` / `analysis` / `debugserver` | unsupported schema → error listing `SUPPORTED_DUMP_SCHEMAS` |
| Transition slice | in-memory (collection → replay) | `sac_transition_v2` (`slice.schema_version`) | `experiment.build_slices` | `transition.validate_transition_slice` (called by `replay.add_slices`) | `ValueError` naming the unsupported schema |
| Policy export blueprint | `runs/<run>/policy/`, `policy_exports/` | `PolicyBlueprint` YAML + TN payload `tn_kernel_v1` (`kernel_version`) | `actor.to_blueprint` | `PolicyBlueprint.load` → `TNActor.from_blueprint` | kernel_version mismatch → error naming the kernel |
| Run config snapshot | `runs/<run>/config.json` | `sac_run_config_v1` (`config_schema`); pre-marker configs are implicitly unversioned | `loop.save_run_config_sac` | fingerprint compare on resume | fingerprint mismatch → `SACCheckpointError`; `config_schema` itself is excluded from the fingerprint via `RESUME_ALLOWED_OVERRIDES` so pre-marker checkpoints stay resumable |

## Unversioned / internal artifacts

| Artifact | Location | Notes |
|---|---|---|
| Runtime state | inside checkpoint: `runtime.pt` | clocks / utd_credit / n_evals_done / RNG state; internal to the loop, not a public contract |
| Experiment state | inside checkpoint: `experiment.json` | schema owned by each experiment (`state()`/`load_state`) |
| Collection episode | in-memory (rollouter → build_slices) | carries `sac_collection_v1` contract marker; in-memory only, marker is documentary not enforced |
| Eval videos | `runs/<run>/videos/s<env_step>.mp4` | filename encodes env_step; served by `debugserver /videos/` |
| Dump request sentinel | `runs/<run>/dump_request.json` | `{critic_tick, hypothesis}`; consumed and deleted by the loop |

## Conventions

- Schema bumps rename the constant (e.g. `sac_dump_v3`); readers may
  accept a whitelist of older schemas (see `SUPPORTED_DUMP_SCHEMAS`)
  but writers always emit the newest.
- Atomicity: checkpoint bundles write to a `checkpoint_s*.XXXX` tmp dir
  then rename; readers must reject a bundle without `manifest.json`.
- Retention: `train_sac(checkpoint_keep_last=N)` prunes old bundles
  (`.pinned` marker exempts); dumps prune via `dump_keep_last`.
