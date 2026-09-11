## [BUG] NVMe-offloaded optimizer checkpoints silently resume with wrong state (loss diverges after `load_checkpoint`)

### Summary

`tests/unit/runtime/zero/test_nvme_checkpointing.py::TestNVMeCheckpointing::test_nvme_checkpointing` fails in CI (Windows CPU runner, `windows-torch-latest.yml`) for every parametrization where `optim_offload_device == nvme` (`none-nvme`, `cpu-nvme`, `nvme-nvme`), while the ones without NVMe optimizer offload (`nvme-none`, `nvme-cpu`) pass:

```
tests/unit/runtime/zero/test_nvme_checkpointing.py:147: in test_nvme_checkpointing
    assert loss_before == loss_after
AssertionError
```

This is **not** CI flakiness or a hardware/fp16-precision quirk. It's a real correctness regression in `DeepSpeedEngine.load_checkpoint()`: when ZeRO-3's optimizer is NVMe-offloaded, `load_checkpoint()` never restores several pieces of in-memory-only optimizer/engine state, so a process that resumes from a checkpoint produces different training results than the uninterrupted run it was checkpointed from.

### Reproduction

Reproduced locally (Windows, CPU-only, fp16 hardware support and `py-cpuinfo` stubbed out since this dev machine lacks both) by running all 5 parametrizations directly (bypassing pytest's default `-m "not sequential"` filter):

```
unit/runtime/zero/test_nvme_checkpointing.py::TestNVMeCheckpointing::test_nvme_checkpointing[cpu-nvme]  FAILED
unit/runtime/zero/test_nvme_checkpointing.py::TestNVMeCheckpointing::test_nvme_checkpointing[none-nvme] FAILED
unit/runtime/zero/test_nvme_checkpointing.py::TestNVMeCheckpointing::test_nvme_checkpointing[nvme-none] PASSED
unit/runtime/zero/test_nvme_checkpointing.py::TestNVMeCheckpointing::test_nvme_checkpointing[nvme-nvme] FAILED
unit/runtime/zero/test_nvme_checkpointing.py::TestNVMeCheckpointing::test_nvme_checkpointing[nvme-cpu]  PASSED
```

This exactly matches the CI failure pattern — all and only the configs where the **optimizer** (not the parameters) is NVMe-offloaded fail, deterministically.

### Root cause

`DeepSpeedEngine.load_checkpoint()` ([deepspeed/runtime/engine.py:4426-4440](deepspeed/runtime/engine.py#L4426-L4440)):

```python
load_zero_checkpoint = load_path is not None and self.zero_optimization()
if load_zero_checkpoint and not self.zero_nvme_offload_optimizer():
    ...
    success = self._load_zero_checkpoint(load_dir, tag, ...)
    ...

if self.zero_nvme_offload_optimizer():
    from shutil import copytree, disk_usage
    ...
    copytree(offload_ckpt_dir, offload_dir, dirs_exist_ok=True)
    ...
    self.optimizer.reset_swap_buffers()
```

Whenever `zero_nvme_offload_optimizer()` is `True`, `_load_zero_checkpoint()` is **skipped entirely**, and restoration relies solely on `copytree`-ing the raw NVMe swap files back into place.

But `_load_zero_checkpoint()` → `DeepSpeedZeroOptimizer_Stage3.load_state_dict()` → `_rigid_load_state_dict()` ([deepspeed/runtime/zero/stage3.py:3264-3310](deepspeed/runtime/zero/stage3.py#L3264-L3310)) is what restores:

- `self.loss_scaler`, `self.dynamic_loss_scale`, `self.overflow` (fp16 dynamic loss-scaling state), and
- `self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])`, which restores Adam's **`step` counter** and any other small scalar optimizer state.

The Adam `step` counter in particular is never written to the NVMe `.tensor.swp` files — only tensors large enough to be "swappable" get offloaded (see `OptimizerSwapper.is_swappable_tensor`); `step` is a plain Python/tensor scalar that only ever lives in `self.optimizer.state[param]['step']` in memory. Skipping `_load_zero_checkpoint()` means this counter is never restored, so the resumed optimizer's bias-correction terms (`1 - beta1**step`, `1 - beta2**step`) are computed with the wrong step count, producing a numerically different update on the very next `step()` call — hence `loss_before != loss_after`.

### Why this isn't a simple revert

This guard was added in commit `e37c37ac` (PR [#7613](https://github.com/deepspeedai/DeepSpeed/pull/7613), "Fixed save_checkpoint race when consolidating NVMe offloaded tensors", merged 2025-10-01), which bundled two unrelated fixes:

1. **The legitimate fix** for issue [#7549](https://github.com/deepspeedai/DeepSpeed/issues/7549): concurrent ranks racing on `shutil.copytree` (no `dirs_exist_ok`) during `save_checkpoint` → `FileExistsError`. Fixed correctly by giving each rank its own `offloaded_tensors/rank{N}/` subdirectory for both save and load.
2. **A workaround for a different, previously-undiscovered bug**: while investigating #7549, contributor `therealnaveenkamal` found that calling `_load_zero_checkpoint()` under NVMe optimizer offload caused a **segfault** during the backward pass after checkpoint loading — `optimizer.load_state_dict()` assumes the optimizer's fp32/state tensors are live in memory, but under NVMe offload they're empty placeholders (`torch.empty(0)`) until explicitly swapped in. Their fix for that crash was exactly the guard quoted above: *"Skip `_load_zero_checkpoint()` when `zero_nvme_offload_optimizer()` returns true."*

So the guard trades a crash for silent wrong-answer checkpoint resumption. **Naively removing `and not self.zero_nvme_offload_optimizer()` will very likely reintroduce the segfault**, not just fix the loss mismatch.

### What a correct fix likely needs

`_rigid_load_state_dict()`'s restore path needs to be made safe when the fp32/optimizer-state tensors are still unmaterialized NVMe placeholders — e.g. ensuring each swappable subgroup is swapped in (and sized correctly) before `optimizer.load_state_dict()` / the `purge_state()` + "touch all parameters" loop touches its `.data`, while still keeping the rank-scoped `copytree` for the bulk tensor data. This is nontrivial ZeRO-3 + NVMe-swap engineering, and per this repo's own contribution rules, a fix at the checkpoint/training-loop level needs an integration test executed on real GPU + NVMe hardware to be trustworthy — not something a Windows CPU-only dev box can validate.

### Suggested next steps

- File this as a GitHub issue against `deepspeedai/DeepSpeed`, referencing #7549 and PR #7613.
- Any fix attempt should include a repro run of `test_nvme_checkpointing.py` (all 5 parametrizations) on hardware with real NVMe + fp16 support, plus a check that the original segfault (backward pass after `load_checkpoint()` under NVMe optimizer offload) does not return.
