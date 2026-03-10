# Known Issues (Auto-Generated)

---

## 1. Thread Safety in Loss Module State Storage

**Description:** The `GigaBrain0Loss` class stores intermediate diffusion values (`self.x_t`, `self.u_t`, `self.time`) as instance variables during `add_noise()`, which are then read during `forward()`. In multi-GPU or multi-threaded training, these values may be overwritten between calls.

**Root Causes:** `giga_brain_0/giga_brain_0_loss.py` — `add_noise()` writes to instance attributes that `forward()` later reads, with no synchronization mechanism.

**Consequences:** Silent gradient corruption in distributed training when batches are processed in parallel across GPUs.

---

## 2. Race Condition in AgileX ROS Deque Access

**Description:** The `RosOperator` class in the AgileX inference client uses shared deques for sensor data that are accessed by multiple threads without locks. ROS subscriber callbacks append to deques while `get_frame()` reads and pops from them concurrently.

**Root Causes:** `scripts/inference_agilex_client.py` — `get_frame()` accesses deques without thread locks; subscriber callbacks modify the same deques from ROS threads. Additionally, `while` loops that call `popleft()` can empty the deque mid-iteration, causing `IndexError`.

**Consequences:** Sporadic crashes during real-time robot control when frame synchronization discards all buffered samples, or data corruption from concurrent access.

---

## 3. No Validation Between Norm Stats and Embodiment ID

**Description:** All inference scripts accept both a `norm_stats_path` and an `embodiment_id` as separate parameters, but nothing validates that the norm stats file corresponds to the specified embodiment. Using mismatched normalization statistics produces silently incorrect actions.

**Root Causes:** All inference scripts (`scripts/inference.py`, `scripts/inference_discrete_action.py`, etc.) and the transform pipeline (`giga_brain_0/giga_brain_0_transforms.py`) — no cross-validation between these two parameters.

**Consequences:** Silently nonsensical action predictions that are difficult to diagnose. Robot may execute dangerous movements with wrong normalization.

---

## 4. Multiple Diverging Inference Implementations

**Description:** Five separate inference scripts each implement their own parameter handling, model loading, observation construction, and inference logic with significant code duplication. Changes to core inference logic must be replicated across all scripts.

**Root Causes:** `scripts/inference.py`, `scripts/inference_discrete_action.py`, `scripts/inference_task_planning.py`, `scripts/inference_server.py`, `scripts/inference_agilex_client.py` — no shared inference wrapper or base class.

**Consequences:** Bugs fixed in one script are not propagated to others. Behavior divergence across inference modes. High maintenance burden when inference pipeline changes.

---

## 5. Hardcoded Dimensions and Joint Limits Throughout Codebase

**Description:** Robot-specific constants (14 joints, 480×640 image size, joint limit values, state dimension of 14) are hardcoded across multiple files instead of being derived from configuration.

**Root Causes:** `scripts/inference_agilex_client.py` (joint limits, state dim=14, image dims 480×640), `scripts/inference.py` (action slicing `[:, :14]`), `scripts/convert_from_hdf5.py` (14 motors, 3 cameras hardcoded).

**Consequences:** Code only works correctly for the AgileX Cobot Magic robot. Other embodiments with different DOF counts or image resolutions will fail or produce silently incorrect results.

---

## 6. Unsafe String Parsing in Task Planning Inference

**Description:** The task planning inference script splits task strings using `' subtask: '` as a delimiter and accesses the second element without checking if the split produced enough parts.

**Root Causes:** `scripts/inference_task_planning.py` — `task.lower().strip().split(' subtask: ')` followed by `pairs[1]` with no length check.

**Consequences:** `IndexError` crash on any task string that doesn't contain the expected `' subtask: '` substring.

---

## 7. Unguarded Dataset Index Access in Inference Scripts

**Description:** Multiple inference scripts iterate over hardcoded indices (0, 1000, 2000, ..., 10000) without checking dataset size, causing crashes if the dataset is smaller than expected.

**Root Causes:** `scripts/inference_task_planning.py`, `scripts/inference_discrete_action.py`, `scripts/inference.py` — `for idx in range(0, 10000, 1000)` without bounds checking against `len(dataset)`.

**Consequences:** `IndexError` crashes when running inference on small datasets. Scripts silently skip indices in some cases (discrete_action) but crash in others (task_planning).

---

## 8. No Data Validation in HDF5 Conversion Pipeline

**Description:** The HDF5-to-LeRobot conversion script processes episodes without validating that required keys exist, images are not corrupted, temporal alignment is correct, or dimensions match expectations.

**Root Causes:** `scripts/convert_from_hdf5.py` — `load_raw_episode_data()` and `populate_dataset()` have minimal validation. `get_cameras()` crashes on empty file lists. Partial episode failures leave dataset in inconsistent state with no rollback.

**Consequences:** Silent data corruption flowing into training. Partially populated datasets on episode conversion failures.

---

## 9. Embodiment Configuration Scattered Across Codebase

**Description:** Embodiment-specific logic (delta action masks, state dimensions, robot type mappings) is distributed across 6+ files with no centralized registry. Adding a new embodiment requires coordinated changes across transforms, configs, inference scripts, and conversion scripts.

**Root Causes:** `giga_brain_0/giga_brain_0_transforms.py` (EmbodimentId enum, delta masks), all config files (per-embodiment delta masks), all inference scripts (embodiment_id parameter). No central embodiment definition file.

**Consequences:** High risk of inconsistency when adding new embodiments. Easy to update one file and forget others.

---

## 10. Loss Masks Can Silently Become All-Zero

**Description:** The transform pipeline creates masks for action loss, language loss, and trajectory loss. If all mask values are zero (e.g., trajectory data consistently missing, or all language tokens masked), the corresponding loss term becomes zero with no warning.

**Root Causes:** `giga_brain_0/giga_brain_0_transforms.py` (mask creation), `giga_brain_0/giga_brain_0_loss.py` (masked loss computation with `clamp(min=1)` division).

**Consequences:** Training degenerates — model learns nothing from masked-out objectives, but no signal is raised. Difficult to diagnose why certain capabilities (e.g., trajectory prediction) don't improve.

---

## 11. Quantile Computation Precision Issues in Norm Stats

**Description:** The `RunningStats` class uses histogram-based streaming quantile computation that redistributes histogram counts when min/max range expands. The redistribution approach (`np.histogram(old_edges[:-1], ...)`) may not correctly preserve statistics.

**Root Causes:** `scripts/compute_norm_stats.py` — `_adjust_histograms()` uses edge midpoints as proxy data for redistribution. Small epsilon (1e-10) for bin edges is negligible for large data ranges.

**Consequences:** Inaccurate q01/q99 quantile estimates, leading to suboptimal normalization bounds. This affects action scaling to [-1, 1] range.

---

## 12. ZMQ Socket Management Lacks Retry and Cleanup Logic

**Description:** The inference client-server architecture has no retry logic for failed connections, no heartbeat mechanism, and insufficient cleanup in destructors.

**Root Causes:** `scripts/inference_client.py` — `BaseInferenceClient.__del__()` doesn't check if socket/context were successfully created. `scripts/inference_server.py` — no reconnection or keep-alive protocol.

**Consequences:** Network glitches terminate the inference pipeline. Resource leaks from partially initialized sockets. No way to monitor server health from client side.

---

## 13. Potential Type Confusion in Trainer Config Access

**Description:** The trainer's `get_models()` method uses `hasattr(model_config, 'pretrained')` to check for configuration keys, but `model_config` may be a dict (where `hasattr` behaves differently than intended).

**Root Causes:** `giga_brain_0/giga_brain_0_trainer.py` — uses `hasattr()` instead of `'key' in dict` for dict-type config objects.

**Consequences:** Pretrained model loading may be silently skipped if config is a dict, since `hasattr(dict_obj, 'pretrained')` returns False even when the key exists. Model would train from random initialization instead of from pretrained weights.

---

## 14. No Logging or Observability in Inference Pipelines

**Description:** None of the inference scripts include debug logging for critical information such as which embodiment is being used, which norm stats were loaded, model checkpoint identity, inference timing, or tensor shape validation.

**Root Causes:** All inference scripts — no `logging` module usage, no shape assertions, no performance metrics collection.

**Consequences:** Debugging production inference issues becomes nearly impossible. Silent failures are undetectable.
