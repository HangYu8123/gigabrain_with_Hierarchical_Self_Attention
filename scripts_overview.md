# Scripts & Files Overview

## Root Level Files

### `setup.cfg`
**Summary:** Configuration file for project build tools including isort, black, flake8, and codespell settings for code formatting and linting with line length set to 150 characters.
**Dependencies:** N/A (configuration file)

---

### `README.md`
**Summary:** Main project documentation for GigaBrain-0, a Vision-Language-Action (VLA) foundation model powered by world model-generated data for robotics. Includes installation instructions, quick start guide, and performance metrics for multiple robot embodiments.
**Dependencies:** N/A (documentation file)

---

### `.gitignore`
**Summary:** Git ignore patterns for Python projects excluding pycache, egg files, virtual environments, jupyter notebooks, and PyTorch checkpoint files.
**Dependencies:** N/A (configuration file)

---

### `.pre-commit-config.yaml`
**Summary:** Pre-commit hooks configuration for code quality checks including isort, black, flake8, trailing whitespace fixes, and markdown formatting.
**Dependencies:** N/A (configuration file)

---

### `.dockerignore`
**Summary:** Docker ignore patterns excluding git directories, egg-info, and cache files from Docker builds.
**Dependencies:** N/A (configuration file)

---

### `.gitattributes`
**Summary:** Git configuration file enabling automatic LF normalization for text files across platforms.
**Dependencies:** N/A (configuration file)

---

### `LICENSE`
**Summary:** Apache License 2.0 legal text granting permission to use, reproduce, and distribute the software with specific conditions.
**Dependencies:** N/A (license file)

---

## docs/ Directory

### `docs/configure_introduction.md`
**Summary:** Comprehensive configuration documentation detailing data configuration, distributed training setup, data processing, model configuration, and training parameters for the GigaBrain-0 training pipeline including support for multiple robot embodiments, depth images, and various supervision modes.
**Dependencies:** N/A (documentation file)

---

## configs/ Directory

### `configs/giga_brain_0_from_scratch.py`
**Summary:** Training configuration for pre-training GigaBrain-0 model from scratch using FSDP distributed training, supporting multi-embodiment training (AgileX, AgiBot G1, AgiBot World) with mixed supervision tasks.
**Dependencies:** External configs reference, imports from training framework

**Configuration Parameters:**
- DataLoader configuration with multiple embodiment support
- FSDP distributed training with 8 GPUs
- Batch size: 32 per GPU, 16 workers
- Mixed task supervision (subtask regression 20%, FAST action regression 40%, subtask+FAST 40%)
- Image augmentation enabled, 224×224 image size
- EMA enabled, 250k max steps, activation checkpointing

---

### `configs/giga_brain_0_agilex_finetune.py`
**Summary:** Fine-tuning configuration for AgileX robot embodiment (embodiment_id=0) starting from pretrained GigaBrain-0-3.5B-Base checkpoint with task-only supervision.
**Dependencies:** External configs reference

**Configuration Parameters:**
- Single embodiment (AgileX) with 14-DOF delta action mask
- 50k max training steps, 30k decay steps
- Pretrained model: open-gigaai/GigaBrain-0-3.5B-Base
- Task-only supervision (no action/subtask autoregression during fine-tuning)
- FSDP distributed setup with 8 GPUs

---

### `configs/giga_brain_0_agibot_finetune.py`
**Summary:** Fine-tuning configuration for Agibot G1 robot embodiment (embodiment_id=1) starting from pretrained GigaBrain-0-3.5B-Base with task-only supervision adapted for 20-DOF action space.
**Dependencies:** External configs reference

**Configuration Parameters:**
- Single embodiment (Agibot G1) with 20-DOF delta action mask
- 50k max training steps, 30k decay steps
- Same pretrained base as AgileX variant
- FSDP distributed training with 8 GPUs

---

## giga_brain_0/ Directory

### `giga_brain_0/__init__.py`
**Summary:** Module initialization file exposing GigaBrain0Trainer and GigaBrain0Transform classes.
**Dependencies:** Internal imports from `giga_brain_0_trainer`, `giga_brain_0_transforms`

#### Modules:
- Exports: `GigaBrain0Trainer`, `GigaBrain0Transform`

---

### `giga_brain_0/giga_brain_0_loss.py`
**Summary:** Loss module implementing multi-task loss function combining diffusion loss for action prediction, cross-entropy loss for language tokens, and MSE loss for 2D trajectory supervision.
**Dependencies:** `torch`, `torch.nn`, `torch.nn.functional`

#### Modules:
- `GigaBrain0Loss()`: Main loss module class for combined multi-task losses
  - `sample_noise(shape: tuple, device: torch.device) -> torch.Tensor`: Generates Gaussian noise for diffusion process.
  - `_sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor`: Samples from Beta distribution using Gamma variables for time-step sampling.
  - `sample_time(bsize: int, device: torch.device) -> torch.Tensor`: Samples timesteps (0.001–0.999) for diffusion process using Beta distribution.
  - `add_noise(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`: Adds noise to actions and stores intermediate values (x_t, u_t, time) for loss computation.
  - `llm_loss(logits: torch.Tensor, gt_lang_tokens: torch.Tensor, lang_loss_masks: torch.Tensor) -> torch.Tensor`: Computes cross-entropy loss for language model predictions with masking.
  - `traj_loss(traj_pred: torch.Tensor, gt_traj: torch.Tensor, traj_loss_mask: torch.Tensor) -> torch.Tensor`: Computes MSE loss for 2D trajectory predictions with masking.
  - `forward(model_pred: dict, gt_lang_tokens: torch.Tensor, lang_loss_masks: torch.Tensor, action_loss_mask: torch.Tensor, traj: torch.Tensor | None, traj_loss_mask: torch.Tensor | None, alpha: float) -> dict[str, torch.Tensor]`: Computes total loss combining diffusion, language, and trajectory losses.

---

### `giga_brain_0/giga_brain_0_trainer.py`
**Summary:** Trainer class extending GigaTrain Trainer for GigaBrain0Policy model training, handling model initialization with optional pretrained checkpoints and performing forward passes with loss computation.
**Dependencies:** `torch`, `giga_models.GigaBrain0Policy`, `giga_train.Trainer`, `giga_brain_0_loss.GigaBrain0Loss`

#### Modules:
- `GigaBrain0Trainer(Trainer)`: Custom trainer for GigaBrain0 model.
  - `get_models(model_config: dict[str, Any]) -> GigaBrain0Policy`: Initializes GigaBrain0Policy from pretrained checkpoint or scratch, handles vision patch embedding resizing, and initializes loss function.
  - `forward_step(batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`: Performs forward pass with noise injection, model inference, and loss computation for training step.
- `process_model(model: GigaBrain0Policy, model_config: dict[str, Any]) -> GigaBrain0Policy`: Processes pre-trained model with updated configuration by resizing patch embeddings and reloading weights.
- `_resize_patch_embedding_weight(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor`: Resizes patch embedding weights to match target input channels (handles both expansion and truncation).

---

### `giga_brain_0/giga_brain_0_transforms.py`
**Summary:** Data transformation pipeline for GigaBrain0 training converting raw LeRobot dataset formats to model-ready tensors with normalization, image processing, prompt tokenization, trajectory encoding, and embodiment-specific handling.
**Dependencies:** `json`, `torch`, `giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils` (DeltaActions, ImageTransform, Normalize, PadStatesAndActions, PromptTokenizerTransform, TrajectoryTransform), `giga_train.TRANSFORMS`

#### Modules:
- `EmbodimentId(IntEnum)`: Enumeration for robot embodiment IDs (AGILEX=0, AGIBOT_G1=1, AGIBOT_WORLD=2).
- `RobotType(StrEnum)`: Enumeration for robot types (AGILEX_COBOT_MAGIC, AGIBOT_G1, AGIBOT_WORLD).
- `robot_type_mapping: dict[RobotType, EmbodimentId]`: Maps robot types to embodiment IDs.
- `GigaBrain0Transform`: Main data transformation pipeline.
  - `__init__(delta_action_cfg, norm_cfg, traj_cfg, image_cfg, prompt_cfg, is_train)`: Initializes all transformation components including delta actions, normalization, trajectory, image, and prompt processing with embodiment-aware configurations.
  - `__call__(data_dict: dict[str, Any]) -> dict[str, Any]`: Applies full transformation pipeline to data including embodiment identification, delta action conversion, state/action normalization, prompt tokenization, image transformation, trajectory encoding, and loss mask creation.

---

## scripts/ Directory

### `scripts/train.py`
**Summary:** Entry point script for model training that sets up the environment and launches training from a configuration file using tyro CLI.
**Dependencies:** `tyro`, `giga_train.launch_from_config`, `giga_train.setup_environment`

#### Modules:
- `train(config: str) -> None`: Main training function that sets up environment and launches training from specified config file.
- CLI entry point using tyro for command-line argument parsing.

---

### `scripts/inference.py`
**Summary:** Continuous action prediction inference script that loads a GigaBrain0 model and dataset, performs prediction on samples, and visualizes ground-truth vs predicted action trajectories.
**Dependencies:** `json`, `os`, `matplotlib`, `numpy`, `torch`, `tyro`, `giga_datasets.load_dataset`, `giga_models.GigaBrain0Pipeline`

#### Modules:
- `inference_giga_brain_0(model_path, data_path, output_path, norm_stats_path, delta_mask, embodiment_id, original_action_dim, action_chunk, enable_2d_traj_output, tokenizer_model_path, fast_tokenizer_path, depth_img_prefix_name, device) -> None`: Main inference function that loads model pipeline, dataset, and generates action predictions with optional trajectory visualization.
- `visualize_result(gt_action, pred_action, out_path, action_names) -> None`: Creates multi-panel matplotlib figure comparing ground-truth and predicted actions across all DOF dimensions with time-series plots.
- `visualize_traj(images, traj_pred, out_path) -> None`: Overlays 2D trajectory predictions on camera image for visualization.

---

### `scripts/inference_client.py`
**Summary:** Simple client script for testing GigaBrain0 inference server that sends random observations and receives action predictions at 1Hz.
**Dependencies:** `time`, `torch`, `tyro`, `giga_models.sockets.RobotInferenceClient`

#### Modules:
- `_random_observation() -> dict`: Creates dummy observation with random images, state, and task placeholder.
- `run_client(host: str, port: int) -> None`: Main client loop connecting to inference server and requesting predictions.
- `model_inference(client: RobotInferenceClient) -> np.ndarray`: Sends observation to client and returns predicted action as numpy array.

---

### `scripts/inference_server.py`
**Summary:** ROS-independent inference server script that sets up GigaBrain0 policy pipeline and runs a server accepting inference requests with multi-modal input (images, state, task) and outputting action predictions.
**Dependencies:** `json`, `types`, `torch`, `tyro`, `giga_models.GigaBrain0Pipeline`, `giga_models.sockets.RobotInferenceServer`

#### Modules:
- `get_policy(model_path, tokenizer_model_path, fast_tokenizer_path, embodiment_id, norm_stats_path, delta_mask, original_action_dim, autoregressive_mode_only, depth_img_prefix_name) -> GigaBrain0Pipeline`: Initializes and compiles GigaBrain0 pipeline with optional depth image support and autoregressive mode, returns pipeline with bound inference method.
- `run_server(model_path, tokenizer_model_path, fast_tokenizer_path, embodiment_id, norm_stats_path, delta_mask, original_action_dim, autoregressive_mode_only, depth_img_prefix_name, host, port) -> None`: Initializes policy and starts RobotInferenceServer listening on specified host:port.

---

### `scripts/inference_agilex_client.py`
**Summary:** ROS-based inference client for real-world AgileX robot manipulation, handling image acquisition from multiple cameras, state synchronization from joint feedback, inference calls, and command publishing for dual manipulators.
**Dependencies:** `threading`, `time`, `collections.deque`, `io.BytesIO`, `numpy`, `rospy`, `torch`, `zmq`, `cv_bridge.CvBridge`, `einops.rearrange`, `geometry_msgs.msg.Twist`, `nav_msgs.msg.Odometry`, `PIL.Image`, `sensor_msgs.msg.Image`, `sensor_msgs.msg.JointState`, `std_msgs.msg.Header`

#### Modules:
- `TorchSerializer`: Serialization utility for torch objects.
  - `to_bytes(data) -> bytes`: Serializes torch object to bytes.
  - `from_bytes(data: bytes)`: Deserializes bytes back to torch object.
- `BaseInferenceClient`: Base ZMQ-based inference client.
  - `__init__(host, port, timeout_ms)`: Initializes ZMQ REQ socket connection.
  - `_init_socket()`: Creates/reinitializes ZMQ socket.
  - `ping() -> bool`: Tests server connectivity.
  - `kill_server()`: Sends kill signal to server.
  - `call_endpoint(endpoint, data, requires_input) -> dict`: Sends serialized request and receives response.
- `RobotInferenceClient(BaseInferenceClient)`: Robot-specific inference client.
  - `inference(observations: dict) -> dict`: Calls inference endpoint with observations.
- `make_infer_data(camera_high, camera_left, camera_right, camera_high_depth, camera_left_depth, camera_right_depth, task_name, qpos) -> dict`: Constructs observation dict with CHW image format and optional depth images.
- `get_obs(ros_operator)`: Continuously polls for synchronized sensor data until successful.
- `inference_process_giga_brain_0(client, ros_operator, task_name, use_robot_base) -> np.ndarray`: Retrieves sensor data, processes images/state, calls inference, returns predicted action.
- `model_inference_giga_brain_0(client, ros_operator, publish_rate, task_name, pos_lookahead_step, max_publish_step, chunk_size, temporal_agg, state_dim, use_robot_base) -> None`: Main control loop that manages timing, temporal aggregation, action clipping, and publishes commands to robot.
- `RosOperator`: ROS interface manager for dual-arm AgileX robot.
  - `__init__(publish_rate, arm_steps_length, use_depth_image, use_robot_base, ...)`: Initializes ROS subscribers/publishers for images, joint state, odometry, and command topics.
  - `init()`: Creates image/state buffers as deques.
  - `puppet_arm_publish(left, right)`: Publishes JointState for left and right arms.
  - `robot_base_publish(vel)`: Publishes Twist for base velocity control.
  - `puppet_arm_publish_continuous(left, right)`: Subscribes and gradually moves arms to target positions.
  - `puppet_arm_publish_continuous_thread(left, right)`: Spawns a thread for continuous arm publishing.
  - `puppet_arm_publish_linear(left, right)`: Performs linear interpolation with 100 steps for arm trajectory execution.
  - `init_ros()`: Sets up ROS subscribers/publishers.
  - `get_frame() -> tuple`: Returns synchronized camera images, joint states, and odometry.

---

### `scripts/inference_discrete_action.py`
**Summary:** Discrete action prediction script running autoregressive action decoding on LeRobot datasets for debugging and validation of discrete token-based action generation.
**Dependencies:** `json`, `os`, `matplotlib.pyplot`, `numpy`, `torch`, `tyro`, `giga_datasets.load_dataset`, `giga_models.GigaBrain0Pipeline`

#### Modules:
- `inference_discrete_action(model_path, data_path, output_path, norm_stats_path, delta_mask, embodiment_id, original_action_dim, action_chunk, tokenizer_model_path, fast_tokenizer_path, depth_img_prefix_name, device) -> None`: Loads model with autoregressive_inference_mode=True, processes dataset samples, generates discrete actions, and visualizes comparisons.
- `visualize_result(gt_action, pred_action, out_path, action_names) -> None`: Creates matplotlib comparison plots with safe handling of shape mismatches.

---

### `scripts/inference_task_planning.py`
**Summary:** Task planning script that predicts current subtask/subgoal from visual observations using GigaBrain0 language model capabilities for hierarchical task decomposition.
**Dependencies:** `json`, `torch`, `tyro`, `giga_datasets.load_dataset`, `giga_models.GigaBrain0Pipeline`

#### Modules:
- `inference_task_planning(model_path, data_path, norm_stats_path, delta_mask, embodiment_id, original_action_dim, action_chunk, tokenizer_model_path, fast_tokenizer_path, depth_img_prefix_name, device) -> None`: Loads model in autoregressive mode with discrete_state_input=False, predicts current subtask using pipeline.predict_current_subtask(), and prints comparisons between predicted and ground-truth subtasks.

---

### `scripts/compute_norm_stats.py`
**Summary:** Utility script for computing and serializing normalization statistics (mean, std, q01, q99) for state and action from LeRobot datasets using running statistics with histogram-based quantile computation.
**Dependencies:** `pathlib`, `numpy`, `numpydantic`, `pydantic`, `tyro`, `giga_datasets.load_dataset`, `giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils` (DeltaActions, PadStatesAndActions), `torch.utils.data`, `tqdm`

#### Modules:
- `NormStats(pydantic.dataclass)`: Data class holding normalization statistics (mean, std, q01, q99).
- `RunningStats`: Running statistics accumulator with on-the-fly quantile computation.
  - `__init__()`: Initializes accumulators for mean, std, min, max, and histogram-based quantiles.
  - `update(batch: np.ndarray) -> None`: Updates running statistics with new batch, handles dynamic min/max expansion with histogram redistribution.
  - `get_statistics() -> NormStats`: Returns computed statistics including quantiles.
  - `_adjust_histograms()`: Redistributes histogram counts when min/max changes.
  - `_update_histograms(batch: np.ndarray)`: Updates quantile histograms with new data.
  - `_compute_quantiles(quantiles: list[float])`: Computes requested quantiles from histograms.
- `_NormStatsDict(pydantic.BaseModel)`: Wrapper for JSON serialization.
- `TransformDataset(Dataset)`: PyTorch dataset wrapper applying transforms and returning specified keys.
  - `__init__(dataset, data_transforms, return_keys)`: Wraps dataset with transforms.
  - `__len__() -> int`: Returns dataset length.
  - `__getitem__(idx) -> dict`: Applies transforms and returns reshaped output.
- `serialize_json(norm_stats: dict[str, NormStats]) -> str`: Serializes norm stats to formatted JSON string.
- `GetEmbodimentId`: Transform that maps robot_type to embodiment_id.
  - `__call__(data: dict) -> dict`: Extracts robot_type and assigns embodiment_id.
- `compute_norm_stats(data_paths, output_path, embodiment_id, delta_mask, sample_rate, action_chunk, action_dim, num_workers) -> None`: Main function that loads datasets, applies transforms, accumulates statistics in parallel using DataLoader, and saves to JSON.

---

### `scripts/convert_from_hdf5.py`
**Summary:** Data conversion utility for transforming HDF5 episode data from AgileX robot to LeRobot v2.1 dataset format with image encoding, state/action normalization, and multi-episode support.
**Dependencies:** `dataclasses`, `pathlib.Path`, `typing`, `h5py`, `numpy`, `psutil`, `torch`, `tqdm`, `tyro`, `giga_datasets.datasets.lerobot_dataset.FastLeRobotDataset`

#### Modules:
- `DatasetConfig(dataclass)`: Configuration for LeRobot dataset creation (use_videos, tolerance_s, image_writer_processes, image_writer_threads, video_backend).
- `DEFAULT_DATASET_CONFIG`: Pre-configured DatasetConfig instance.
- `get_cpu_memory(unit: str) -> str`: Returns formatted current system memory usage.
- `create_empty_dataset(out_dir, repo_id, robot_type, mode, dataset_config) -> FastLeRobotDataset`: Creates empty LeRobot dataset with predefined motor/camera schema and feature definitions.
- `get_cameras(hdf5_files: list[Path]) -> list[str]`: Extracts non-depth camera names from first HDF5 file.
- `has_velocity(hdf5_files: list[Path]) -> bool`: Checks for velocity observations in episodes.
- `has_effort(hdf5_files: list[Path]) -> bool`: Checks for effort observations in episodes.
- `load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]`: Loads image data handling both compressed JPEG and uncompressed formats per camera.
- `load_raw_episode_data(ep_path: Path) -> tuple`: Loads complete episode state/action/images from HDF5, concatenates base actions with arm actions.
- `populate_dataset(dataset, hdf5_files, task, episodes) -> FastLeRobotDataset`: Iterates through episodes, loads frames, adds to dataset with task label, handles graceful error recovery.
- `convert_lerobot(data_path, out_dir, task) -> None`: Main conversion function creating empty dataset and populating with episodes from HDF5 directory.
