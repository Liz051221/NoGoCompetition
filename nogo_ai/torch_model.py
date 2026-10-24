from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .network import Prediction
from .rules import BOARD_SIZE, POINTS, Board, legal_moves


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install it in the uv environment with "
            "`uv sync --extra torch`, or keep using `--backend tiny`."
        ) from exc
    return torch, nn


class _ResidualBlock:
    pass


def _build_modules(
    channels: int,
    blocks: int,
    input_planes: int = 4,
    policy_channels: int = 2,
    value_channels: int = 1,
    value_hidden: int | None = None,
    norm_type: str = "batch",
    policy_head: str = "legacy",
) -> tuple[Any, Any]:
    torch, nn = _import_torch()
    value_hidden = value_hidden or channels
    if norm_type not in {"batch", "group"}:
        raise ValueError("norm_type must be batch or group")
    if policy_head not in {"legacy", "spatial"}:
        raise ValueError("policy_head must be legacy or spatial")

    def make_norm(width: int) -> Any:
        if norm_type == "batch":
            return nn.BatchNorm2d(width)
        groups = min(8, width)
        while width % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, width)

    class ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
            self.bn1 = make_norm(width)
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
            self.bn2 = make_norm(width)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: Any) -> Any:
            residual = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.relu(x + residual)

    class AlphaZeroNoGoNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(input_planes, channels, kernel_size=3, padding=1),
                make_norm(channels),
                nn.ReLU(inplace=True),
            )
            self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
            if policy_head == "spatial":
                # A shared 1x1 output preserves board translations/symmetries
                # much better than a location-specific fully connected layer.
                self.policy = nn.Sequential(
                    nn.Conv2d(channels, policy_channels, kernel_size=1),
                    make_norm(policy_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(policy_channels, 1, kernel_size=1),
                )
            else:
                self.policy = nn.Sequential(
                    nn.Conv2d(channels, policy_channels, kernel_size=1),
                    make_norm(policy_channels),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(policy_channels * POINTS, POINTS),
                )
            self.value_conv = nn.Sequential(
                nn.Conv2d(channels, value_channels, kernel_size=1),
                make_norm(value_channels),
                nn.ReLU(inplace=True),
                nn.Flatten(),
            )
            self.value = nn.Sequential(
                nn.Linear(value_channels * POINTS, value_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(value_hidden, 1),
                nn.Tanh(),
            )

        def forward(self, x: Any) -> tuple[Any, Any]:
            x = self.trunk(self.stem(x))
            policy = self.policy(x)
            if policy_head == "spatial":
                policy = policy.flatten(start_dim=1)
            return policy, self.value(self.value_conv(x)).squeeze(-1)

    return torch, AlphaZeroNoGoNet


def resolve_device(device: str = "auto") -> str:
    torch, _ = _import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def encode_board_tensor(
    board: Board,
    color: int,
    device: str = "auto",
    input_planes: int = 4,
    legal: list[int] | tuple[int, ...] | None = None,
) -> Any:
    torch, _ = _import_torch()
    device = resolve_device(device)
    planes = encode_board_planes(board, color, input_planes, legal=legal)
    return torch.tensor(planes, dtype=torch.float32, device=device)


def encode_board_planes(
    board: Board,
    color: int,
    input_planes: int = 4,
    legal: list[int] | tuple[int, ...] | None = None,
    include_phase: bool = True,
) -> list[list[list[float]]]:
    if input_planes < 4:
        raise ValueError("input_planes must be at least 4")
    planes = [[[0.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] for _ in range(input_planes)]
    legal = set(legal if legal is not None else legal_moves(board, color))
    opponent_legal = set(legal_moves(board, -color)) if input_planes >= 5 else set()
    phase = board.count(0)
    phase_value = (POINTS - phase) / POINTS
    for idx, stone in enumerate(board):
        y = idx // BOARD_SIZE
        x = idx % BOARD_SIZE
        if stone == color:
            planes[0][y][x] = 1.0
        elif stone == -color:
            planes[1][y][x] = 1.0
        else:
            planes[2][y][x] = 1.0
        if idx in legal:
            planes[3][y][x] = 1
        if input_planes >= 5 and idx in opponent_legal:
            planes[4][y][x] = 1
        if input_planes >= 6 and include_phase:
            planes[5][y][x] = phase_value
    return planes


class TorchPolicyValueNet:
    """AlphaZero-style convolutional policy-value network for NoGo."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        channels: int = 16,
        blocks: int = 1,
        input_planes: int = 4,
        policy_channels: int = 2,
        value_channels: int = 1,
        value_hidden: int | None = None,
        norm_type: str = "batch",
        policy_head: str = "legacy",
        phase_plane: bool = True,
        inference_amp: bool = False,
        inference_tf32: bool = False,
    ) -> None:
        checkpoint_data: Any | None = None
        resolved_device = resolve_device(device)
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                torch_for_load, _ = _import_torch()
                checkpoint_data = _safe_torch_load(torch_for_load, checkpoint_path, resolved_device)
                if isinstance(checkpoint_data, dict):
                    channels = int(checkpoint_data.get("channels", channels))
                    blocks = int(checkpoint_data.get("blocks", blocks))
                    input_planes = int(checkpoint_data.get("input_planes", input_planes))
                    policy_channels = int(checkpoint_data.get("policy_channels", policy_channels))
                    value_channels = int(checkpoint_data.get("value_channels", value_channels))
                    value_hidden = int(checkpoint_data.get("value_hidden", value_hidden or channels))
                    norm_type = str(checkpoint_data.get("norm_type", norm_type))
                    # Older checkpoints predate this field and use the
                    # location-specific fully connected policy head.
                    policy_head = str(checkpoint_data.get("policy_head", "legacy"))
                    # Older six-plane checkpoints were trained while this
                    # fractional plane was accidentally stored as uint8.
                    phase_plane = bool(
                        checkpoint_data.get("phase_plane", False)
                        if input_planes >= 6
                        else False
                    )
        value_hidden = value_hidden or channels
        self.torch, model_cls = _build_modules(
            channels,
            blocks,
            input_planes,
            policy_channels,
            value_channels,
            value_hidden,
            norm_type,
            policy_head,
        )
        self.device = resolved_device
        self.channels = channels
        self.blocks = blocks
        self.input_planes = input_planes
        self.policy_channels = policy_channels
        self.value_channels = value_channels
        self.value_hidden = value_hidden
        self.norm_type = norm_type
        self.policy_head = policy_head
        self.phase_plane = bool(phase_plane and input_planes >= 6)
        self.inference_amp = bool(inference_amp and str(self.device).startswith("cuda"))
        self.inference_tf32 = bool(inference_tf32 and str(self.device).startswith("cuda"))
        self._host_input_buffer: Any | None = None
        self._host_legal_buffer: Any | None = None
        self._host_input_capacity = 0
        self.checkpoint_data = checkpoint_data
        self.model = model_cls().to(self.device)
        self.model.eval()
        if str(self.device).startswith("cuda"):
            self._configure_cuda_inference()
        if checkpoint_data is not None:
            self._load_state(checkpoint_data)
        elif checkpoint:
            self.load(checkpoint)

    def load(self, path: str | Path) -> None:
        data = _safe_torch_load(self.torch, Path(path), self.device)
        self.checkpoint_data = data if isinstance(data, dict) else None
        self._load_state(data)

    def _load_state(self, data: Any) -> None:
        state = data["model"] if isinstance(data, dict) and "model" in data else data
        self.model.load_state_dict(state)
        self.model.eval()

    def save(self, path: str | Path, training_state: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "channels": self.channels,
            "blocks": self.blocks,
            "board_size": BOARD_SIZE,
            "input_planes": self.input_planes,
            "policy_channels": self.policy_channels,
            "value_channels": self.value_channels,
            "value_hidden": self.value_hidden,
            "norm_type": self.norm_type,
            "policy_head": self.policy_head,
            "phase_plane": self.phase_plane,
            "architecture": (
                "nogo-v4-spatial" if self.policy_head == "spatial"
                else "nogo-v3" if self.input_planes >= 6 or self.norm_type != "batch"
                else "nogo-v2" if self.input_planes >= 5
                else "legacy"
            ),
        }
        if training_state is not None:
            payload["training_state"] = training_state
        temporary = path.with_name(path.name + ".tmp")
        self.torch.save(payload, temporary)
        temporary.replace(path)

    def predict(self, board: Board, color: int) -> Prediction:
        return self.predict_batch([(board, color)])[0]

    def _host_inputs(
        self,
        positions: list[tuple[Board, int]],
        active: list[int],
        legal_by_position: list[list[int]],
    ) -> tuple[Any, Any]:
        """Encode inputs and legal masks into reusable pinned host buffers."""
        count = len(active)
        pin_memory = str(self.device).startswith("cuda")
        if self._host_input_buffer is None or self._host_input_capacity < count:
            capacity = max(8, 1 << (count - 1).bit_length())
            self._host_input_buffer = self.torch.empty(
                (capacity, self.input_planes, BOARD_SIZE, BOARD_SIZE),
                dtype=self.torch.float32,
                pin_memory=pin_memory,
            )
            self._host_legal_buffer = self.torch.empty(
                (capacity, POINTS),
                dtype=self.torch.bool,
                pin_memory=pin_memory,
            )
            self._host_input_capacity = capacity

        x = self._host_input_buffer[:count]
        legal_mask = self._host_legal_buffer[:count]
        flat = x.view(count, self.input_planes, POINTS)
        flat.zero_()
        legal_mask.zero_()
        boards = self.torch.from_numpy(
            np.asarray([positions[index][0] for index in active], dtype=np.int8)
        )
        colors = self.torch.from_numpy(
            np.asarray([positions[index][1] for index in active], dtype=np.int8)
        ).unsqueeze(1)
        flat[:, 0].copy_(boards == colors)
        flat[:, 1].copy_(boards == -colors)
        flat[:, 2].copy_(boards == 0)
        for row, index in enumerate(active):
            flat[row, 3, legal_by_position[index]] = 1.0
            legal_mask[row, legal_by_position[index]] = True
            if self.input_planes >= 5:
                flat[row, 4, legal_moves(positions[index][0], -positions[index][1])] = 1.0
            if self.input_planes >= 6 and self.phase_plane:
                flat[row, 5].fill_(
                    (POINTS - positions[index][0].count(0)) / POINTS
                )
        return x, legal_mask

    def predict_batch(self, positions: list[tuple[Board, int]]) -> list[Prediction]:
        """Predict several leaves in one GPU call for batched MCTS."""
        if not positions:
            return []
        legal_by_position = [legal_moves(board, color) for board, color in positions]
        active = [index for index, legal in enumerate(legal_by_position) if legal]
        result = [Prediction([0.0] * POINTS, -1.0) for _ in positions]
        if not active:
            return result

        pin_memory = str(self.device).startswith("cuda")
        # These switches are process-global in PyTorch. Re-apply them before
        # each forward pass so two models with different opt-in settings in
        # one Arena process cannot silently change each other's arithmetic.
        self._configure_cuda_inference()
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda",
            dtype=self.torch.float16,
            enabled=self.inference_amp,
        ):
            x_cpu, legal_mask_cpu = self._host_inputs(positions, active, legal_by_position)
            x = x_cpu.to(self.device, non_blocking=pin_memory)
            legal_mask = legal_mask_cpu.to(self.device, non_blocking=pin_memory)
            logits, values = self.model(x)
            invalid_logit = self.torch.finfo(logits.dtype).min
            probs = self.torch.softmax(
                logits.masked_fill(~legal_mask, invalid_logit), dim=1
            ).float().cpu().tolist()
            values_cpu = values.float().cpu().tolist()
            for row, index in enumerate(active):
                result[index] = Prediction(
                    [float(probability) for probability in probs[row]],
                    float(values_cpu[row]),
                )
        return result

    def _configure_cuda_inference(self) -> None:
        if not str(self.device).startswith("cuda"):
            return
        self.torch.backends.cudnn.benchmark = True
        self.torch.backends.cuda.matmul.allow_tf32 = self.inference_tf32
        self.torch.backends.cudnn.allow_tf32 = self.inference_tf32
        self.torch.set_float32_matmul_precision(
            "high" if self.inference_tf32 else "highest"
        )


def _safe_torch_load(torch: Any, path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def upgrade_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    input_planes: int = 5,
    policy_channels: int = 4,
    value_channels: int = 4,
    value_hidden: int = 128,
    norm_type: str = "batch",
    policy_head: str = "legacy",
) -> dict[str, Any]:
    source = TorchPolicyValueNet(checkpoint=checkpoint, device="cpu")
    if norm_type != source.norm_type:
        raise ValueError(
            "normalization changes are not function-preserving: "
            f"source={source.norm_type} target={norm_type}; "
            "train the new normalization from scratch or use distillation"
        )
    if policy_head != source.policy_head:
        raise ValueError(
            "policy-head changes are not function-preserving: "
            f"source={source.policy_head} target={policy_head}; "
            "use --checkpoint directly for same-head fine-tuning, or train the new head from scratch"
        )
    target = TorchPolicyValueNet(
        device="cpu",
        channels=source.channels,
        blocks=source.blocks,
        input_planes=input_planes,
        policy_channels=policy_channels,
        value_channels=value_channels,
        value_hidden=value_hidden,
        norm_type=norm_type,
        policy_head=policy_head,
    )
    torch = target.torch
    source_state = source.model.state_dict()
    target_state = target.model.state_dict()
    copied_tensors = 0

    with torch.no_grad():
        for key, source_tensor in source_state.items():
            target_tensor = target_state.get(key)
            if target_tensor is None or source_tensor.ndim != target_tensor.ndim:
                continue
            if source_tensor.ndim == 0:
                target_tensor.copy_(source_tensor)
            else:
                slices = tuple(slice(0, min(a, b)) for a, b in zip(source_tensor.shape, target_tensor.shape))
                target_tensor[slices].copy_(source_tensor[slices])
            copied_tensors += 1

        # Preserve the old function exactly while leaving new features trainable.
        target_state["stem.0.weight"][:, source.input_planes :].zero_()
        old_policy_features = source.policy_channels * POINTS
        if target.policy_head == "legacy":
            target_state["policy.4.weight"][:, old_policy_features:].zero_()
        old_value_features = source.value_channels * POINTS
        target_state["value.0.weight"][: source.value_hidden, old_value_features:].zero_()
        target_state["value.2.weight"][:, source.value_hidden :].zero_()

    target.model.load_state_dict(target_state)
    target.model.eval()
    target.save(output)
    return {
        "copied_tensors": copied_tensors,
        "source_parameters": sum(parameter.numel() for parameter in source.model.parameters()),
        "target_parameters": sum(parameter.numel() for parameter in target.model.parameters()),
        "input_planes": target.input_planes,
        "policy_channels": target.policy_channels,
        "value_channels": target.value_channels,
        "value_hidden": target.value_hidden,
        "norm_type": target.norm_type,
        "policy_head": target.policy_head,
        "phase_plane": target.phase_plane,
        "policy_projection": False,
    }
