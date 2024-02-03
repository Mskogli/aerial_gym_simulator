import torch

from typing import Tuple


class DynamicAssetController:

    def __init__(self, device: torch.device, Kp: float = 1.5, Kd: float = 2.00) -> None:
        self.Kp = Kp
        self.Kd = Kd
        self.device = device

    def __call__(
        self, states: torch.tensor, setpoints: torch.tensor
    ) -> Tuple[torch.tensor, ...]:  # 2 Tuple

        position_errors = setpoints - states[:, :, 0:3]
        velocity_errors = -states[:, :, 7:10]

        forces = self.Kp * position_errors + self.Kd * velocity_errors
        forces[:, :, -1] += 9.81
        torques = torch.zeros_like(forces)

        return (forces, torques)
