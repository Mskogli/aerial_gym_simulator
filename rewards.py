import torch
import timeit


def transform_actions(actions):
    # type (Tensor) -> Tensor
    s_max = 1
    i_max = 1
    omega_max = 1

    v_x = torch.unsqueeze(
        s_max * ((actions[..., 0] + 1) / 2 * torch.cos(i_max * actions[..., 1])), -1
    )
    v_y = torch.zeros_like(v_x)
    v_z = torch.unsqueeze(
        s_max * (((actions[..., 0] + 1) / 2 * torch.sin(i_max * actions[..., 1]))), -1
    )
    omega_z = torch.unsqueeze(omega_max * actions[..., 2], -1)
    return torch.cat((v_x, v_y, v_z, omega_z), dim=-1)


def calculate_rewards(
    positions,
    prev_positions,
    goal_positions,
    trajectory,
    collisions,
    progress_buffer,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # Tuning parameters
    nu_1 = 1
    nu_2 = 1
    nu_3 = 1

    nu_4_v = torch.tensor([0.1, 0.1, 0.1, 0.1], device=trajectory.device)
    nu_4 = 0.1
    nu_5_v = torch.tensor([0.1, 0.1, 0.1, 0.1], device=trajectory.device)
    nu_5 = 0.1
    nu_6 = 1

    distance_to_target = torch.linalg.vector_norm(goal_positions - positions)
    prev_distance_to_target = torch.linalg.vector_norm(goal_positions - prev_positions)
    distance_diff = prev_distance_to_target - distance_to_target
    trajectory_diff = trajectory[:, :-1] - trajectory[:, 1:]

    ones = torch.ones_like(progress_buffer)
    zeros = torch.zeros_like(progress_buffer)

    # Reward Terms
    r_1 = torch.exp(-(torch.square(distance_to_target) / nu_1))
    r_2 = torch.exp(-(torch.square(distance_to_target) / nu_2))
    r_3 = nu_3 * (distance_diff)
    reward = r_1 + r_2 + r_3

    # Penalty
    p_1 = nu_4 * torch.sum(
        torch.exp(-torch.square(trajectory) / nu_4_v) - 1, dim=(-1, -2)
    )
    p_2 = nu_5 * torch.sum(
        torch.exp(-torch.square(trajectory_diff) / nu_5_v) - 1, dim=(-1, -2)
    )
    p_3 = -nu_6 * torch.where(collisions > 0, ones, zeros)
    penalty = p_1 + p_2 + p_3

    resets_timeout = torch.where(progress_buffer >= max_episode_length - 1, ones, zeros)
    resets = resets_timeout + p_3

    return reward + penalty, resets


if __name__ == "__main__":
    NUM_ENVS = 10
    PRED_HORIZON = 10

    predicted_trajectory = torch.randn((NUM_ENVS, PRED_HORIZON, 3), device="cuda:0")

    transformed_trajectory = transform_actions(predicted_trajectory)
    positions = torch.randn(NUM_ENVS, 3, device="cuda:0")
    prev_positions = torch.rand_like(positions)
    goal_positions = torch.rand_like(prev_positions)
    collisions = torch.zeros(NUM_ENVS, device="cuda:0")
    progress_buffer = torch.zeros(NUM_ENVS, device="cuda:0")
    max_episode_length = 20

    reward, resets = calculate_rewards(
        positions,
        prev_positions,
        goal_positions,
        transformed_trajectory,
        collisions,
        progress_buffer,
        max_episode_length,
    )

    start = timeit.timeit()
    _ = calculate_rewards(
        positions,
        prev_positions,
        goal_positions,
        transformed_trajectory,
        collisions,
        progress_buffer,
        max_episode_length,
    )
    end = timeit.timeit()
