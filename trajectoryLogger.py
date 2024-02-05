import jsonlines
import numpy as np
from typing import List
from dataclasses import dataclass


class LogObject:
    def __init__(self):
        self.latents = []
        self.actions = []
        self.states = [] 

    def append(self, latents, actions, states):
        self.latents.append(latents)
        self.actions.append(actions)
        self.states.append(states)

class TrajectoryLogger:

    def __init__(self, log_dir: str, num_envs: int, trajectory_length: int) -> None:
        self.num_envs = num_envs
        self.trajectory_length = trajectory_length
        self.writer = jsonlines.open(log_dir, mode="a")
        self.log_buffer = [LogObject() for _ in range(self.num_envs)]

    def update_log_buffer(self, latents, states, actions, resets) -> None:
        for env_id in range(self.num_envs):
            log_object = self.log_buffer[env_id]

            log_object.append(latents[env_id].tolist(), states[env_id].tolist(), actions[env_id].tolist())

            if resets[env_id] and len(log_object.latents) >= self.trajectory_length:
                self.writer.write({"latents": log_object.latents, "actions": log_object.actions, "states": log_object.states})
                self.log_buffer[env_id] = LogObject() 


if __name__ == "__main__":

    latents = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
    actions = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
    states = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)

    logger = TrajectoryLogger("trajectories.jsonl", 2, 1)

    logger.update_log_buffer(latents, states, actions)

