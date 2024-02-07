import numpy as np
import h5py


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

    def __init__(self, log_name: str, num_envs: int, trajectory_length: int) -> None:
        self.num_envs = num_envs
        self.trajectory_length = trajectory_length
        self.file = h5py.File(log_name, "w")
        self.log_buffer = [LogObject() for _ in range(self.num_envs)]
        self.logged_trajetories = 0

    def update_log_buffer(self, latents, states, actions, resets) -> None:
        for env_id in range(self.num_envs):
            log_object = self.log_buffer[env_id]

            log_object.append(
                latents[env_id].tolist(),
                actions[env_id].tolist(),
                states[env_id].tolist(),
            )

            if resets[env_id] and len(log_object.latents) >= self.trajectory_length:
                trajectory_grp = self.file.create_group(
                    f"trajectory_{self.logged_trajetories}"
                )
                self.logged_trajetories += 1

                for i in range(len(log_object.latents)):
                    dset = trajectory_grp.create_dataset(
                        f"image_{i}", data=log_object.latents[i]
                    )
                    dset.attrs["actions"] = log_object.actions[i]
                    dset.attrs["states"] = log_object.states[i]

                self.log_buffer[env_id] = LogObject()


if __name__ == "__main__":

    latents = np.random.rand(2, 3, 3)
    actions = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
    states = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
    resets = np.array([[1], [1]]).reshape(2, 1)

    logger = TrajectoryLogger("quad_depth_trajectories", 2, 1)

    logger.update_log_buffer(latents, states, actions, resets)

    with h5py.File("quad_depth_trajectories", "r") as f:
        print("Keys", f.keys())
        print(f["trajectory_1"]["image_0"][:])
