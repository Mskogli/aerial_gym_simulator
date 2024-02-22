# %%
import numpy as np
import h5py


class LogObject:
    def __init__(self):
        self.latents = []
        self.actions = []
        self.states = []

    def append(self, depth_imgs, actions, states):
        self.latents.append(depth_imgs)
        self.actions.append(actions)
        self.states.append(states)


class TrajectoryLogger:
    def __init__(self, log_name: str, num_envs: int, trajectory_length: int) -> None:
        self.num_envs = num_envs
        self.trajectory_length = trajectory_length
        self.file = h5py.File(log_name, "w")
        self.log_buffer = [LogObject() for _ in range(self.num_envs)]
        self.logged_trajetories = 0

    def update_log_buffer(self, depth_imgs, states, actions, resets) -> None:
        for env_id in range(self.num_envs):
            log_object = self.log_buffer[env_id]

            # Assuming depth_imgs, actions, and states are already in an efficient data format
            log_object.append(
                depth_imgs[env_id].tolist(),
                actions[env_id].tolist(),
                states[env_id].tolist(),
            )

            if (
                resets[env_id] and len(log_object.latents) >= self.trajectory_length
            ) or len(log_object.latents) >= self.trajectory_length:
                print(
                    f"Logging trajectories from env: {env_id}, num logged trajectories: {self.logged_trajetories}"
                )
                trajectory_grp = self.file.create_group(
                    f"trajectory_{self.logged_trajetories}"
                )
                self.logged_trajetories += 1

                for i in range(len(log_object.latents)):
                    # Apply compression using gzip with a compression level of 4 as an example
                    dset = trajectory_grp.create_dataset(
                        f"image_{i}",
                        data=log_object.latents[i],
                        compression="gzip",
                        compression_opts=5,
                    )
                    dset.attrs["actions"] = log_object.actions[i]
                    dset.attrs["states"] = log_object.states[i]

                self.log_buffer[env_id] = LogObject()
            elif resets[env_id]:
                self.log_buffer[env_id] = LogObject()


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    latents = np.random.rand(2, 3, 3)
    actions = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
    states = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
    resets = np.array([[1], [1]]).reshape(2, 1)

    # logger = TrajectoryLogger(
    #     "/home/mathias/dev/aerial_gym_simulator/aerial_gym/rl_training/rl_games/quad_depth_imgs",
    #     2,
    #     1,
    # )

    # logger.update_log_buffer(latents, states, actions, resets)

    with h5py.File(
        "/home/mathias/dev/aerial_gym_simulator/aerial_gym/rl_training/rl_games/quad_depth_imgs",
        "r",
    ) as f:
        print(f.keys())

        TRAJ_NUM = 2

        traj_grp = f[f"trajectory_{TRAJ_NUM}"]

        for _, img_dataset in traj_grp.items():
            actions = img_dataset.attrs["actions"]
            imgs = img_dataset[:]
            plt.imshow(imgs)
            plt.show()


# %%
