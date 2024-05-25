import numpy as np
from collections import deque


class EpisodeLogger:
    def __init__(self, logger_episode_length=1001, ep_length_multiplier=4):
        self.crash_count_list = deque(maxlen=logger_episode_length)
        self.success_count_list = deque(maxlen=logger_episode_length)
        self.timeout_count_list = deque(maxlen=logger_episode_length)
        self.episode_lengths_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )

        self.crash_episode_lengths = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )

        self.spawn_crash_count_list = deque(maxlen=logger_episode_length)
        self.path_lengths_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )
        self.logger_episode_length = logger_episode_length
        self.ep_length_multiplier = ep_length_multiplier

    def update_lists(
        self,
        success_count=0,
        timeout_count=0,
        crash_count=0,
        episode_lengths_list=[],
        crash_episode_lengths=[],
        path_lengths=[],
    ):
        self.crash_count_list.append(crash_count)
        self.success_count_list.append(success_count)
        self.timeout_count_list.append(timeout_count)
        self.episode_lengths_list.extend(episode_lengths_list)
        self.crash_episode_lengths.extend(crash_episode_lengths)
        self.path_lengths_list.extend(path_lengths)

    def get_stats(self):
        success_sum = np.sum(self.success_count_list)
        timeout_sum = np.sum(self.timeout_count_list)
        crash_sum = np.sum(self.crash_count_list)
        total_sum = success_sum + timeout_sum + crash_sum
        episode_lengths_mean = (
            np.mean(self.episode_lengths_list)
            if len(self.episode_lengths_list) > 0
            else 0.0
        )
        episode_lengths_std = (
            np.std(self.episode_lengths_list)
            if len(self.episode_lengths_list) > 0
            else 0.0
        )
        path_lengths_mean = (
            np.mean(self.path_lengths_list) if len(self.path_lengths_list) > 0 else 0.0
        )
        path_lengths_std = (
            np.std(self.path_lengths_list) if len(self.path_lengths_list) > 0 else 0.0
        )
        crash_episode_lengths = (
            np.mean(self.crash_episode_lengths)
            if len(self.crash_episode_lengths) > 0
            else 0.0
        )
        crash_episode_lengths_std = (
            np.std(self.crash_episode_lengths)
            if len(self.crash_episode_lengths) > 0
            else 0.0
        )
        # make a dict here
        self.logger_stats = {}
        self.logger_stats["success_sum"] = success_sum
        self.logger_stats["timeout_sum"] = timeout_sum
        self.logger_stats["crash_sum"] = crash_sum
        self.logger_stats["episode_lengths_mean"] = episode_lengths_mean
        self.logger_stats["episode_lengths_std"] = episode_lengths_std
        self.logger_stats["num_episodes"] = len(self.success_count_list)
        self.logger_stats["success_rate"] = (
            success_sum / total_sum if total_sum > 0 else 0.0
        )
        self.logger_stats["timeout_rate"] = (
            timeout_sum / total_sum if total_sum > 0 else 0.0
        )
        self.logger_stats["crash_rate"] = (
            (crash_sum) / total_sum if total_sum > 0 else 0.0
        )
        self.logger_stats["crash_episode_lengths_mean"] = crash_episode_lengths
        self.logger_stats["crash_episode_lengths_std"] = crash_episode_lengths_std

        self.logger_stats["path_lengths_mean"] = path_lengths_mean
        self.logger_stats["path_lengths_std"] = path_lengths_std

        return self.logger_stats
