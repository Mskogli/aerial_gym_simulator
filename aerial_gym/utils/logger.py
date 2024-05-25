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
        self.alive_reset_distance_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )
        self.crash_timings_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )
        self.spawn_crash_count_list = deque(maxlen=logger_episode_length)

        self.logger_episode_length = logger_episode_length
        self.ep_length_multiplier = ep_length_multiplier

    def update_lists(
        self,
        success_count=0,
        timeout_count=0,
        crash_count=0,
        spawn_crash_count=0,
        episode_lengths_list=[],
        alive_reset_distance=[],
        crash_timings_list=[],
    ):
        self.crash_count_list.append(crash_count)
        self.success_count_list.append(success_count)
        self.timeout_count_list.append(timeout_count)
        self.episode_lengths_list.extend(episode_lengths_list)
        self.alive_reset_distance_list.extend(alive_reset_distance)
        self.spawn_crash_count_list.append(spawn_crash_count)
        self.crash_timings_list.append(crash_timings_list)

    def get_stats(self):
        success_sum = np.sum(self.success_count_list)
        timeout_sum = np.sum(self.timeout_count_list)
        crash_sum = np.sum(self.crash_count_list)
        spawn_crash_sum = np.sum(self.spawn_crash_count_list)
        total_sum = success_sum + timeout_sum + crash_sum - spawn_crash_sum
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
        alive_reset_distance_mean = (
            np.mean(self.alive_reset_distance_list)
            if len(self.alive_reset_distance_list) > 0
            else 0.0
        )
        alive_reset_distance_std = (
            np.std(self.alive_reset_distance_list)
            if len(self.alive_reset_distance_list) > 0
            else 0.0
        )
        crash_timings_mean = (
            np.mean(self.crash_timings_list)
            if len(self.crash_timings_list) > 0
            else 0.0
        )
        crash_timings_std = (
            np.std(self.crash_timings_list) if len(self.crash_timings_list) > 0 else 0.0
        )
        # make a dict here
        self.logger_stats = {}
        self.logger_stats["success_sum"] = success_sum
        self.logger_stats["timeout_sum"] = timeout_sum
        self.logger_stats["crash_sum"] = crash_sum - spawn_crash_sum
        self.logger_stats["episode_lengths_mean"] = episode_lengths_mean
        self.logger_stats["episode_lengths_std"] = episode_lengths_std
        self.logger_stats["alive_reset_distance_mean"] = alive_reset_distance_mean
        self.logger_stats["alive_reset_distance_std"] = alive_reset_distance_std
        self.logger_stats["num_episodes"] = len(self.success_count_list)
        self.logger_stats["success_rate"] = (
            success_sum / total_sum if total_sum > 0 else 0.0
        )
        self.logger_stats["timeout_rate"] = (
            timeout_sum / total_sum if total_sum > 0 else 0.0
        )
        self.logger_stats["crash_rate"] = (
            (crash_sum - spawn_crash_sum) / total_sum if total_sum > 0 else 0.0
        )
        self.logger_stats["spawn_crash_sum"] = spawn_crash_sum
        self.logger_stats["crash_timings_mean"] = crash_timings_mean
        self.logger_stats["crash_timings_std"] = crash_timings_std

        return self.logger_stats
        # return success_sum, timeout_sum, crash_sum, episode_lengths_mean, episode_lengths_std, alive_reset_distance_mean, alive_reset_distance_std, len(self.success_count_list)
