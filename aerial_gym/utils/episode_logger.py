import numpy as np
from collections import deque


class EpisodeLogger:
    def __init__(self, logger_episode_length=1001, ep_length_multiplier=4):
        self.crash_count_list = deque(maxlen=logger_episode_length)
        self.success_count_list = deque(maxlen=logger_episode_length)
        self.timeout_count_list = deque(maxlen=logger_episode_length)

        self.successful_episode_lengths_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )
        self.successful_path_lengths_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )

        self.crash_episode_lengths_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )
        self.crash_path_lengths_list = deque(
            maxlen=logger_episode_length * ep_length_multiplier
        )

        self.logger_episode_length = logger_episode_length
        self.ep_length_multiplier = ep_length_multiplier

    def update_lists(
        self,
        success_count=0,
        timeout_count=0,
        crash_count=0,
        successfull_episode_length_list=[],
        crash_episode_length_list=[],
        crash_path_lengths_list=[],
        successful_path_lengths_list=[],
    ):
        self.crash_count_list.append(crash_count)
        self.success_count_list.append(success_count)
        self.timeout_count_list.append(timeout_count)

        self.successful_episode_lengths_list.extend(successfull_episode_length_list)
        self.successful_path_lengths_list.extend(successful_path_lengths_list)

        self.crash_episode_lengths_list.extend(crash_episode_length_list)
        self.crash_path_lengths_list.extend(crash_path_lengths_list)

    def get_stats(self):
        success_sum = np.sum(self.success_count_list)
        timeout_sum = np.sum(self.timeout_count_list)
        crash_sum = np.sum(self.crash_count_list)
        total_sum = success_sum + timeout_sum + crash_sum
        successful_episode_lengths_mean = (
            np.mean(self.successful_episode_lengths_list)
            if len(self.successful_episode_lengths_list) > 0
            else 0.0
        )
        succesful_episode_lengths_std = (
            np.std(self.successful_episode_lengths_list)
            if len(self.successful_episode_lengths_list) > 0
            else 0.0
        )
        successful_path_lengths_mean = (
            np.mean(self.successful_path_lengths_list)
            if len(self.successful_path_lengths_list) > 0
            else 0.0
        )
        successful_path_lengths_std = (
            np.std(self.successful_path_lengths_list)
            if len(self.successful_path_lengths_list) > 0
            else 0.0
        )
        crash_episode_lengths_mean = (
            np.mean(self.crash_episode_lengths_list)
            if len(self.crash_episode_lengths_list) > 0
            else 0.0
        )
        crash_episode_lengths_std = (
            np.std(self.crash_episode_lengths_list)
            if len(self.crash_episode_lengths_list) > 0
            else 0.0
        )
        crash_path_lengths_mean = (
            np.mean(self.crash_path_lengths_list)
            if len(self.crash_path_lengths_list) > 0
            else 0.0
        )

        crash_path_lengths_std = (
            np.std(self.crash_path_lengths_list)
            if len(self.crash_path_lengths_list) > 0
            else 0.0
        )

        # make a dict here
        self.logger_stats = {}
        self.logger_stats["success_sum"] = success_sum
        self.logger_stats["timeout_sum"] = timeout_sum
        self.logger_stats["crash_sum"] = crash_sum

        self.logger_stats["successful_episode_lengths_mean"] = (
            successful_episode_lengths_mean
        )
        self.logger_stats["successful_episode_lengths_std"] = (
            succesful_episode_lengths_std
        )
        self.logger_stats["successful_path_lengths_mean"] = successful_path_lengths_mean
        self.logger_stats["successful_path_lengths_std"] = successful_path_lengths_std

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

        self.logger_stats["crash_episode_lengths_mean"] = crash_episode_lengths_mean
        self.logger_stats["crash_episode_lengths_std"] = crash_episode_lengths_std
        self.logger_stats["crash_path_lengths_mean"] = crash_path_lengths_mean
        self.logger_stats["crash_path_lengths_std"] = crash_path_lengths_std

        return self.logger_stats
