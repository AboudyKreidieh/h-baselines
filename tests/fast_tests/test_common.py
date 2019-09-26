"""Contains tests for the model abstractions and different models."""
import unittest
from hbaselines.common.train import parse_options, get_hyperparameters
from hbaselines.common.train import DEFAULT_TD3_HP


class TestTrain(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_parse_options(self):
        # Test the default case.
        args = parse_options("", "", args=["AntMaze"])
        self.assertEqual(args.env_name, "AntMaze")
        self.assertEqual(args.n_training, 1)
        self.assertEqual(args.steps, 1e6)
        self.assertEqual(args.gamma, DEFAULT_TD3_HP["gamma"])
        self.assertEqual(args.tau, DEFAULT_TD3_HP["tau"])
        self.assertEqual(args.batch_size, DEFAULT_TD3_HP["batch_size"])
        self.assertEqual(args.reward_scale, DEFAULT_TD3_HP["reward_scale"])
        self.assertEqual(args.actor_lr, DEFAULT_TD3_HP["actor_lr"])
        self.assertEqual(args.critic_lr, DEFAULT_TD3_HP["critic_lr"])
        self.assertEqual(args.nb_train_steps, DEFAULT_TD3_HP["nb_train_steps"])
        self.assertEqual(args.nb_rollout_steps,
                         DEFAULT_TD3_HP["nb_rollout_steps"])
        self.assertEqual(args.nb_eval_episodes,
                         DEFAULT_TD3_HP["nb_eval_episodes"])
        self.assertEqual(args.render, False)
        self.assertEqual(args.verbose, 2)
        self.assertEqual(args.buffer_size, DEFAULT_TD3_HP["buffer_size"])
        self.assertEqual(args.evaluate, False)
        self.assertEqual(args.meta_period, DEFAULT_TD3_HP["meta_period"])
        self.assertEqual(args.relative_goals, False)
        self.assertEqual(args.off_policy_corrections, False)
        self.assertEqual(args.use_fingerprints, False)
        self.assertEqual(args.centralized_value_functions, False)
        self.assertEqual(args.connected_gradients, False)

        # Test custom cases.
        args = parse_options("", "", args=[
            "AntMaze",
            "--n_training", "1",
            "--steps", "2",
            "--gamma", "3",
            "--tau", "4",
            "--batch_size", "5",
            "--reward_scale", "6",
            "--actor_lr", "7",
            "--critic_lr", "8",
            "--nb_train_steps", "11",
            "--nb_rollout_steps", "12",
            "--nb_eval_episodes", "13",
            "--normalize_observations",
            "--render",
            "--verbose", "14",
            "--buffer_size", "15",
            "--evaluate",
            "--meta_period", "16",
            "--relative_goals",
            "--off_policy_corrections",
            "--use_fingerprints",
            "--centralized_value_functions",
            "--connected_gradients",
        ])
        hp = get_hyperparameters(args)
        self.assertEqual(args.n_training, 1)
        self.assertEqual(args.steps, 2)
        self.assertEqual(hp["gamma"], 3)
        self.assertEqual(hp["tau"], 4)
        self.assertEqual(hp["batch_size"], 5)
        self.assertEqual(hp["reward_scale"], 6)
        self.assertEqual(hp["actor_lr"], 7)
        self.assertEqual(hp["critic_lr"], 8)
        self.assertEqual(hp["nb_train_steps"], 11)
        self.assertEqual(hp["nb_rollout_steps"], 12)
        self.assertEqual(hp["nb_eval_episodes"], 13)
        self.assertEqual(hp["render"], True)
        self.assertEqual(hp["verbose"], 14)
        self.assertEqual(hp["buffer_size"], 15)
        self.assertEqual(args.evaluate, True)
        self.assertEqual(hp["meta_period"], 16)
        self.assertEqual(hp["relative_goals"], True)
        self.assertEqual(hp["off_policy_corrections"], True)
        self.assertEqual(hp["use_fingerprints"], True)
        self.assertEqual(hp["centralized_value_functions"], True)
        self.assertEqual(hp["connected_gradients"], True)


class TestStats(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_reduce_var(self):
        pass

    def test_reduce_std(self):
        pass


if __name__ == '__main__':
    unittest.main()
