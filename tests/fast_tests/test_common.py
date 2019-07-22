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
        self.assertEqual(args.critic_l2_reg, DEFAULT_TD3_HP["critic_l2_reg"])
        self.assertEqual(args.clip_norm, DEFAULT_TD3_HP["clip_norm"])
        self.assertEqual(args.nb_train_steps, DEFAULT_TD3_HP["nb_train_steps"])
        self.assertEqual(args.nb_rollout_steps,
                         DEFAULT_TD3_HP["nb_rollout_steps"])
        self.assertEqual(args.nb_eval_episodes,
                         DEFAULT_TD3_HP["nb_eval_episodes"])
        self.assertEqual(args.normalize_observations, False)
        self.assertEqual(args.render, False)
        self.assertEqual(args.verbose, 2)
        self.assertEqual(args.buffer_size, DEFAULT_TD3_HP["buffer_size"])
        self.assertEqual(args.evaluate, False)

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
            "--critic_l2_reg", "9",
            "--clip_norm", "10",
            "--nb_train_steps", "11",
            "--nb_rollout_steps", "12",
            "--nb_eval_episodes", "13",
            "--normalize_observations",
            "--render",
            "--verbose", "14",
            "--buffer_size", "15",
            "--evaluate",
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
        self.assertEqual(hp["critic_l2_reg"], 9)
        self.assertEqual(hp["clip_norm"], 10)
        self.assertEqual(hp["nb_train_steps"], 11)
        self.assertEqual(hp["nb_rollout_steps"], 12)
        self.assertEqual(hp["nb_eval_episodes"], 13)
        self.assertEqual(hp["normalize_observations"], True)
        self.assertEqual(hp["render"], True)
        self.assertEqual(hp["verbose"], 14)
        self.assertEqual(hp["buffer_size"], 15)
        self.assertEqual(args.evaluate, True)


class TestStats(unittest.TestCase):
    """A simple test to get Travis running."""

    def test_normalize(self):
        pass

    def test_denormalize(self):
        pass

    def test_reduce_var(self):
        pass

    def test_reduce_std(self):
        pass


if __name__ == '__main__':
    unittest.main()
