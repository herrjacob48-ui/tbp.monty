# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.simulators.robodk.experiments import (
    RoboDKObjectRecognitionExperiment,
)


def _make_mock_experiment(cli_logging: bool = True):
    config = OmegaConf.create({
        "seed": 42,
        "do_train": True,
        "do_eval": True,
        "max_train_steps": 50,
        "max_eval_steps": 50,
        "max_total_steps": 100,
        "n_train_epochs": 2,
        "n_eval_epochs": 1,
        "model_name_or_path": "",
        "min_lms_match": 1,
        "show_sensor_output": False,
        "supervised_lm_ids": [],
        "cli_logging": cli_logging,
        "logging": {"run_name": "test_run"},
        "monty_config": {"learning_modules": {}},
        "environment": {"env_init_args": {"object_names": ["cube", "cylinder"]}},
    })
    exp = RoboDKObjectRecognitionExperiment.__new__(
        RoboDKObjectRecognitionExperiment
    )
    exp.config = config
    exp.cli_logging = cli_logging
    exp.n_train_epochs = 2
    exp.n_eval_epochs = 1
    exp.max_train_steps = 50
    exp.max_eval_steps = 50
    exp.show_sensor_output = False
    exp.train_epochs = 0
    exp.eval_epochs = 0
    exp.train_episodes = 0
    exp.eval_episodes = 0
    exp.total_train_steps = 0
    exp.total_eval_steps = 0
    exp.experiment_mode = ExperimentMode.TRAIN
    exp._eval_records = []
    exp.env_interface = MagicMock()
    exp.env_interface.object_names = ["cube", "cylinder"]
    exp.env_interface.primary_target = {"object": "cube"}
    exp.env_interface.env = MagicMock()
    exp.env_interface.env.prescan_enabled = True
    exp.env_interface.env._current_prescan_view_index = 0
    exp.env_interface.env.prescan_targets = [object(), object()]
    exp.env_interface.env.prescan_target_names = ("View 1", "View 2")
    exp.train_env_interface = exp.env_interface
    exp.eval_env_interface = exp.env_interface
    exp.logger_handler = MagicMock()
    exp.rng = MagicMock()
    exp._rng_seed_history = [42]
    exp.model = MagicMock()
    exp.reset_episode_rng = MagicMock()
    exp._recreation_restore = MagicMock()
    exp._recreation_snapshot = MagicMock()
    return exp


def test_cli_logging_toggle_enabled(caplog):
    exp = _make_mock_experiment(cli_logging=True)
    with caplog.at_level(logging.INFO):
        exp.pre_epoch()
        exp.pre_episode()
        exp.post_episode(25)

    messages = [record.message for record in caplog.records]
    assert any("[TRAINING] Starting Epoch 0/2" in m for m in messages)
    assert any("[TRAIN] Episode 1 | Object: 'cube'" in m for m in messages)
    assert any("View: 1/2 ('View 1')" in m for m in messages)
    assert any("[TRAIN DONE] Episode 1 completed in 25 steps" in m for m in messages)


def test_cli_logging_toggle_disabled(caplog):
    exp = _make_mock_experiment(cli_logging=False)
    with caplog.at_level(logging.INFO):
        exp.pre_epoch()
        exp.pre_episode()
        exp.post_episode(25)

    messages = [record.message for record in caplog.records]
    assert not any("[TRAINING] Starting Epoch" in m for m in messages)
    assert not any("[TRAIN] Episode" in m for m in messages)
    assert not any("[TRAIN DONE]" in m for m in messages)


def test_eval_cli_logging_records_outcomes(caplog):
    exp = _make_mock_experiment(cli_logging=True)
    exp.experiment_mode = ExperimentMode.EVAL
    exp.eval_episodes = 0
    exp.monty_logger = MagicMock()
    exp.monty_logger.data = {
        "BASIC": {
            "eval_stats": {
                0: {
                    "LM_0": {
                        "primary_performance": "correct",
                        "most_likely_object": "cube",
                        "rotation_error": 1.25,
                    }
                }
            }
        }
    }

    with caplog.at_level(logging.INFO):
        exp.post_episode(30)

    messages = [record.message for record in caplog.records]
    assert any("[EVAL RESULT]" in m and "CORRECT" in m for m in messages)
    assert any("Rot Error: 1.25 deg" in m for m in messages)
    assert len(exp._eval_records) == 1
    assert exp._eval_records[0]["is_correct"] is True


def test_eval_cli_logging_with_flushed_logger_and_cycled_target(caplog):
    exp = _make_mock_experiment(cli_logging=True)
    exp.experiment_mode = ExperimentMode.EVAL
    exp.eval_episodes = 0
    exp.monty_logger = MagicMock()
    # Simulate logger flush occurring during post_episode
    exp.monty_logger.data = {"BASIC": {"eval_stats": {}}}

    # Simulate model learning module state
    lm = MagicMock()
    lm.detected_object = "cube"
    lm.terminal_state = "match"
    lm.graph_id_to_target = {"cube": {"cube"}}
    lm.primary_target = "cube"
    lm.buffer.on_object = [1]
    lm.buffer.get_num_matching_steps.return_value = 21
    lm.buffer.get_current_location.return_value = [0.0, 0.0, 0.0]
    lm.get_possible_matches.return_value = ["cube"]
    lm.buffer.stats = {
        "symmetric_rotations": None,
        "detected_rotation_quat": [1.0, 0.0, 0.0, 0.0],
    }
    exp.model.learning_modules = [lm]

    # Simulate env_interface.post_episode() cycling target to the next object
    exp.env_interface.primary_target = {"object": "cube"}
    def cycle_target():
        exp.env_interface.primary_target = {"object": "sphere"}
    exp.env_interface.post_episode.side_effect = cycle_target

    with caplog.at_level(logging.INFO):
        exp.post_episode(21)

    messages = [record.message for record in caplog.records]
    assert any("[EVAL RESULT] Target: 'cube' | Prediction: 'cube' | Status: CORRECT" in m for m in messages)
    assert len(exp._eval_records) == 1
    assert exp._eval_records[0]["target"] == "cube"
    assert exp._eval_records[0]["prediction"] == "cube"
    assert exp._eval_records[0]["is_correct"] is True
