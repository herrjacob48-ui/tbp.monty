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
from typing import Any

from omegaconf import DictConfig

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

__all__ = ["RoboDKObjectRecognitionExperiment"]

logger = logging.getLogger(__name__)


class RoboDKObjectRecognitionExperiment(MontyObjectRecognitionExperiment):
    """Object recognition experiment with enhanced CLI logging for RoboDK.

    Provides clear command-line banners and progress markers that differentiate
    between training and evaluation phases, logging target objects, prescan
    viewpoints, episode step counts, and recognition outcomes.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.cli_logging: bool = bool(config.get("cli_logging", True))
        self._eval_records: list[dict[str, Any]] = []

    def _object_names_list(self) -> list[str]:
        if self.env_interface and hasattr(self.env_interface, "object_names"):
            return list(self.env_interface.object_names)
        env_args = self.config.get("environment", {}).get("env_init_args", {})
        return list(env_args.get("object_names", []))

    def _current_view_str(self) -> str:
        env = getattr(self.env_interface, "env", None)
        if env is not None and getattr(env, "prescan_enabled", False):
            idx = getattr(env, "_current_prescan_view_index", -1)
            targets = getattr(env, "prescan_targets", [])
            names = getattr(env, "prescan_target_names", ())
            if targets and idx >= 0:
                name = names[idx] if idx < len(names) else f"Target {idx}"
                return f" | View: {idx + 1}/{len(targets)} ({name!r})"
        return ""

    def train(self) -> None:
        if self.cli_logging:
            logger.info(
                "\n"
                "================================================================================\n"
                ">>> ROBODK EXPERIMENT: TRAINING PHASE\n"
                "    Configured Epochs    : %d\n"
                "    Max Steps / Episode  : %d\n"
                "    Target Objects       : %s\n"
                "================================================================================",
                self.n_train_epochs,
                self.max_train_steps,
                self._object_names_list(),
            )
        super().train()
        if self.cli_logging:
            logger.info(
                "\n"
                "================================================================================\n"
                "<<< ROBODK EXPERIMENT: TRAINING PHASE COMPLETE\n"
                "    Completed Epochs     : %d\n"
                "    Total Train Episodes : %d\n"
                "    Total Steps Taken    : %d\n"
                "================================================================================",
                self.train_epochs,
                self.train_episodes,
                self.total_train_steps,
            )

    def evaluate(self) -> None:
        if self.cli_logging:
            logger.info(
                "\n"
                "================================================================================\n"
                ">>> ROBODK EXPERIMENT: EVALUATION PHASE\n"
                "    Configured Epochs    : %d\n"
                "    Max Steps / Episode  : %d\n"
                "    Target Objects       : %s\n"
                "================================================================================",
                self.n_eval_epochs,
                self.max_eval_steps,
                self._object_names_list(),
            )
        super().evaluate()
        if self.cli_logging:
            total_eval = len(self._eval_records)
            correct_count = sum(1 for r in self._eval_records if r.get("is_correct"))
            accuracy = (correct_count / total_eval * 100.0) if total_eval > 0 else 0.0
            rot_errors = [
                r["rotation_error"]
                for r in self._eval_records
                if r.get("rotation_error") is not None
            ]
            avg_rot = (sum(rot_errors) / len(rot_errors)) if rot_errors else 0.0
            logger.info(
                "\n"
                "================================================================================\n"
                "<<< ROBODK EXPERIMENT: EVALUATION PHASE COMPLETE\n"
                "    Completed Epochs     : %d\n"
                "    Total Eval Episodes  : %d\n"
                "    Recognition Accuracy : %.1f%% (%d/%d correct)\n"
                "    Avg Rotation Error   : %.2f deg\n"
                "    Total Eval Steps     : %d\n"
                "================================================================================",
                self.eval_epochs,
                self.eval_episodes,
                accuracy,
                correct_count,
                total_eval,
                avg_rot,
                self.total_eval_steps,
            )

    def pre_epoch(self) -> None:
        super().pre_epoch()
        if self.cli_logging:
            if self.experiment_mode is ExperimentMode.TRAIN:
                logger.info(
                    "--- [TRAINING] Starting Epoch %d/%d ---",
                    self.train_epochs,
                    self.n_train_epochs,
                )
            else:
                logger.info(
                    "--- [EVALUATION] Starting Epoch %d/%d ---",
                    self.eval_epochs,
                    self.n_eval_epochs,
                )

    def pre_episode(self) -> None:
        super().pre_episode()
        if self.cli_logging:
            target_name = "Unknown"
            if (
                self.env_interface
                and hasattr(self.env_interface, "primary_target")
                and self.env_interface.primary_target
            ):
                target_name = self.env_interface.primary_target.get(
                    "object", "Unknown"
                )
            view_info = self._current_view_str()
            if self.experiment_mode is ExperimentMode.TRAIN:
                logger.info(
                    "[TRAIN] Episode %d | Object: %r%s",
                    self.train_episodes + 1,
                    target_name,
                    view_info,
                )
            else:
                logger.info(
                    "[EVAL] Episode %d | Target Object: %r%s",
                    self.eval_episodes + 1,
                    target_name,
                    view_info,
                )

    def post_episode(self, last_step: int) -> None:
        target_name = "Unknown"
        if (
            self.env_interface
            and hasattr(self.env_interface, "primary_target")
            and self.env_interface.primary_target
        ):
            target_name = self.env_interface.primary_target.get(
                "object", "Unknown"
            )

        model_lm = None
        if (
            self.model
            and hasattr(self.model, "learning_modules")
            and self.model.learning_modules
        ):
            model_lm = self.model.learning_modules[0]

        super().post_episode(last_step)
        if self.cli_logging:
            if self.experiment_mode is ExperimentMode.TRAIN:
                logger.info(
                    "[TRAIN DONE] Episode %d completed in %d steps | Object: %r",
                    self.train_episodes,
                    last_step,
                    target_name,
                )
            else:
                ep_idx = self.eval_episodes - 1
                stats_pool = {}
                if hasattr(self, "monty_logger") and hasattr(
                    self.monty_logger, "data"
                ):
                    stats_pool = (
                        self.monty_logger.data.get("BASIC", {}).get("eval_stats", {})
                    )
                ep_stats = stats_pool.get(ep_idx, {})
                pred_obj = "Unknown"
                perf = "no_match"
                rot_err = 0.0
                for lm_stats in ep_stats.values():
                    if isinstance(lm_stats, dict) and "primary_performance" in lm_stats:
                        perf = lm_stats.get("primary_performance") or "no_match"
                        pred_obj = (
                            lm_stats.get("result")
                            or lm_stats.get("most_likely_object")
                            or "None"
                        )
                        rot_err = float(lm_stats.get("rotation_error") or 0.0)
                        break

                # If logger flushed stats before this hook, extract from the learning module directly
                if perf == "no_match" and model_lm is not None:
                    try:
                        from tbp.monty.frameworks.utils.logging_utils import (
                            get_graph_lm_episode_stats,
                        )

                        direct_stats = get_graph_lm_episode_stats(model_lm)
                        perf = direct_stats.get("primary_performance") or "no_match"
                        pred_obj = (
                            direct_stats.get("result")
                            or direct_stats.get("most_likely_object")
                            or "None"
                        )
                        rot_err = float(direct_stats.get("rotation_error") or 0.0)
                    except Exception:
                        detected = getattr(model_lm, "detected_object", None)
                        terminal_state = getattr(model_lm, "terminal_state", "no_match")
                        if terminal_state == "match" and detected is not None:
                            pred_obj = detected
                            target_to_graph = getattr(
                                model_lm, "graph_id_to_target", {}
                            ).get(detected, {detected})
                            if target_name in target_to_graph or detected == target_name:
                                perf = "correct"
                            else:
                                perf = "confused"

                is_correct = perf in {"correct", "correct_mlh", "consistent_child_obj"}
                self._eval_records.append(
                    {
                        "target": target_name,
                        "prediction": pred_obj,
                        "status": perf,
                        "is_correct": is_correct,
                        "rotation_error": rot_err,
                        "steps": last_step,
                    }
                )
                status_str = "CORRECT" if is_correct else perf.upper()
                logger.info(
                    "[EVAL RESULT] Target: %r | Prediction: %r | Status: %s | "
                    "Rot Error: %.2f deg | Steps: %d",
                    target_name,
                    pred_obj,
                    status_str,
                    rot_err,
                    last_step,
                )
