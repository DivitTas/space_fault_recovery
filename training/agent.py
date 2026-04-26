"""Small linear Q-learning agent used by the training pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class LinearQAgent:
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        learning_rate: float = 0.03,
        gamma: float = 0.95,
        gradient_clip: float = 5.0,
        seed: int = 0,
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.rng = random.Random(seed)
        self.weights = [
            [self.rng.uniform(-0.01, 0.01) for _ in range(n_features)]
            for _ in range(n_actions)
        ]
        self.bias = [0.0 for _ in range(n_actions)]

    def q_values(self, features: list[float]) -> list[float]:
        return [
            self.bias[action_idx]
            + sum(weight * value for weight, value in zip(self.weights[action_idx], features))
            for action_idx in range(self.n_actions)
        ]

    def select_action(self, features: list[float], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        return self.greedy_action(features)

    def greedy_action(self, features: list[float]) -> int:
        q_values = self.q_values(features)
        best_value = max(q_values)
        best_actions = [
            action_idx
            for action_idx, value in enumerate(q_values)
            if abs(value - best_value) <= 1e-12
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        features: list[float],
        action_idx: int,
        reward: float,
        next_features: list[float],
        done: bool,
    ) -> float:
        current_q = self.q_values(features)[action_idx]
        next_q = 0.0 if done else max(self.q_values(next_features))
        target_q = reward + self.gamma * next_q
        td_error = target_q - current_q
        clipped_error = max(-self.gradient_clip, min(self.gradient_clip, td_error))

        for feature_idx, feature_value in enumerate(features):
            self.weights[action_idx][feature_idx] += (
                self.learning_rate * clipped_error * feature_value
            )
        self.bias[action_idx] += self.learning_rate * clipped_error

        return 0.5 * td_error * td_error

    def to_dict(
        self,
        *,
        feature_names: list[str],
        action_specs: list[dict[str, Any]],
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "model_type": "linear_q",
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gradient_clip": self.gradient_clip,
            "feature_names": feature_names,
            "action_specs": action_specs,
            "weights": self.weights,
            "bias": self.bias,
            "config": extra_config or {},
        }

    def save(
        self,
        path: Path,
        *,
        feature_names: list[str],
        action_specs: list[dict[str, Any]],
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_dict(
                    feature_names=feature_names,
                    action_specs=action_specs,
                    extra_config=extra_config,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

