# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Space Fault Recovery Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SpaceFaultAction, SpaceFaultObservation


class SpaceFaultRecoveryEnv(
    EnvClient[SpaceFaultAction, SpaceFaultObservation, State]
):
    """
    Client for the Space Fault Recovery Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SpaceFaultRecoveryEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.battery_pct)
        ...
        ...     result = client.step(SpaceFaultAction(command="shed_load", target="science_a"))
        ...     print(result.observation.mission_status)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SpaceFaultRecoveryEnv.from_docker_image("space_fault_recovery-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SpaceFaultAction(command="query_thermal"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SpaceFaultAction) -> Dict:
        """
        Convert SpaceFaultAction to JSON payload for step message.

        Args:
            action: SpaceFaultAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict = {"command": action.command}
        if action.target is not None:
            payload["target"] = action.target
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SpaceFaultObservation]:
        """
        Parse server response into StepResult[SpaceFaultObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SpaceFaultObservation
        """
        obs_data = payload.get("observation", {})

        observation = SpaceFaultObservation(
            # Power
            battery_pct=obs_data.get("battery_pct", 100.0),
            battery_drain_rate=obs_data.get("battery_drain_rate", 0.0),
            solar_a_sensor_output_w=obs_data.get("solar_a_sensor_output_w", 0.0),
            solar_b_sensor_output_w=obs_data.get("solar_b_sensor_output_w", 0.0),
            bus_voltage=obs_data.get("bus_voltage", 28.0),
            # Attitude
            star_tracker_deg=obs_data.get("star_tracker_deg", 0.0),
            gyro_deg=obs_data.get("gyro_deg", 0.0),
            sun_sensor_deg=obs_data.get("sun_sensor_deg", 0.0),
            attitude_mode=obs_data.get("attitude_mode", "nominal"),
            rw_status=obs_data.get("rw_status", "nominal"),
            fuel_units=obs_data.get("fuel_units", 50.0),
            # Comms
            signal_strength_db=obs_data.get("signal_strength_db", 0.0),
            transponder_status=obs_data.get("transponder_status", "nominal"),
            link_bandwidth=obs_data.get("link_bandwidth", "high"),
            # Thermal
            battery_temp_c=obs_data.get("battery_temp_c", 20.0),
            heater_status=obs_data.get("heater_status", "online"),
            # Meta
            subsystems_online=obs_data.get("subsystems_online", []),
            step=obs_data.get("step", 0),
            mission_status=obs_data.get("mission_status", "nominal"),
            last_action_result=obs_data.get("last_action_result", "ok"),
            # Base Observation fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
