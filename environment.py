class DroneEnv:
    """Grid-based survival environment with confidence and fear mechanics."""

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Agent starts at center of grid
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.confidence_map = {}
        self.trigger_confidence = {
            "low_confidence_tile": 0.0,
            "sudden_direction_change": 0.0,
        }
        self.fear_triggers = {
            "low_confidence_tile": False,
            "sudden_direction_change": False,
        }
        self.stabilization_timer = 0
        self.last_direction = None
        return self.get_state()

    def get_state(self):
        return tuple(self.agent_pos)

    def _move_agent(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size - 1, x + 1)
        self.agent_pos = [x, y]

    def step(self, action):
        # Hover if stabilizing
        if self.stabilization_timer > 0:
            self.stabilization_timer -= 1
            return self.get_state(), 0.05, False

        # Triggered fear leads to stabilization
        if any(self.fear_triggers.values()):
            self.stabilization_timer = 3
            self.fear_triggers = {k: False for k in self.fear_triggers}
            return self.get_state(), 0.05, False

        # Move the agent
        self._move_agent(action)
        pos = tuple(self.agent_pos)

        # Update tile confidence
        self.confidence_map[pos] = min(
            self.confidence_map.get(pos, 0.0) + 0.1, 1.0
        )

        # Detect low-confidence tile
        if (
            self.confidence_map.get(pos, 0.0) < 0.2
            and self.trigger_confidence["low_confidence_tile"] < 0.9
        ):
            self.fear_triggers["low_confidence_tile"] = True

        # Detect sudden direction change
        if self.last_direction:
            dx = pos[0] - self.last_direction[0]
            dy = pos[1] - self.last_direction[1]
            if abs(dx) + abs(dy) > 1 and self.trigger_confidence[
                "sudden_direction_change"
            ] < 0.9:
                self.fear_triggers["sudden_direction_change"] = True
        self.last_direction = pos

        # Increase trigger confidence after surviving
        for trigger in self.trigger_confidence:
            if not self.fear_triggers[trigger]:
                self.trigger_confidence[trigger] = min(
                    self.trigger_confidence[trigger] + 0.01, 1.0
                )

        return self.get_state(), 0.0, False
