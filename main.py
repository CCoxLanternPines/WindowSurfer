import random

from environment import DroneEnv


def run_episode(steps=10):
    env = DroneEnv()
    env.reset()
    for _ in range(steps):
        action = random.choice([0, 1, 2, 3])
        state, reward, done = env.step(action)
        if env.stabilization_timer > 0:
            print("\u26a0\ufe0f  Fear response: Hovering to stabilize.")
        print(f"Tile confidence: {env.confidence_map.get(tuple(env.agent_pos), 0.0):.2f}")
        print(f"Trigger confidence: {env.trigger_confidence}")
        if done:
            break


if __name__ == "__main__":
    run_episode()
