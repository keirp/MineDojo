import minedojo


if __name__ == "__main__":
    env = minedojo.make(
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(360, 640),
        world_seed=123,
        seed=42,
    )

    print(f"[INFO] Create a task with prompt: {env.task_prompt}")

    # Print the obs and action spaces
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    env.reset()
    for _ in range(20):
        obs, reward, done, info = env.step(env.action_space.no_op())
    env.close()

    print("[INFO] Installation Success")
