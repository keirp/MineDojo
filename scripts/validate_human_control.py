import minedojo
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

if __name__ == "__main__":
    env1 = minedojo.make(
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(360, 640),
        world_seed=123,
        seed=42,
    )

    env2 = HumanSurvival(**ENV_KWARGS).make()

    print(f"[INFO] Create a task with prompt: {env1.task_prompt}")

    # Print the obs and action spaces
    print(f"[INFO] Observation space: {env1.observation_space['pov']}")
    print(f"[INFO] Action space: {env1.action_space}")

    print(f"[INFO] Observation space: {env2.observation_space['pov']}")
    print(f"[INFO] Action space: {env2.action_space}")

    obs1 = env1.reset()
    obs2 = env2.reset()

    # print some stats about the obs including the shape, dtype, and min/max values

    print(f"[INFO] obs1 shape: {obs1['pov'].shape}")
    print(f"[INFO] obs1 dtype: {obs1['pov'].dtype}")
    print(f"[INFO] obs1 min: {obs1['pov'].min()}")
    print(f"[INFO] obs1 max: {obs1['pov'].max()}")
    print(f"[INFO] obs1 mean: {obs1['pov'].mean()}")
    print(f"[INFO] obs1 std: {obs1['pov'].std()}")
    print(f"[INFO] obs2 shape: {obs2['pov'].shape}")
    print(f"[INFO] obs2 dtype: {obs2['pov'].dtype}")
    print(f"[INFO] obs2 min: {obs2['pov'].min()}")
    print(f"[INFO] obs2 max: {obs2['pov'].max()}")
    print(f"[INFO] obs2 mean: {obs2['pov'].mean()}")
    print(f"[INFO] obs2 std: {obs2['pov'].std()}")

    # Save the two observations as images
    import matplotlib.pyplot as plt

    plt.imsave("obs1.png", obs1["pov"])
    plt.imsave("obs2.png", obs2["pov"])

