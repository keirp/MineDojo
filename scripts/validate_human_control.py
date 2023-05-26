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
    env = minedojo.make(
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(360, 640),
        world_seed=123,
        seed=42,
    )

    env2 = HumanSurvival(**ENV_KWARGS).make()

    print(f"[INFO] Create a task with prompt: {env.task_prompt}")

    # Print the obs and action spaces
    print(f"[INFO] Observation space: {env.observation_space['pov']}")
    print(f"[INFO] Action space: {env.action_space}")

    print(f"[INFO] Observation space: {env2.observation_space['pov']}")
    print(f"[INFO] Action space: {env2.action_space}")