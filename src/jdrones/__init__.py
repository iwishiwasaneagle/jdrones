from gymnasium.envs.registration import register

register("DroneEnv-v0", entry_point="jdrones.envs:DroneEnv")
register(
    "AttitudeAltitudeDroneEnv-v0",
    entry_point="jdrones.envs:AttitudeAltitudeDroneEnv",
)
register(
    "VelHeadAltDroneEnv-v0",
    entry_point="jdrones.envs:VelHeadAltDroneEnv",
)
register(
    "PIDTrajectoryDroneEnv-v0",
    entry_point="jdrones.envs:PIDTrajectoryDroneEnv",
)
