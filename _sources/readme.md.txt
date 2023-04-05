![jDrones Logo](https://jdrones.janhendrikewers.uk/_static/banner.svg)
[![codecov](https://codecov.io/gh/iwishiwasaneagle/jdrones/branch/master/graph/badge.svg?token=ZILBLXACL6)](https://codecov.io/gh/iwishiwasaneagle/jdrones)[![CD](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CD.yml/badge.svg)](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CD.yml)[![CI](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CI.yml/badge.svg)](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CI.yml)
![docstring coverage](https://jdrones.janhendrikewers.uk/_static/docstr-cov.svg)

## Why

Provide a [gymnasium] style interface using a physics simulation engine ([pybullet] in this case) to drone models. This in-turn enables faster prototyping of controllers, and reinforcement models. I'm maintaining this particular repo for my own research which is focused on waypoint generation, hence my priority is the trajectory drone environment. However, any upgrades and updates on the others would be hugely appreciated.

> :warning: This code is still in alpha and **will** change over time as I use it :warning:

## Environments

The environment documentation can be found [here](https://jdrones.janhendrikewers.uk/envs.html)

### Base Dynamics

1. `PyBulletDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#pybulletdroneenv)
2. `NonLinearDynamicModelDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#nonlineardynamicmodeldroneenv)
3. `LinearDynamicModelDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#lineardynamicmodeldroneenv)

### Attitude
1. `LQRDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#lqrdroneenv)

### Position
1. `FirstOrderPolyPositionDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#firstorderpolypositiondroneenv)
2. `FifthOrderPolyPositionDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#fifthorderpolypositiondroneenv)
3. `FifthOrderPolyPositionWithLookAheadDroneEnv-v0` [:link:](https://jdrones.janhendrikewers.uk/envs.html#fifthorderpolypositionwithlookaheaddroneenv)

## Development

Create the local development environment:

```bash
conda create --name jdrones python=3.10
conda activate jdrones
pip install -r requirements.txt -r tests/requirements.txt
```

The run *****all** tests with

```bash
#!/bin/bash
GIT_DIR=$(git rev-parse --show-toplevel)
PYTHONPATH=$GIT_DIR/src python -m pytest -s -q -n auto $GIT_DIR
PYTHONPATH=$GIT_DIR/src python -m pytest -s -q -n auto --only-integration $GIT_DIR
PYTHONPATH=$GIT_DIR/src python -m pytest -s -q -n auto --only-slow-integration $GIT_DIR
```

## Citations
> [1] J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, and A. P. Schoellig, ‘Learning to Fly -- a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control’. arXiv, Jul. 25, 2021. doi: 10.48550/arXiv.2103.02142.
>
> [2] J. Meyer, A. Sendobry, S. Kohlbrecher, U. Klingauf, and O. von Stryk, ‘Comprehensive Simulation of Quadrotor UAVs Using ROS and Gazebo’, in Simulation, Modeling, and Programming for Autonomous Robots, Berlin, Heidelberg, 2012, pp. 400–411. doi: 10.1007/978-3-642-34327-8_36.


## Future Work

- [ ] Better sensor modelling and kalman filters
- [ ] Performance improvements of simulation using either compiled code or a JIT
- [x] Better controllers
  - LQR
- [x] Better trajectory generation between waypoints
  - First- and fifth-order polynomial trajectory generation
- [x] Examples
- [x] Proper integration testing
- [ ] Higher fidelity motor models

[gymnasium]: https://gymnasium.farama.org/
[pybullet]: https://github.com/bulletphysics/bullet3
