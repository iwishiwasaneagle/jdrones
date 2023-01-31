![jDrones Logo](https://jdrones.janhendrikewers.uk/_static/banner.svg)
[![codecov](https://codecov.io/gh/iwishiwasaneagle/jdrones/branch/master/graph/badge.svg?token=ZILBLXACL6)](https://codecov.io/gh/iwishiwasaneagle/jdrones)[![CD](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CD.yml/badge.svg)](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CD.yml)[![CI](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CI.yml/badge.svg)](https://github.com/iwishiwasaneagle/jdrones/actions/workflows/CI.yml)
![docstring coverage](https://jdrones.janhendrikewers.uk/_static/docstr-cov.svg)

## Why

Provide a [gymnasium] style interface using a physics simulation engine ([pybullet] in this case) to drone models. This in-turn enables faster prototyping of controllers, and reinforcement models. I'm maintaining this particular repo for my own research which is focused on waypoint generation, hence my priority is the trajectory drone environment. However, any upgrades and updates on the others would be hugely appreciated.

## Development

```bash
conda create --name jdrones
conda activate jdrones
pip install -r requirements.txt -r tests/requirements.txt
PYTHONPATH=$PWD/src pytest tests
```

## Citations
> [1] J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, and A. P. Schoellig, ‘Learning to Fly -- a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control’. arXiv, Jul. 25, 2021. doi: 10.48550/arXiv.2103.02142.
>
> [2] J. Meyer, A. Sendobry, S. Kohlbrecher, U. Klingauf, and O. von Stryk, ‘Comprehensive Simulation of Quadrotor UAVs Using ROS and Gazebo’, in Simulation, Modeling, and Programming for Autonomous Robots, Berlin, Heidelberg, 2012, pp. 400–411. doi: 10.1007/978-3-642-34327-8_36.


## Future Work

- [ ] Better sensor modelling and kalman filters
- [ ] Performance improvements of simulation using either compiled code or a JIT
- [ ] Better controllers
- [ ] Better trajectory generation between waypoints
- [ ] Examples
- [ ] Proper integration testing
- [ ] Higher fidelity motor models

[gymnasium]: https://gymnasium.farama.org/
[pybullet]: https://github.com/bulletphysics/bullet3
