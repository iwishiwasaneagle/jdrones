# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2024-08-07

### Bug Fixes

- Use py3.11 as it still ships pkg_resources by default which docstr-coverage needs [(25d649f)](https://github.com/iwishiwasaneagle/jdrones/commit/25d649f738b6c0d406c56e7cf58505b92a894245)
- Issue was with CI.yml not installing jdrones [(7898915)](https://github.com/iwishiwasaneagle/jdrones/commit/7898915d91b9be0725743c46f6f4839285f2298f)
- Check for different env types [(5aae992)](https://github.com/iwishiwasaneagle/jdrones/commit/5aae99229f1fb3cf094d7609eaf30b2da7e9a78f)
- Check for different env types [(d24b766)](https://github.com/iwishiwasaneagle/jdrones/commit/d24b7667464ad48d7d2d01fd251c9ea02d5002a7)
- Legacy imports [(f6f196c)](https://github.com/iwishiwasaneagle/jdrones/commit/f6f196c913385120ae472f31aa746d0b76c6cd89)
- Tensorflow-probability 0.24.0 not compatible with sbx 0.12.0 [(6441dc2)](https://github.com/iwishiwasaneagle/jdrones/commit/6441dc200fc65463f0b97def3bf12605a0861d49)
- Reset to initial state z not current z [(5679bf1)](https://github.com/iwishiwasaneagle/jdrones/commit/5679bf14edaa32eb37f1f0e2a7c93f344d73d6e5)
- Re-added is_success logging [(f429dd1)](https://github.com/iwishiwasaneagle/jdrones/commit/f429dd11e0f26409fbbb457b6de19481fc3602a1)
- Switch to vector-based velocity method [(04ea745)](https://github.com/iwishiwasaneagle/jdrones/commit/04ea7458049a397b400bd863eb9c633e6afdc1c2)
- Enable policy net_arch for RecurrentPPO to be customizable from the CLI [(dcea513)](https://github.com/iwishiwasaneagle/jdrones/commit/dcea513be879073e8e580c62097de35e4c9746e6)
- Success distance is 1 not 1.5 [(32e96bd)](https://github.com/iwishiwasaneagle/jdrones/commit/32e96bd5241a21ef58b79a56e6567886238a6e65)
- Don't break the loop if a target is reached to ensure that time stays synchronized [(7a2db4a)](https://github.com/iwishiwasaneagle/jdrones/commit/7a2db4af0556687502a21868f84ae36704f9eb77)
- Switch to SB3's DummyVecEnv to handle the multi-env logic [(29bbe52)](https://github.com/iwishiwasaneagle/jdrones/commit/29bbe52c9552196f7aa6dde58c2f1d66f3cf3ff6)
- Properly handle collision reward [(703274e)](https://github.com/iwishiwasaneagle/jdrones/commit/703274e8bc450048591a791a6d1b91b545714a11)
- Ensure eval env is the same as training env [(3a7833f)](https://github.com/iwishiwasaneagle/jdrones/commit/3a7833f62b3d1f5e260015d405f294da7a77dc58)
- Re-include time in the observation [(a5c30e4)](https://github.com/iwishiwasaneagle/jdrones/commit/a5c30e436a27e1779220ee70d25d82b7e06e689d)
- Ensure observation is (1,X) in shape [(a7ee23c)](https://github.com/iwishiwasaneagle/jdrones/commit/a7ee23c0eeb96866f16c30fb9843c2e643ecaf45)
- Disable JIT for codecov, but also run the same setup with jit without codecov [(4f2e626)](https://github.com/iwishiwasaneagle/jdrones/commit/4f2e626f2ca4f971ce111d3f8722cac736bf79db)

### Documentation

- Quick readme for the drl example [(e1cdc81)](https://github.com/iwishiwasaneagle/jdrones/commit/e1cdc816541db762345e2471e791ff014662d71c)

### Features

- Install pytest-testmon from git and run in CI (#61) [(b54d03b)](https://github.com/iwishiwasaneagle/jdrones/commit/b54d03bc65efa0a26685b76a0bc3fd71901e9e71)
- Energy consumption [(5dc0a29)](https://github.com/iwishiwasaneagle/jdrones/commit/5dc0a299d21ebc5e264b67c74ec4f018edd24671)
- Energy consumption [(23ca884)](https://github.com/iwishiwasaneagle/jdrones/commit/23ca8841b3a22df96be228695bb56f1b1d490a41)
- Energy consumption [(75c28be)](https://github.com/iwishiwasaneagle/jdrones/commit/75c28be2787ab226c48e568c139a2cdb63a769d5)
- Switch to Hatch from setuptools [(8ce1d14)](https://github.com/iwishiwasaneagle/jdrones/commit/8ce1d1409f30b31c655999ba322f96c667ddbfe4)
- DRL controlled drone going from A to hovering at B [(675947e)](https://github.com/iwishiwasaneagle/jdrones/commit/675947e6ca17b0f7b390535305794e2f60820d63)
- A working DRL example with plotting [(2e42527)](https://github.com/iwishiwasaneagle/jdrones/commit/2e42527e3dd5e2b8e3fbaf0e4534687ccfc157be)
- Use optuna for hyperparameter sweep in DRL example [(f244822)](https://github.com/iwishiwasaneagle/jdrones/commit/f244822abcf555ce7d336d93d837cc621b1d9388)
- A working DRL example with plotting [(34014e2)](https://github.com/iwishiwasaneagle/jdrones/commit/34014e2ab370a0958a6be6eacc0500ec1af636a3)
- Add wandb to example [(a5204d3)](https://github.com/iwishiwasaneagle/jdrones/commit/a5204d36b5268265e67abde9ecf3cb3a730d13c7)
- LQR+DRL meta-controller framework [(2be6be6)](https://github.com/iwishiwasaneagle/jdrones/commit/2be6be64ced0410fe20b12e9839646bff00c4ded)
- Additional eval logging [(da18e46)](https://github.com/iwishiwasaneagle/jdrones/commit/da18e4677997ce009bd5000da6a8600e98490c57)
- Change LQR tuner to be velocity centric [(c0f4849)](https://github.com/iwishiwasaneagle/jdrones/commit/c0f4849e91e947fe257021aa7a317fe9b3787935)
- Add 2D plot to graphing callback [(38b4a5e)](https://github.com/iwishiwasaneagle/jdrones/commit/38b4a5e0f94abe2a4dfae67a6889827a817a577d)
- Swap to velocity-based action space [(3afe0a2)](https://github.com/iwishiwasaneagle/jdrones/commit/3afe0a2dc6eeb8b95d0f82b65e8bb62cf349a62e)
- Add colors and circles to 2D position plot to show the targets, their radius and order [(e8d6e71)](https://github.com/iwishiwasaneagle/jdrones/commit/e8d6e715cec9735162b7b74d4a3acc9f63778103)
- Multi-env DRL [(d21b26d)](https://github.com/iwishiwasaneagle/jdrones/commit/d21b26d0f817a861d39e7d9e461bfe62ddde5ca9)
- Extend to any n env [(04d3e72)](https://github.com/iwishiwasaneagle/jdrones/commit/04d3e721ed6a3a279b1a27897a46090765811fc3)
- Number of sub-envs from CLI [(91209c0)](https://github.com/iwishiwasaneagle/jdrones/commit/91209c056a80aa255343970abf341b21006ae480)
- Tensorboard and optuna for the 3d drl example [(73bed0d)](https://github.com/iwishiwasaneagle/jdrones/commit/73bed0d692b00e72fa28dabd695637e0876bf12d)
- Optimal quintic poly trajectory generation [(aa8b488)](https://github.com/iwishiwasaneagle/jdrones/commit/aa8b488bcb4799ec6b11a4558058b8f4accd31a6)

### Miscellaneous Tasks

- Remove the status updater. Was causing more issues than was worth. [(bd80432)](https://github.com/iwishiwasaneagle/jdrones/commit/bd80432b023aef0142b2968d79a04cfd1187d560)
- Bump gymnasium from 0.28.1 to 0.29.1 [(234a5bb)](https://github.com/iwishiwasaneagle/jdrones/commit/234a5bb535bbdc6854c192d694b85d75f056e381)
- Bump pydantic from 1.10.6 to 2.3.0 [(2f309aa)](https://github.com/iwishiwasaneagle/jdrones/commit/2f309aa6c038b3fcb4010fa923c5c693f8edab80)
- Bump pandas from 2.0.0 to 2.1.0 [(51ff0a1)](https://github.com/iwishiwasaneagle/jdrones/commit/51ff0a194114deea4a46e127e04eae0a104dca28)
- Bump JamesIves/github-pages-deploy-action [(71b93c7)](https://github.com/iwishiwasaneagle/jdrones/commit/71b93c7bcdc310ec0d2a096603b4d6f100ba8907)
- Bump actions/setup-python from 4 to 5 [(d56ff9e)](https://github.com/iwishiwasaneagle/jdrones/commit/d56ff9eff9551ec447fd8f780f41c89009cb35c7)
- Bump actions/upload-artifact from 3 to 4 [(d406f45)](https://github.com/iwishiwasaneagle/jdrones/commit/d406f4559acabd1ae0d3686c199ee5d89180dc9c)
- Bump actions/download-artifact from 3 to 4 [(0f3299c)](https://github.com/iwishiwasaneagle/jdrones/commit/0f3299c98d06b476774a4a84b219daeabd48408b)
- Bump actions/checkout from 3 to 4 [(465029a)](https://github.com/iwishiwasaneagle/jdrones/commit/465029a6e82ab532226092a3c6f3ae5e211f9cd1)
- Flake8 config in .flake8 ini file [(871c0ca)](https://github.com/iwishiwasaneagle/jdrones/commit/871c0cade5142c1a7c15c5235816db765cfc59af)
- Add a fifth action that controls the scaling of the propeller actions [(17a6e79)](https://github.com/iwishiwasaneagle/jdrones/commit/17a6e792447bfe52632449e6f158781f65c737c7)
- Explicitly add gitignored file __version__.py [(c2d4183)](https://github.com/iwishiwasaneagle/jdrones/commit/c2d418350a885e796aa0d862e85a618ff1f50ae1)
- Use requirements.txt file for all so that dependabot can auto-update [(6b68265)](https://github.com/iwishiwasaneagle/jdrones/commit/6b68265f3186180cc4325e55a17287f4294c2650)
- __version__ -> _version to avoid name conflicts [(f93a6df)](https://github.com/iwishiwasaneagle/jdrones/commit/f93a6df3a62680193d354e809fc3a731081d113a)
- Bump orhun/git-cliff-action from 2 to 3 [(e4c9070)](https://github.com/iwishiwasaneagle/jdrones/commit/e4c9070f73f0979c5accce95f48cf1dfe7942f26)
- Bump codecov/codecov-action from 3.1.1 to 4.0.2 [(dd37dc5)](https://github.com/iwishiwasaneagle/jdrones/commit/dd37dc5843b1755ad8e19e3bcef8a9a624dc8a71)
- Bump matplotlib from 3.7.1 to 3.8.3 [(0f7149f)](https://github.com/iwishiwasaneagle/jdrones/commit/0f7149f8472a5d3b70c5254a280dfa222402632f)
- Bump seaborn from 0.12.2 to 0.13.2 [(3af4c7b)](https://github.com/iwishiwasaneagle/jdrones/commit/3af4c7b8a858dd056f173c16d51239894a280b07)
- Bump scipy from 1.10.1 to 1.12.0 [(386e883)](https://github.com/iwishiwasaneagle/jdrones/commit/386e88328afd7e449f325581e284ac5fc6eeb27c)
- Bump numpy from 1.24.2 to 1.26.4 [(8a02a35)](https://github.com/iwishiwasaneagle/jdrones/commit/8a02a35f10ea5b130ba38b20bf88817e3a22f2ec)
- Bump pydantic from 2.3.0 to 2.6.2 [(ff1f8a6)](https://github.com/iwishiwasaneagle/jdrones/commit/ff1f8a64ddeb5b6a6f2d19c1875076f3c0e62539)
- Switch to using "unwrapped" [(a5467e2)](https://github.com/iwishiwasaneagle/jdrones/commit/a5467e28f7716b467d5cdce533ba8605da8b5e27)
- Import order [(d6f9879)](https://github.com/iwishiwasaneagle/jdrones/commit/d6f987957b8c08ceefadc28d9d76f20d0a0b3c4d)
- Further edits to energy example [(5371a38)](https://github.com/iwishiwasaneagle/jdrones/commit/5371a38b3e8fce02d05da157f8718c8af0d5667f)
- Try out sbx [(556968c)](https://github.com/iwishiwasaneagle/jdrones/commit/556968c2c9386db9ec6d6af8b035cfeab2d996dd)
- Update types [(28f728c)](https://github.com/iwishiwasaneagle/jdrones/commit/28f728cbd965d0b5750e06672c4d054d83322cea)
- Allow a reset state to be given within options [(11bc080)](https://github.com/iwishiwasaneagle/jdrones/commit/11bc080cb77f759d4d8d123a2c3eead59955c4ed)
- Bump codecov/codecov-action from 4.0.2 to 4.1.0 [(bb6c90e)](https://github.com/iwishiwasaneagle/jdrones/commit/bb6c90e1c49e7f5b9b24345b6bc174da85c03890)
- Bump pandas from 2.1.0 to 2.2.1 [(cd083b1)](https://github.com/iwishiwasaneagle/jdrones/commit/cd083b16b30d179360759873d93f64f0a9fdac87)
- Make sure num_timesteps is set properly [(8127e41)](https://github.com/iwishiwasaneagle/jdrones/commit/8127e419e516c583e338f434a1563412444e36a2)
- Make sure num_timesteps is set properly [(83606e4)](https://github.com/iwishiwasaneagle/jdrones/commit/83606e4f7688b7ae29c4777f104cdee1ea6b2a48)
- Make sure num_timesteps is set properly [(6a8b455)](https://github.com/iwishiwasaneagle/jdrones/commit/6a8b4556b121bce7846a46a213f3a38b0c1cc618)
- Change example rewards scheme to be normalized [(3271621)](https://github.com/iwishiwasaneagle/jdrones/commit/3271621f34f579f091e1e89ee5721c20a36b8509)
- Use subprocvecenv when n_envs > 1 [(f3b724f)](https://github.com/iwishiwasaneagle/jdrones/commit/f3b724f9dbbb1139f0757c8a681c50d8a9cf36ac)
- Make Monitor log other info keywords [(010ab83)](https://github.com/iwishiwasaneagle/jdrones/commit/010ab83abc0fc477affa5917e763e86cc63d5de8)
- Normalize reward by maximum action command T [(f82ffff)](https://github.com/iwishiwasaneagle/jdrones/commit/f82ffff95e836f6e75992e2c42a8fa799fbfd3f0)
- Normalize reward by maximum action command T [(ee96f4a)](https://github.com/iwishiwasaneagle/jdrones/commit/ee96f4a106b77ca82e0ed8dc996059f8cc77b74d)
- Normalize reward by maximum action command T [(fd66268)](https://github.com/iwishiwasaneagle/jdrones/commit/fd662682ba3a98f025a30c0d4973a21cf15925ac)
- Play around with how the control is structured. Only let policy define target x and y pos [(e0d7690)](https://github.com/iwishiwasaneagle/jdrones/commit/e0d7690f0b34bba9442c0bcaa0dd36640b3360c2)
- Enable maximum time to be specified from CLI [(294cfbc)](https://github.com/iwishiwasaneagle/jdrones/commit/294cfbc5356aebde33ce8ebb03b3d627b79db929)
- Enable maximum time to be specified from CLI [(844d19b)](https://github.com/iwishiwasaneagle/jdrones/commit/844d19b5d9eae1d88fcebb64e83f33f7727c126b)
- Reduce observation space to x-y plane [(04347f6)](https://github.com/iwishiwasaneagle/jdrones/commit/04347f6da193f6c878e06cde0a5f611c5b58eefd)
- Include angular position and velocity in the observation [(f3dcd89)](https://github.com/iwishiwasaneagle/jdrones/commit/f3dcd89be12b6250278b6f47e9fe59ada626eae4)
- Plot actions [(0216f3e)](https://github.com/iwishiwasaneagle/jdrones/commit/0216f3e3a97d7e63443aa849ca9f3be26734b9b6)
- Adjust observation shape to match observation [(94cbc0e)](https://github.com/iwishiwasaneagle/jdrones/commit/94cbc0ec23a7e91236da09d7ef14dcb9ec029394)
- Ensure fig.tight_layout() is called for plotting during callbacks to avoid weird layouts [(6490375)](https://github.com/iwishiwasaneagle/jdrones/commit/64903758c79be8fc916588c937d5ff19402a480d)
- Set target z to env z [(769741b)](https://github.com/iwishiwasaneagle/jdrones/commit/769741b0fa7b73a7c9fee436dd65f0c3873cff7a)
- Use env.unwrapped pattern [(d8674e3)](https://github.com/iwishiwasaneagle/jdrones/commit/d8674e3b01c46664dbfb965c35447dd02b14e42b)
- Ensure target is set before it is used within reset [(8a8e3d5)](https://github.com/iwishiwasaneagle/jdrones/commit/8a8e3d515476acc1c2f0eab59c6b1dd8de94dec9)
- Adjust reward normalization [(6d7a036)](https://github.com/iwishiwasaneagle/jdrones/commit/6d7a036cd5dbcc15ccd4878881644925a973f7fa)
- Change step simulation time to be 3 [(3d79c69)](https://github.com/iwishiwasaneagle/jdrones/commit/3d79c698b68ad0ae964a24e40229835ffe4c1a1d)
- Expose initial_state [(ab834d7)](https://github.com/iwishiwasaneagle/jdrones/commit/ab834d7e0b1cbf497316ca6b2b78b8f418709a9e)
- Move to relative control rather than absolute [(c1df545)](https://github.com/iwishiwasaneagle/jdrones/commit/c1df545b7ba788482a2559021962adb0bcdba7d1)
- Reduce step sim time down to 0.25s [(6334782)](https://github.com/iwishiwasaneagle/jdrones/commit/6334782d50242f2840873678cc245a5ce9d9c921)
- Reduce eval episodes to 10 [(f930698)](https://github.com/iwishiwasaneagle/jdrones/commit/f930698a149014f0b2651279cfb7c071bcace0af)
- Log number of targets achieved [(b07881e)](https://github.com/iwishiwasaneagle/jdrones/commit/b07881efdfbf467c46402b445b300c8d490df665)
- Increase dx magnitude [(915ec5f)](https://github.com/iwishiwasaneagle/jdrones/commit/915ec5f51c13455af293e9b68e459fc024014ecf)
- The reward is dependent on the target z as well, so we have to include that data in the observation to make the env fully observable [(3ca0bc3)](https://github.com/iwishiwasaneagle/jdrones/commit/3ca0bc3dca6385d1057167488e7fdd3c18887dc9)
- Include clip_range in CLI inputs [(4b2f8a9)](https://github.com/iwishiwasaneagle/jdrones/commit/4b2f8a94fe9f837216d77730c86c66da7bbd47f3)
- Reduce sub-env sim time to 0.1s [(0b6d02a)](https://github.com/iwishiwasaneagle/jdrones/commit/0b6d02a5b663d10940a059830c438fa51d3bae36)
- Increase dx magnitude [(eaa10a4)](https://github.com/iwishiwasaneagle/jdrones/commit/eaa10a499fa4086f5f4c53065b1c1ed416781f35)
- Include the next target in the observation [(fd287ce)](https://github.com/iwishiwasaneagle/jdrones/commit/fd287ce4912ca0a7d13d8fa62afa475bbd5634fc)
- Reduce sub-env sim time to 0.1s [(d9b97da)](https://github.com/iwishiwasaneagle/jdrones/commit/d9b97daf6de50113bcc866db79ea17fc67819965)
- Plot targets [(783177c)](https://github.com/iwishiwasaneagle/jdrones/commit/783177cbcd7c75aa63fea9e68eb1e118f6a6c6fe)
- Switch to a heading-velocity scheme and reintroduce the control action penalty [(e8a4fb7)](https://github.com/iwishiwasaneagle/jdrones/commit/e8a4fb7ef3b8767f88d9c4fd2dda7e3ccbaa11c6)
- Set squash_output to true when using SDE [(a265f65)](https://github.com/iwishiwasaneagle/jdrones/commit/a265f655d06fca47fef1cb045cb230dda68ff08b)
- Rename env to be more representative of what it is [(db08c1a)](https://github.com/iwishiwasaneagle/jdrones/commit/db08c1aea7b25e1f2df58e5863cfd8edba2840a8)
- Bump pydantic from 2.6.2 to 2.7.1 [(a10848d)](https://github.com/iwishiwasaneagle/jdrones/commit/a10848d2c768ae900f522091e5359dd86cd76a97)
- Pass total sim time to the mult DRL env [(6040eca)](https://github.com/iwishiwasaneagle/jdrones/commit/6040eca92e646ca5823e47641f43e1e15462f8dc)
- Bump pydantic from 2.7.1 to 2.8.2 [(2b8b402)](https://github.com/iwishiwasaneagle/jdrones/commit/2b8b402120ce73a428c9541877cdb6a0436a1ff2)
- Make reward calculations optional for polynomial envs [(6e38d55)](https://github.com/iwishiwasaneagle/jdrones/commit/6e38d55af7d64f224c8f3c80ebf3674c11fc4174)
- Silence pybullet messages at import [(1683157)](https://github.com/iwishiwasaneagle/jdrones/commit/1683157b40a2dfcfb52e23a8e546103c5b3cb535)

### Performance

- Improve State.normed and move into main repo [(1afdf85)](https://github.com/iwishiwasaneagle/jdrones/commit/1afdf858db027fb44f059111e2dad00a66fdcf11)
- Improve to_x and from_x by up to 2x speeds [(2669d4b)](https://github.com/iwishiwasaneagle/jdrones/commit/2669d4b8155bb97decf5c04472e9fcf80eb64858)
- Use numpy+numba in transformation calculations for better performance [(fe5b953)](https://github.com/iwishiwasaneagle/jdrones/commit/fe5b95382f7dff2a9cb428061e3e1d107bac0bd7)
- Use numba jit for core simulation methods [(f9b27ca)](https://github.com/iwishiwasaneagle/jdrones/commit/f9b27cab2a099f5f8ad22b351fb0185c48219342)
- Use faster way of checking if any item is nan [(eb81c10)](https://github.com/iwishiwasaneagle/jdrones/commit/eb81c107d4898b89dd28eaa9c2f4622a99a9154c)
- Use a faster method to zero clip than np.clip [(ea5d3bc)](https://github.com/iwishiwasaneagle/jdrones/commit/ea5d3bc66a62b80c898bc214545b05b47130aabd)

### Refactor

- Example got too big, so split into sub-modules [(3960e07)](https://github.com/iwishiwasaneagle/jdrones/commit/3960e075fe4c96632701b5658253451d6fc7f1f3)
- Rename to be more accurate [(1de2e16)](https://github.com/iwishiwasaneagle/jdrones/commit/1de2e169bfa719dd0d9c43f6015c5b031a1066aa)
- Remove old env and optuna sweeps [(14090b4)](https://github.com/iwishiwasaneagle/jdrones/commit/14090b4aa08de0834b1873c0a8e26272c34155ac)
- Handle rewards normalization in the top-level env [(8bfc7a7)](https://github.com/iwishiwasaneagle/jdrones/commit/8bfc7a7a85af20f1f571b1871d0eeadd0ba2fa1b)

## [0.7.2] - 2023-04-26

### Bug Fixes

- Proper type casting to np.float64 [(0b2b805)](https://github.com/iwishiwasaneagle/jdrones/commit/0b2b80546ebc26b2512b9c4e5d570d807931327c)
- Ignore __init__ for doctests due to circular import errors [(3cfb121)](https://github.com/iwishiwasaneagle/jdrones/commit/3cfb12149868565745f357113b9d0570f46be0a8)
- Inherit the action space from the base clase [(423ce55)](https://github.com/iwishiwasaneagle/jdrones/commit/423ce55979a24e77df10373bbf6210bf561ba539)
- Skip CI jobs/steps that interact with outside resources if triggered by bot [(6e5c357)](https://github.com/iwishiwasaneagle/jdrones/commit/6e5c357f1eddb7d1881403fe133f839464d6dd95)
- Set stack=True to ensure state observations are stacked [(edf45d5)](https://github.com/iwishiwasaneagle/jdrones/commit/edf45d59788763a60d301c973c570e27bd262bb9)

### Documentation

- Docstring for new DType and FloatLike types [(dd9551a)](https://github.com/iwishiwasaneagle/jdrones/commit/dd9551afa9cdccfd12830dfdf1d0848da7e865d6)
- Describe the reason behind the util functions for step [(25ffe00)](https://github.com/iwishiwasaneagle/jdrones/commit/25ffe00d1fc05f582535f31aff03495feddc4116)

### Miscellaneous Tasks

- Replace np.matrix with np.array [(5ab2a94)](https://github.com/iwishiwasaneagle/jdrones/commit/5ab2a9442530d0166880d548404c6824fcce5dc9)
- Bump loguru from 0.6.0 to 0.7.0 [(edfb2ca)](https://github.com/iwishiwasaneagle/jdrones/commit/edfb2ca4793cf8ebd13432daf924af6c759e3863)
- Bump gymnasium from 0.27.1 to 0.28.1 [(7068122)](https://github.com/iwishiwasaneagle/jdrones/commit/70681221e5078ab0a7768096ca057dda5a80c4c5)
- More sensible dependabot settings to prevent the PR spam [(29e781e)](https://github.com/iwishiwasaneagle/jdrones/commit/29e781e5fcd3e914287b90c300b40c2b13ad175e)
- Update changelog for v0.7.2 [skip pre-commit.ci] [(b50132b)](https://github.com/iwishiwasaneagle/jdrones/commit/b50132b4adb5348e693f317519f7250eeef27a18)

### Performance

- Optimize numpy state update code for non-linear drone model [(7de713f)](https://github.com/iwishiwasaneagle/jdrones/commit/7de713feff3f46e7f0563ff669097f498ae68b43)
- Use caching to pre-calculate time invariant params [(154f3af)](https://github.com/iwishiwasaneagle/jdrones/commit/154f3afb22f2d29a430c62a9bd345fc99db2c30f)
- Use caching to pre-calculate mixing matrix for drone plus [(a0a207f)](https://github.com/iwishiwasaneagle/jdrones/commit/a0a207f7b582ac0357f9ff919c7b6886e709cd8c)

### Refactor

- Stop using @property for action and observation spaces [(3b89837)](https://github.com/iwishiwasaneagle/jdrones/commit/3b89837facbd1a9795eee8571d4fb2b865356f1e)

### Testing

- More testing for trajectories [(b66d7ca)](https://github.com/iwishiwasaneagle/jdrones/commit/b66d7ca33720676fd9bda3d0ae8e72165e58fa04)
- Test URDFModel hashing capabilities [(ebfbcba)](https://github.com/iwishiwasaneagle/jdrones/commit/ebfbcbaa23dbd1f93b4e91b17107a001b4dc827f)
- Test the cached model data accessing method [(c4b92c5)](https://github.com/iwishiwasaneagle/jdrones/commit/c4b92c537a740e7d42d8f491d0e9a85da1982c09)

## [0.7.1] - 2023-04-05

### Miscellaneous Tasks

- Update changelog for v0.7.1 [skip pre-commit.ci] [(8926597)](https://github.com/iwishiwasaneagle/jdrones/commit/892659781deae73fb1cd938df48c68aecbe3704a)

## [0.7.0] - 2023-04-05

### Bug Fixes

- Position drones would crash if target position was the same as current position [(1e97511)](https://github.com/iwishiwasaneagle/jdrones/commit/1e97511ce477d863650bb3fcfa59a7a6e0f2eae1)
- White space in doctest for PID [(ce9ef68)](https://github.com/iwishiwasaneagle/jdrones/commit/ce9ef68d4ea64bcd6fc12d8fd5b6d9818a209a84)
- White space in doctest for PID [(09585ab)](https://github.com/iwishiwasaneagle/jdrones/commit/09585ab624ada6abe6fb88142707ff8eafab6533)
- Option to not show, to enable saving of figure [(8c004df)](https://github.com/iwishiwasaneagle/jdrones/commit/8c004df83fbce2069e09f0a15420ffbac5e345df)
- Reset was calling States(States(...)) causing invalid shaped arrays) [(f8c3233)](https://github.com/iwishiwasaneagle/jdrones/commit/f8c3233a85baddb1939d33c7ae9047064025671f)
- Use local version of jdrones. Installing it previously means that a cached version could be used during unit testing [(6a1edb3)](https://github.com/iwishiwasaneagle/jdrones/commit/6a1edb3ae0907d98031eed9dcc6b683711b5100b)
- Reset was calling States(States(...)) causing invalid shaped arrays) [(901cb08)](https://github.com/iwishiwasaneagle/jdrones/commit/901cb080bc4123f093a9d577e0fdcffe816c8a56)
- Use local version of jdrones. Installing it previously means that a cached version could be used during unit testing [(3343d30)](https://github.com/iwishiwasaneagle/jdrones/commit/3343d30fda9ffb863eb62c8c2f0fbeecea2963a3)
- Improve consistency between docstrings and code (#45) [(2e1a819)](https://github.com/iwishiwasaneagle/jdrones/commit/2e1a8194fbd9405ac5ec54d021f87af70f75cb70)

### Documentation

- Add gymnasium env names to README [(42361c1)](https://github.com/iwishiwasaneagle/jdrones/commit/42361c1ec0fdd18deba9c2f6e9ebfcd860a7227b)
- Add gymnasium env names to README [(3a1da91)](https://github.com/iwishiwasaneagle/jdrones/commit/3a1da91885d5f8e16bba000ecfe362e5b4417d80)
- Fifth order polynomial with look-ahead drone env [(fe55815)](https://github.com/iwishiwasaneagle/jdrones/commit/fe55815d29ebb32f3deb9fb50436bc3409cb1490)
- Add new gymnasium env names to README [(a94a720)](https://github.com/iwishiwasaneagle/jdrones/commit/a94a72045c3380e0cabad5a07aae26162b323320)
- Add new gymnasium env names to README [(dffdde7)](https://github.com/iwishiwasaneagle/jdrones/commit/dffdde7f31ed6a99f83f4ab9b501d806c9d66fa6)
- Fifth order polynomial with look-ahead drone env [(bc79fa6)](https://github.com/iwishiwasaneagle/jdrones/commit/bc79fa61e39b4b31dc8e0c22f36c083a5a33f184)

### Features

- Move state labels to an enum to ensure consistency across the codebase [(c4e5229)](https://github.com/iwishiwasaneagle/jdrones/commit/c4e5229f3f98264e6e3b5e3c1c241bcd0d2b0a0f)
- Plotting utility functions [(1da6998)](https://github.com/iwishiwasaneagle/jdrones/commit/1da699845553988e209ca3f451caff36e092a11a)
- Allow velocity as an input to the polynomial position envs [(13cc6de)](https://github.com/iwishiwasaneagle/jdrones/commit/13cc6de2a19621de3887578f468f964bf10e2852)
- Fifth order polynomial with look-ahead drone env [(5f9d8bd)](https://github.com/iwishiwasaneagle/jdrones/commit/5f9d8bdb1ff03e76484b2d9ebb6deefd79eebe8e)

### Miscellaneous Tasks

- Add new plotting module in notebook quick setup [(5cc8fc9)](https://github.com/iwishiwasaneagle/jdrones/commit/5cc8fc907fcaf08eb1df8abfbbac3ab127ac45b2)
- Use new plotting utility functions [(ec60216)](https://github.com/iwishiwasaneagle/jdrones/commit/ec60216988de9623348777f9874815346317175c)
- Add matplotlib and seaborn as a requirement [(72f99ae)](https://github.com/iwishiwasaneagle/jdrones/commit/72f99ae1903af9c04917dc58a3df16bdac0efec2)
- Bump pandas from 1.5.3 to 2.0.0 [(2f2f1dd)](https://github.com/iwishiwasaneagle/jdrones/commit/2f2f1ddb48d040bf0251194623f1781d2ea87243)
- Proper axis labels for 3D path plot [(f6c03aa)](https://github.com/iwishiwasaneagle/jdrones/commit/f6c03aa04de9f4f8eb1461fe5ca7dabfa2f2cc8f)
- Proper axis labels for standard plots [(585d2d3)](https://github.com/iwishiwasaneagle/jdrones/commit/585d2d3cc535e9a5b9394f16f79589b88525d4bf)
- Clean up imports [(b8ef4a5)](https://github.com/iwishiwasaneagle/jdrones/commit/b8ef4a5620a3d6eb9544f696835a259450e7f429)
- Clean up imports [(16b2b94)](https://github.com/iwishiwasaneagle/jdrones/commit/16b2b9422dbc4825cb590528bdce9b5317cade70)
- Update changelog for v0.7.0 [skip pre-commit.ci] [(f9b1512)](https://github.com/iwishiwasaneagle/jdrones/commit/f9b15122f325d62c99053fc606a11ca75910b80e)

### Refactor

- Change gymnasium env names + doctests [(74d37a6)](https://github.com/iwishiwasaneagle/jdrones/commit/74d37a6460e6a6b2f1e27860d46e1c94523bf0c9)
- Change gymnasium env names + doctests [(3e0d4e6)](https://github.com/iwishiwasaneagle/jdrones/commit/3e0d4e62000b9a3e88b800bed53da15cf26ce3c9)

### Testing

- Add proper doctests into files [(9e134dc)](https://github.com/iwishiwasaneagle/jdrones/commit/9e134dcc4ba488f1d8984fdcdd2f94792a886cc6)
- Add proper doctests into files [(063c55e)](https://github.com/iwishiwasaneagle/jdrones/commit/063c55ef35a80a9b4f4d05fe972dda36f1a1715a)
- Fifth order polynomial with look-ahead drone env [(3aa4e30)](https://github.com/iwishiwasaneagle/jdrones/commit/3aa4e306f0666db273bf79cd42d4d408175603d9)
- Fifth order polynomial with look-ahead drone env [(ccdf8eb)](https://github.com/iwishiwasaneagle/jdrones/commit/ccdf8ebc00708f0e8a5a11b1b60ef1dbf12c3528)

## [0.6.0] - 2023-03-16

### Bug Fixes

- Use \mathbf instead of \textbf to ensure symbols are being rendered correctly [(96e9376)](https://github.com/iwishiwasaneagle/jdrones/commit/96e9376bb0794232030e7104144444993602f60b)
- Python versions (was >3.8, now is >3.10) [(8f5e9f7)](https://github.com/iwishiwasaneagle/jdrones/commit/8f5e9f7cbe197665ab69fdb1c5471b4924dafc2c)
- Use short ref in docstr-cov CI [(ae6bf2a)](https://github.com/iwishiwasaneagle/jdrones/commit/ae6bf2a79fd39a12a230fa2383d73fe5b3859b06)
- Use python 3.10 to build the package in the CD [(320a88c)](https://github.com/iwishiwasaneagle/jdrones/commit/320a88c93977c3b86682322e6e7c8e9fea981722)
- --no-deps for wheels to only build jdrones and none of the deps [(7b8599d)](https://github.com/iwishiwasaneagle/jdrones/commit/7b8599d3608f30a96238099f43d539a7312059b3)

### Documentation

- Docstr for parent poly traj class [(3874400)](https://github.com/iwishiwasaneagle/jdrones/commit/387440027d81b4759fac87981d65198864666a6c)
- Docstring for Controller base class [(0c7c2c6)](https://github.com/iwishiwasaneagle/jdrones/commit/0c7c2c68fd55128a673af2ff191fc4c5c51f2701)
- Docstring for AngleController [(8daafa2)](https://github.com/iwishiwasaneagle/jdrones/commit/8daafa2c1589de109d4d422b9a37923ec5cfbb1a)
- Docstring for PID [(3a5fbd5)](https://github.com/iwishiwasaneagle/jdrones/commit/3a5fbd5ddb58df235c2adcdfd45cb9a56a744863)
- Docstring for LQR top-level [(fdaed6a)](https://github.com/iwishiwasaneagle/jdrones/commit/fdaed6acd39b92ce67161fc0d3e2bd2c2e7c7c3d)

### Features

- Extract integration test markers to seperate package (pytest-extra-markers) [(f0453b2)](https://github.com/iwishiwasaneagle/jdrones/commit/f0453b2df78618bc548c080ee4811821af776294)
- Stabilize the simple position drone by using a straight line trajectory [(35d3b70)](https://github.com/iwishiwasaneagle/jdrones/commit/35d3b702a68751661b1e237b96977375f6cd761c)
- Update check status for docstr-cov [(a3f275c)](https://github.com/iwishiwasaneagle/jdrones/commit/a3f275c626e03a4cb4f7a4bdee75c59688fa7553)

### Miscellaneous Tasks

- Update graphics [(d30bb66)](https://github.com/iwishiwasaneagle/jdrones/commit/d30bb66592f8ae9233eb7442dd78c332d2aafa85)
- Update changelog for v0.5.3 [skip pre-commit.ci] [(482652c)](https://github.com/iwishiwasaneagle/jdrones/commit/482652c22ab8b9416c118e6099148c5efaaf462d)
- Bump pandas from 1.3.4 to 1.5.3 [(9051ed2)](https://github.com/iwishiwasaneagle/jdrones/commit/9051ed2d932f4b360dbd7f6fdfe3ea550ff7fc85)
- Bump pydantic from 1.10.4 to 1.10.6 [(5b69bd5)](https://github.com/iwishiwasaneagle/jdrones/commit/5b69bd56832f561cb138758b7cca56a129b3346e)
- Update changelog for v0.6.0 [skip pre-commit.ci] [(a183231)](https://github.com/iwishiwasaneagle/jdrones/commit/a18323100ba220fc08b1c50bdb6a70e20a50876c)
- Delete changelog for redo of v0.6.0 [skip pre-commit.ci] [(b8c872d)](https://github.com/iwishiwasaneagle/jdrones/commit/b8c872d3144a9df557ea057b56b0c42230d9e3a6)
- Update changelog for v0.6.0 [skip pre-commit.ci] [(1b98622)](https://github.com/iwishiwasaneagle/jdrones/commit/1b9862282f87429984485b96f5d42212edaf5d40)
- Delete changelog for redo of v0.6.0 [skip pre-commit.ci] [(85b3de8)](https://github.com/iwishiwasaneagle/jdrones/commit/85b3de8e4fd98934f9b36438ca9b4f6dfc522215)
- Update changelog for v0.6.0 [skip pre-commit.ci] [(b714e69)](https://github.com/iwishiwasaneagle/jdrones/commit/b714e6981df048b0ec3793bb53c1d6a8962e7613)

### Testing

- Search for jdrones envs rather than manually specify them [(4198604)](https://github.com/iwishiwasaneagle/jdrones/commit/4198604ca70b4d79f776777c59e2d1c3502ef30b)
- Use syphar/restore-virtualenv@v1 to cache the python venv [(f57d4fc)](https://github.com/iwishiwasaneagle/jdrones/commit/f57d4fc4f9f3f5186c830907012845d037d1971a)
- Build the wheel for jdrones as part of the CI [(a457ba5)](https://github.com/iwishiwasaneagle/jdrones/commit/a457ba5ef66708ee029c252bda8d71785f8fbc4b)

## [0.5.2] - 2023-03-02

### Bug Fixes

- Rename job to match what's going on (caching all deps, not just PB) [(09c1b99)](https://github.com/iwishiwasaneagle/jdrones/commit/09c1b99135c8d0b8d38143be2f4365af2962f741)
- Set up git cliff to output proper markdown for prettier releases [(4a54e72)](https://github.com/iwishiwasaneagle/jdrones/commit/4a54e72f7aa6ee2e3861e43d2a01ffa222444afb)

### Miscellaneous Tasks

- Bump nptyping from 2.4.1 to 2.5.0 [(2d615da)](https://github.com/iwishiwasaneagle/jdrones/commit/2d615da17f9e2cd884f073406757d5b84b21bd45)
- Clean up old jpdmgen references [(2647eca)](https://github.com/iwishiwasaneagle/jdrones/commit/2647ecaedadc01e43a709c42d3a69f361c5fb03c)
- Update changelog for v0.5.2 [skip pre-commit.ci] [(1e446c9)](https://github.com/iwishiwasaneagle/jdrones/commit/1e446c9683d79be5d817644ec57b5e35a8c3f37a)

## [0.5.0] - 2023-03-02

### Bug Fixes

- Make velocity depend on the yaw error [(55e3e09)](https://github.com/iwishiwasaneagle/jdrones/commit/55e3e09fc2aebbfa462d11539db00e5a7a1eb1c4)
- Update states after step [(e768caf)](https://github.com/iwishiwasaneagle/jdrones/commit/e768caf14def4a536484bccbf6bdec6f2fe2b079)
- Small changes to simplify drone model [(6c540a5)](https://github.com/iwishiwasaneagle/jdrones/commit/6c540a55d69d4d537ae0c2148a6931ac5c0ce08b)
- General controller as return, rather than PID [(c7a0f3c)](https://github.com/iwishiwasaneagle/jdrones/commit/c7a0f3c03f57e8d3ef41608f6e1c201c81064fc8)
- Top-level collect errors [(9d8bd3f)](https://github.com/iwishiwasaneagle/jdrones/commit/9d8bd3f607468515c50da618c2ec652d568c7db9)
- Remove and add the appropriate envs [(19e835e)](https://github.com/iwishiwasaneagle/jdrones/commit/19e835e24e75c2c58b9731685af3453a9d0e37c7)
- Incorrect classmethod implementation [(df08a83)](https://github.com/iwishiwasaneagle/jdrones/commit/df08a83f59ca653501beed42d2be789d34c10fb5)
- Bug in PID code after updating to Controller parent class [(93a6a5a)](https://github.com/iwishiwasaneagle/jdrones/commit/93a6a5aa7b3cee6fc470ac8319ec7a7d811220e0)
- Add all in the dir. Controlling what gets published is done via VSC [(92db0a8)](https://github.com/iwishiwasaneagle/jdrones/commit/92db0a8155c56700a3107d401532e85ac1b9976e)
- Droneenv -> pbdroneenv as per conftest [(9b7d206)](https://github.com/iwishiwasaneagle/jdrones/commit/9b7d206efa6a3bccc22ad20958f267312bc423f8)
- Sign changes due to linear model direction changes in d394ad6e1778d2add26d0c792f706db06c0d8ccd [(72ab2cb)](https://github.com/iwishiwasaneagle/jdrones/commit/72ab2cba2f51a354b096caf0fc3ba01847ad472e)
- Top level warning at every import to warn about the different coordinate systems in use [(67d330a)](https://github.com/iwishiwasaneagle/jdrones/commit/67d330a7b232c861e606b7681ab1b0e026226f00)
- Other dependencies were causing huge increases in CI build time, so just cache them all for now [(efd7e00)](https://github.com/iwishiwasaneagle/jdrones/commit/efd7e00632d31d7e3491ec6327db47e0c58bdb23)
- Stupid dumb typo... [(1702be4)](https://github.com/iwishiwasaneagle/jdrones/commit/1702be459d21826a4d477f05e8a2f47f90bb43b8)
- Correctly define drone motors [(1860e97)](https://github.com/iwishiwasaneagle/jdrones/commit/1860e9738c79d26c278b086d9bad1adb65c12ad5)
- Simulation_name -> tag [(80e3fa8)](https://github.com/iwishiwasaneagle/jdrones/commit/80e3fa8e4ddda49cf671490b1d22d514ed00ae86)
- Add condition within PositionDroneEnv to truncate sim if any value is nan [(024387b)](https://github.com/iwishiwasaneagle/jdrones/commit/024387b02ae5f7ff78028a45a625df0d825d6ca8)
- Update docs to reflect changes in f56225d9 (previously forgot to do this) [(4ff1508)](https://github.com/iwishiwasaneagle/jdrones/commit/4ff1508b797d1b51eec7db9ab2763011a8b04642)
- Typo [(8af69f8)](https://github.com/iwishiwasaneagle/jdrones/commit/8af69f88b10d9c7f4e0d6844333396932f445461)
- Correction on the maths. This step is done elsewhere [(3d8a93c)](https://github.com/iwishiwasaneagle/jdrones/commit/3d8a93c8b41da4a74dbdc0333970c868b35bcbc9)
- Oversight from aa0e1e3 after refactor [(a011166)](https://github.com/iwishiwasaneagle/jdrones/commit/a0111668312e7831d1c4b378f09446b60710dbe5)

### Documentation

- Incude PositionDroneEnv [(f488f7b)](https://github.com/iwishiwasaneagle/jdrones/commit/f488f7bba9dc883a3eec65bbb762f52076f79852)
- Docstring for LQR solve [(1730bdf)](https://github.com/iwishiwasaneagle/jdrones/commit/1730bdf2dff6b8c5fece8412e2852aa000374670)
- Catagorise example notebooks [(9fd5531)](https://github.com/iwishiwasaneagle/jdrones/commit/9fd5531373fa82e7d52627b44a101827efb8f2f0)
- Include trajectory code in docs [(3057bd0)](https://github.com/iwishiwasaneagle/jdrones/commit/3057bd0d8e4fdfe7d1900ff09592b5ad6795b847)
- Full docs (incl. maths) for QuinticPolynomialTrajectory [(53b56c1)](https://github.com/iwishiwasaneagle/jdrones/commit/53b56c1aa6aa27aed0679f13a998ddf07d962d08)
- Explain cost function within get_reward() [(b175e1c)](https://github.com/iwishiwasaneagle/jdrones/commit/b175e1cf068dcaf7f77a1831d0761cc19e8709fe)
- Docs for BasePositionDroneEnv [(aa0e1e3)](https://github.com/iwishiwasaneagle/jdrones/commit/aa0e1e3a2dd30a63a416c94ff5b705a562fa73cb)
- Docs for PolyPositionDroneEnv [(95d28d0)](https://github.com/iwishiwasaneagle/jdrones/commit/95d28d0d521a10ee907b86d18dec565c18430d40)
- Docs for LQRPositionDroneEnv [(52ab54f)](https://github.com/iwishiwasaneagle/jdrones/commit/52ab54fb1b1e49acbcf727cba1ba237c04783dbd)
- Fix intersphinx links [(5e0a308)](https://github.com/iwishiwasaneagle/jdrones/commit/5e0a308766f5eb1eed30b10fd1bd9b382f964a8e)
- Update README.md to show how to run all types of tests [(dcb07b2)](https://github.com/iwishiwasaneagle/jdrones/commit/dcb07b293a73fbd335622b09efa11141a760761f)

### Feat

- GA-tuned LQR [(d35f8b2)](https://github.com/iwishiwasaneagle/jdrones/commit/d35f8b2766ee1ee336a3a11a6a43863ffa3dda63)

### Features

- Allow wrappers to be added to the sub-env [(647cc1f)](https://github.com/iwishiwasaneagle/jdrones/commit/647cc1f5c9d835493fa0c4c545221e5f68e67652)
- Implement nonlinear, linear, and nonlinear PB3 models [(d394ad6)](https://github.com/iwishiwasaneagle/jdrones/commit/d394ad6e1778d2add26d0c792f706db06c0d8ccd)
- Create helper functions for both rpm->rpyT and vise versa [(6aea072)](https://github.com/iwishiwasaneagle/jdrones/commit/6aea0723893a617bf77a88e7aa9ff6346980b522)
- LQR controlled drone env [(eb30f2d)](https://github.com/iwishiwasaneagle/jdrones/commit/eb30f2d4fc059a5a1cdc1a13a8a1eba973aa26cb)
- Helper functions and classes to make going from and to state logs easier using pandas [(8fd2b89)](https://github.com/iwishiwasaneagle/jdrones/commit/8fd2b897b7de90021f4d928892ad11a0a9673c9f)
- Lqrdroneenv fixture [(4e725fc)](https://github.com/iwishiwasaneagle/jdrones/commit/4e725fcd325e549137eb6631a8c09209e06edca3)
- Use scipy [(f8c96a1)](https://github.com/iwishiwasaneagle/jdrones/commit/f8c96a1a70d76e2e0d9c3e3e7ffa2ea36e718310)
- PositionDroneEnv [(11e812d)](https://github.com/iwishiwasaneagle/jdrones/commit/11e812df0c19a43de219b41da3ff044bbfd835e2)
- Visually validate models to step inputs [(6123dd2)](https://github.com/iwishiwasaneagle/jdrones/commit/6123dd2bdf7745a12c3072c4660a1f26ea4c6842)
- Position example [(8a3c694)](https://github.com/iwishiwasaneagle/jdrones/commit/8a3c694275ac6ce06f99250241f94fab6e6e9fdf)
- Import script to make the imports less clunky across scripts [(4000218)](https://github.com/iwishiwasaneagle/jdrones/commit/40002188b3ea3bc28fa7d0fbe3328ba9940cfb96)
- Positiondroneenv fixture [(345c5e8)](https://github.com/iwishiwasaneagle/jdrones/commit/345c5e88df9d9f1510fb73b754d266f4458f3ed0)
- MPC Drone Example [(80f4458)](https://github.com/iwishiwasaneagle/jdrones/commit/80f4458cdaedcf9562fbf5aca1c0e4e37c386b3b)
- Add pandas [(450c653)](https://github.com/iwishiwasaneagle/jdrones/commit/450c653913aa9fc413c58e4bc704dde3a5947c7b)
- Quaternion multiplication [(083e53f)](https://github.com/iwishiwasaneagle/jdrones/commit/083e53fc04a9dc270df96f108c8615e3f50e9ba3)
- Rotate state by quat [(8aeba67)](https://github.com/iwishiwasaneagle/jdrones/commit/8aeba67aaea7d97e4ce60403daf2331e2c639deb)
- Ensure all models use RHR coordinate system, consistent with URDF and other sim packages [(2e90952)](https://github.com/iwishiwasaneagle/jdrones/commit/2e90952e0460d0e4d797426d185d88be422a6e94)
- Polynomial trajectory drone env [(7ef1768)](https://github.com/iwishiwasaneagle/jdrones/commit/7ef17680231542d3515b6ac2d09b6d423af1b05c)
- Stress test the position environments to ensure they don't crash over time [(513dcf2)](https://github.com/iwishiwasaneagle/jdrones/commit/513dcf2c75b61a2cffd88087da00a5d394d19518)

### Miscellaneous Tasks

- Clean up application of forces and torques to use body [(d6b2ed4)](https://github.com/iwishiwasaneagle/jdrones/commit/d6b2ed41e550e81e5d8888ddb22e51c98220a07c)
- Commit before deleting, to save the current state of the AttitudeAltitudeDroneEnv [(54c99a9)](https://github.com/iwishiwasaneagle/jdrones/commit/54c99a9aed4b2a70a4f10e6ffa08d53f7f7676a5)
- Commit before deleting, to save the current state of the VelHeadAltDroneEnv [(f312e9d)](https://github.com/iwishiwasaneagle/jdrones/commit/f312e9dfd40801f649236d4f33b4e594d84dc77d)
- Commit before deleting, to save the current state of the trajectory envs [(fa55246)](https://github.com/iwishiwasaneagle/jdrones/commit/fa552469456e945a1a9fc2c8e3fc74c52a257472)
- Update Q and R matrix gains from tuning via GA [(5a0c71c)](https://github.com/iwishiwasaneagle/jdrones/commit/5a0c71caf36fbc2fed249f6a137726183e4113ec)
- Fix API [(be1e6c2)](https://github.com/iwishiwasaneagle/jdrones/commit/be1e6c217a60a0d455c2bae1651f171fa3a052f6)
- Add more test cases to transform tests [(2e850c3)](https://github.com/iwishiwasaneagle/jdrones/commit/2e850c34ecebd3845f32d46fec65325e23e748b5)
- Commit before deletion for archival purposes [(1b2c99c)](https://github.com/iwishiwasaneagle/jdrones/commit/1b2c99c4519cf75f8eaad1ab214669e2cb4de754)
- Specify action space [(497e74f)](https://github.com/iwishiwasaneagle/jdrones/commit/497e74fe85213e251a9193c1e75d027b4ea68a9a)
- Implement reset in LQR controller and be more verbose about shape of error [(eb828e1)](https://github.com/iwishiwasaneagle/jdrones/commit/eb828e16834753cc2dea79b6741c582146414c13)
- Cleanup imports [(e416f5b)](https://github.com/iwishiwasaneagle/jdrones/commit/e416f5b377fd03d40e5a72b834f83c149ac97a36)
- Version getting [(664d4aa)](https://github.com/iwishiwasaneagle/jdrones/commit/664d4aae2b2235ac308c2b5240498e521660d549)
- Remove [(c1ac6cf)](https://github.com/iwishiwasaneagle/jdrones/commit/c1ac6cf9072d52be6863201e6326a4136118f4e9)
- Update Q and R matrix gains from tuning via GA [(e59f514)](https://github.com/iwishiwasaneagle/jdrones/commit/e59f514a248afa784f0692958bb9706eba2da35e)
- Update graphics [(d8b8ac8)](https://github.com/iwishiwasaneagle/jdrones/commit/d8b8ac8b60a7b7c24169f29d55de9cd4e79bc1d5)
- Still broken, but at least it's been unified... [(adc136a)](https://github.com/iwishiwasaneagle/jdrones/commit/adc136aac0319eb6ce9a45f0719eb855928a10d0)
- Mark integration tests [(c54f3fe)](https://github.com/iwishiwasaneagle/jdrones/commit/c54f3fed1ae9a120f28f55f121c12c997cd1f8f9)
- Rename CI step to something more descriptive [(475303d)](https://github.com/iwishiwasaneagle/jdrones/commit/475303de23281efe4714b4cdcad3fb32fdafc19b)
- Update Q and R matrix gains from tuning via GA [(6d14ee9)](https://github.com/iwishiwasaneagle/jdrones/commit/6d14ee95cdd6b7297163982364a587c3681d1c79)
- Update graphics [(e78074f)](https://github.com/iwishiwasaneagle/jdrones/commit/e78074fe9869197cbe1757f51aa9e2758f5f3a6c)
- Add automatic linearisation of NL model to examples [(e365fb1)](https://github.com/iwishiwasaneagle/jdrones/commit/e365fb1e7ad3446cf7f91c3db4e317d69eaadbe1)
- Mock pandas [(4b2399c)](https://github.com/iwishiwasaneagle/jdrones/commit/4b2399c93b1d0fd8777c48a4f04b95cebba3a63a)
- Document BaseControlledEnv [(f4b039e)](https://github.com/iwishiwasaneagle/jdrones/commit/f4b039e028bc43f69f82b4d12fada85b4a62fc98)
- Explicitly state the stratgies to prevent duplicates and only run the stress tests once [(e976499)](https://github.com/iwishiwasaneagle/jdrones/commit/e9764992be829fb20ee3c7166ea0b03402903a71)
- Update changelog for v0.5.0 [skip pre-commit.ci] [(3a7f9a7)](https://github.com/iwishiwasaneagle/jdrones/commit/3a7f9a778778869f6b368000581c11da2b8d7856)

### Refactor

- Remove AttitudeAltitudeDroneEnv to be replaced by LQR controller [(6df31dd)](https://github.com/iwishiwasaneagle/jdrones/commit/6df31ddb984b6491f71232de6b6676635844ebd4)
- Remove VelHeadAltDroneEnv to be replaced by LQR controller [(9de143a)](https://github.com/iwishiwasaneagle/jdrones/commit/9de143a64e627d83b00bd34e2ed603be0da3216b)
- Remove trajectory envs to be replaced by LQR controller [(1807e57)](https://github.com/iwishiwasaneagle/jdrones/commit/1807e57868fa82536a7a8e093644882351cd9f20)
- Reflect envs refactory changes [(07e91ad)](https://github.com/iwishiwasaneagle/jdrones/commit/07e91ad753d1d6ac4eb75a47f8826f3a7a6aff4d)
- Merge into BaseDroneEnv [(d3df134)](https://github.com/iwishiwasaneagle/jdrones/commit/d3df134604d4034cc6328f60bba3ffe65dcee8c5)
- Switch to creating matrices via staticmethod to enable easier access without creating the class [(4b972da)](https://github.com/iwishiwasaneagle/jdrones/commit/4b972daf93152f69846304a26a57ecb0cd4cdff7)
- Move to/from x tests as per refactor [(387cf9f)](https://github.com/iwishiwasaneagle/jdrones/commit/387cf9ffb385de65458ebd35c150a35fa0c1f2a3)
- Delete [(1f4fd4f)](https://github.com/iwishiwasaneagle/jdrones/commit/1f4fd4fbb39a08b8c30f8e2a33cd6c8e0d3110fc)
- Use the new PositionDroneEnv [(3149d74)](https://github.com/iwishiwasaneagle/jdrones/commit/3149d74d63f1e9af5fc2f677074f710cbcfe8e38)
- Split true types and data models into seperate files [(f56225d)](https://github.com/iwishiwasaneagle/jdrones/commit/f56225d9ea1d646418bcbf004b3822ea02562245)

### Testing

- Unify tests for NL and L models to ensure compatibility when designing controllers [(606317d)](https://github.com/iwishiwasaneagle/jdrones/commit/606317d349f56c49c1f9eed9dd39d09387d9047e)
- Correct input to rotation [(820540d)](https://github.com/iwishiwasaneagle/jdrones/commit/820540dce216f25dcb2d20ca576e6586361418e1)
- Mark pybullet env's test_vel_from_rot as skipped for now [(1b30792)](https://github.com/iwishiwasaneagle/jdrones/commit/1b30792a508f795611cd7f04e8e7aeb51b039bc8)
- Tested State.apply_quat [(683a12e)](https://github.com/iwishiwasaneagle/jdrones/commit/683a12ecafc8a6206a6acf698f1dcf1c513f6572)
- Ensure assets are consistent between lib and tests [(581d0d5)](https://github.com/iwishiwasaneagle/jdrones/commit/581d0d5156cca059e79e23b781f7cfed5f5e8861)
- Ensure states are float as per 683a12ec [(ab208fd)](https://github.com/iwishiwasaneagle/jdrones/commit/ab208fdac37b3f7ef113b3b84e15307d94e4c290)
- Quat mul via hypothesis using scipy.spatial.transform.Rotation [(5cfcd55)](https://github.com/iwishiwasaneagle/jdrones/commit/5cfcd551cdcf606da43457b099552c60c691e3ff)
- Add pytest-xdist to allow threaded test execution [(92c480b)](https://github.com/iwishiwasaneagle/jdrones/commit/92c480b9d873e58a67e1abc151cfa578a4207c7f)

## [0.4.0] - 2023-02-09

### Bug Fixes

- Getting versions via setuptools_scm.get_version in docs [(9355812)](https://github.com/iwishiwasaneagle/jdrones/commit/93558120b3de774f16cfa618752cc2e1285158f8)
- Import errors when getting version for docs [(0366a1f)](https://github.com/iwishiwasaneagle/jdrones/commit/0366a1f782f987c2c9ffead3acc81f1ca7b67c95)

### Documentation

- PID trajectory env documentation to explain the control logic [(f9568f0)](https://github.com/iwishiwasaneagle/jdrones/commit/f9568f09250fc4ed3e027362939cbb07559ab127)
- Improve envs page readability through headings which now show up in toctree [(cedc2cc)](https://github.com/iwishiwasaneagle/jdrones/commit/cedc2cc379ce62b41f0f4bfe83d7af42f75ea343)

### Features

- Add GPL3 headers to all project files [(737214c)](https://github.com/iwishiwasaneagle/jdrones/commit/737214c61f2ee894e2109e9a36032941062253a4)

### Miscellaneous Tasks

- Bump numpy from 1.24.1 to 1.24.2 [(55af72e)](https://github.com/iwishiwasaneagle/jdrones/commit/55af72e03ab7d1eb0de8276b607d94dba407ce6c)
- MathJax configurations [(53fef7e)](https://github.com/iwishiwasaneagle/jdrones/commit/53fef7e0d971034f0dc7466405cd8179957f9652)
- Add warning about this being in alpha [(dbcb855)](https://github.com/iwishiwasaneagle/jdrones/commit/dbcb855046442ff874e836cc36a4c2cc47b036f5)
- Update changelog for v0.4.0 [skip pre-commit.ci] [(4e2afb6)](https://github.com/iwishiwasaneagle/jdrones/commit/4e2afb6d7af3332207f9b77894f96a7050cade8a)

## [0.3.1] - 2023-02-03

### Bug Fixes

- Install pandoc in CD pipeline [(8ede886)](https://github.com/iwishiwasaneagle/jdrones/commit/8ede88665934c84b9854f2a9af9a9bb48be49074)

### Miscellaneous Tasks

- Update changelog for v0.3.1 [skip pre-commit.ci] [(98c3597)](https://github.com/iwishiwasaneagle/jdrones/commit/98c35970785ad5942496eec865da5aba022d1de9)

## [0.3.0] - 2023-02-03

### Bug Fixes

- Install pandoc which is required by sphinx [(433b146)](https://github.com/iwishiwasaneagle/jdrones/commit/433b1463907f526bbee2c35b872338ee26770903)
- Sudo for apt-get commands [(1ea68ea)](https://github.com/iwishiwasaneagle/jdrones/commit/1ea68ea0bb24ef6b9301874a777556d33dcbd49a)

### Features

- Examples [(8baabd4)](https://github.com/iwishiwasaneagle/jdrones/commit/8baabd41d11a2f2e997299b1db5ad19da3cd682f)

### Miscellaneous Tasks

- Fix building of docs [(4b6fffc)](https://github.com/iwishiwasaneagle/jdrones/commit/4b6fffc6358508d7a05e3bb7f29747a5d7535a8b)
- Mock nbtyping for docs [(9f9f88f)](https://github.com/iwishiwasaneagle/jdrones/commit/9f9f88f8ccbae2c965b502ddeff3f74f10d6c1e7)
- Check for empty properly [(4a6e034)](https://github.com/iwishiwasaneagle/jdrones/commit/4a6e034e307b3bac057715fc92dc3c3d568cf781)
- Update changelog for v0.3.0 [skip pre-commit.ci] [(40cfe19)](https://github.com/iwishiwasaneagle/jdrones/commit/40cfe197437323d09def72258aa7cad500489f8d)

## [0.2.0] - 2023-02-02

### Bug Fixes

- Unused variable in the URDFModel from refactoring [(95ae8b1)](https://github.com/iwishiwasaneagle/jdrones/commit/95ae8b184ea1c5779dae65b7b06e3164c2129328)

### Features

- Move yaw and rpy transform calculations into maths module [(99e6532)](https://github.com/iwishiwasaneagle/jdrones/commit/99e65324a517e07a9feff84097b432bb6f2f1950)
- Move poly traj into own module and fully test [(a727658)](https://github.com/iwishiwasaneagle/jdrones/commit/a727658c41fc53ddd20d58b22eb60d868efcf799)
- Use nptyping library to define shapes of data [(d824d27)](https://github.com/iwishiwasaneagle/jdrones/commit/d824d271a8314a3930cad0c85fafaeb0dea44607)
- Trajectory env refactored [(2aeeb96)](https://github.com/iwishiwasaneagle/jdrones/commit/2aeeb96af850c67a535d3dc264ac37d73d083883)
- Cache pybullet by installing in empty workflow [(15e4a88)](https://github.com/iwishiwasaneagle/jdrones/commit/15e4a888fc26856304c582cdd71bd77484438820)
- Properly handle asset paths [(583c600)](https://github.com/iwishiwasaneagle/jdrones/commit/583c6006dbf07b5f1c6b01b82ba6a4bf6949f92c)

### Miscellaneous Tasks

- Use more appropriate clip_scalar [(e288313)](https://github.com/iwishiwasaneagle/jdrones/commit/e288313a97f4818f7644d12cc02bfe5326a2b396)
- Deal with refactoring of drone envs [(05fe4c5)](https://github.com/iwishiwasaneagle/jdrones/commit/05fe4c5db17d3e626c4268efe83572d4a5eabdca)
- Give ground plane collision data in info dict [(417de31)](https://github.com/iwishiwasaneagle/jdrones/commit/417de315f390125fe163231629594e022538efa9)
- Deal with the various refactors [(0455adb)](https://github.com/iwishiwasaneagle/jdrones/commit/0455adb6254d79fef835a87562d4984494d51c78)
- Deal with the various refactors [(76d4715)](https://github.com/iwishiwasaneagle/jdrones/commit/76d47153ae65c6d96a9271e14f125f1d461921eb)
- Add nptyping to requirements [(c0a45f6)](https://github.com/iwishiwasaneagle/jdrones/commit/c0a45f62bfa0b7c34a2e72a9d0e4724a84abad58)
- Export DronePlus from envs [(7860f19)](https://github.com/iwishiwasaneagle/jdrones/commit/7860f19cd6a1a85fb6fb37f94ed62d93fedd4b82)
- Update changelog for v0.2.0 [skip pre-commit.ci] [(1a4ccd8)](https://github.com/iwishiwasaneagle/jdrones/commit/1a4ccd82bcd5854b3623c5a9da39b81ffabdf152)

### Refactor

- Remove custom action types and remodel them as types [(025ca4e)](https://github.com/iwishiwasaneagle/jdrones/commit/025ca4e7bef6a8099d9e1e34f9e3db88eb90a896)
- Alter way the info dict is manipulated [(a17a9a4)](https://github.com/iwishiwasaneagle/jdrones/commit/a17a9a4adb3c2164658f75cdf9907e0bec6e5413)
- Merge postion alt drone env into trajectory control commands [(4f931bd)](https://github.com/iwishiwasaneagle/jdrones/commit/4f931bdfa6e33fdcf1a0358e84b251d3ec76a44b)
- Move into own module [(1244c96)](https://github.com/iwishiwasaneagle/jdrones/commit/1244c967e0c8c624d3b9e3f4789a58fab5ceb2fd)
- Use apply_rpy from maths module [(b7182dc)](https://github.com/iwishiwasaneagle/jdrones/commit/b7182dca4ae1ccd1c49f56d1b0857a1d5dd40ba0)
- Merged with trajectory env API [(a0c30c3)](https://github.com/iwishiwasaneagle/jdrones/commit/a0c30c3dbd62e2f840ebb443c358d4f27c754e93)

### Testing

- Fully test PID controllers [(94862d4)](https://github.com/iwishiwasaneagle/jdrones/commit/94862d4d8247dbcf4b13290cfc6a511275126cad)

## [0.1.2] - 2023-01-31

### Bug Fixes

- Use different URLs for files that are hosted on gh-pages [(d84ccb3)](https://github.com/iwishiwasaneagle/jdrones/commit/d84ccb37136503aa81f8c1b8ec446dd29b61e6c0)
- Docstr-cov was failing whilst trying to get baseline [(49d2077)](https://github.com/iwishiwasaneagle/jdrones/commit/49d207739829f082ba04ad548004884753f0f2fc)
- Hierarchy error in titles [(21ea9b7)](https://github.com/iwishiwasaneagle/jdrones/commit/21ea9b7546144286fdbc361bf5be74535ebcc350)

### Documentation

- Add controllers to docs [(856c336)](https://github.com/iwishiwasaneagle/jdrones/commit/856c336ab2a64cc35594d2041ac8114ab0577654)
- Improve index.rst to act as the landing page for the hosted docs [(1091aed)](https://github.com/iwishiwasaneagle/jdrones/commit/1091aed4ea211d706283a5affa9643670617a9e5)

### Miscellaneous Tasks

- Mock pybullet_data [(d858099)](https://github.com/iwishiwasaneagle/jdrones/commit/d858099acad558b9fe0665b587fa062b6ac07ef9)
- Remove -q from checkout [(f5493c6)](https://github.com/iwishiwasaneagle/jdrones/commit/f5493c671f64fd533d1c04e85c68443de829af79)
- Update changelog for v0.1.2 [skip pre-commit.ci] [(0495c56)](https://github.com/iwishiwasaneagle/jdrones/commit/0495c5663b11d65370a22533b63f43361d7eba51)

### Refactor

- Speed up docstr build by only requireing docs/requirements.txt [(6d6cb64)](https://github.com/iwishiwasaneagle/jdrones/commit/6d6cb6401a33edb7ce3fa1eac218750f0e50e9ef)
- Move PyBulletIds to types module [(b3ad632)](https://github.com/iwishiwasaneagle/jdrones/commit/b3ad632744045e7f411f6fcf1e10f2d660b8712e)
- Remove license from docs [(e4a551d)](https://github.com/iwishiwasaneagle/jdrones/commit/e4a551dbe4b1ce761885c09058c3b7eb9d370a05)

## [0.1.1] - 2023-01-31

### Bug Fixes

- Running very slow because of np.asarray [(6e19344)](https://github.com/iwishiwasaneagle/jdrones/commit/6e193445972e9aa307bb648cf01d512ffbf3fa16)
- Reused function name [(2690f4e)](https://github.com/iwishiwasaneagle/jdrones/commit/2690f4ebc6f16445502014dcf6438ed3acccbdb8)
- Ignore license.rst as this is pretty much immutable [(4ea47ca)](https://github.com/iwishiwasaneagle/jdrones/commit/4ea47ca3c09d105832e1bb23effae80fead3c226)
- Use pyproject.toml for rstcheck config [(3985349)](https://github.com/iwishiwasaneagle/jdrones/commit/3985349b59fe73de1a3e1de90e55bf05e6261d26)
- If there's no coverage, set to 0 [(6949c23)](https://github.com/iwishiwasaneagle/jdrones/commit/6949c230201bf4b12eb6dc88ac0ef589be7b7a3a)
- Allow --accept-empty in case there's no python files [(1714bbf)](https://github.com/iwishiwasaneagle/jdrones/commit/1714bbfcbb494225aa0b51b6b3beb353eec4861e)

### Documentation

- Start of BaseDroneEnv docs [(7b9553f)](https://github.com/iwishiwasaneagle/jdrones/commit/7b9553fe5d68e92fc78e4c9b594c4cac55101db8)
- Do most of the types docs [(c3e3a18)](https://github.com/iwishiwasaneagle/jdrones/commit/c3e3a18e178a60e5f5ee8c85ed141d305b48e2c3)
- More docs for BaseDroneEnv [(a17a30c)](https://github.com/iwishiwasaneagle/jdrones/commit/a17a30cc6230177180e2f2eafcc576f74c9c1169)
- Include BaseDroneEnv in generated docs [(de0f058)](https://github.com/iwishiwasaneagle/jdrones/commit/de0f0585d1f734483369e133056703d41fda9812)

### Features

- Check docstr coverage isn't being reduced [(a4be21b)](https://github.com/iwishiwasaneagle/jdrones/commit/a4be21b1a618060c360cdff4c0f32b828117f1e2)

### Miscellaneous Tasks

- Improve readme [(f909511)](https://github.com/iwishiwasaneagle/jdrones/commit/f9095111c7fbc999cab9a9a45d2f8d1d3ae4ac4f)
- Add future work on motor modelling [(9307e91)](https://github.com/iwishiwasaneagle/jdrones/commit/9307e91da249088486242203515f0ff98a985c58)
- Mute git checkouts in docstr-cov CI [(9721eb9)](https://github.com/iwishiwasaneagle/jdrones/commit/9721eb9116cca9b4bb33f97a3428768292861812)
- Fix flake errors (ambiguous variable, unused imports) [(1f2b750)](https://github.com/iwishiwasaneagle/jdrones/commit/1f2b750848dd948d5bdd97bf785750efc76b78f5)
- Add intersphinx mapping for gymnasium [(c0316f8)](https://github.com/iwishiwasaneagle/jdrones/commit/c0316f8ac8fada3d1022b7e4da9474469b216d64)
- Don't upload badge as artifact. Can't currently use it [(6d9817b)](https://github.com/iwishiwasaneagle/jdrones/commit/6d9817b7cbea934cfb9c10a25de86693cbf35ccb)
- Upload docstr-cov badge to gh-pages [(d7e923a)](https://github.com/iwishiwasaneagle/jdrones/commit/d7e923aa0b44b813db5d4bb8b947f29a9d7a3e4a)
- Update changelog for v0.1.1 [skip pre-commit.ci] [(066c6c3)](https://github.com/iwishiwasaneagle/jdrones/commit/066c6c3a9b3a432ae86bfb8c695314d55f9f80a2)

## [0.1.0] - 2023-01-27

### Bug Fixes

- Updates to setup since there's no CPP [(6e489fb)](https://github.com/iwishiwasaneagle/jdrones/commit/6e489fb22f5aa2edf54887734eae587d649b61bc)
- Use pip wheel rather than cibuildwheel as it's currently a pure python package [(94a0837)](https://github.com/iwishiwasaneagle/jdrones/commit/94a0837ee37f26784de4231afd42b5258e44f09a)
- No need for many and musl as it's pure python for now [(3e57819)](https://github.com/iwishiwasaneagle/jdrones/commit/3e57819690da9d1468f14666b7ff663d5dbd06fb)
- Move to only python >=3.10 [(87570d9)](https://github.com/iwishiwasaneagle/jdrones/commit/87570d989c2609620579e67a1546c18c85d96c58)
- Space [(a870dfe)](https://github.com/iwishiwasaneagle/jdrones/commit/a870dfe5564cca971943ea753c5ae3071246aacf)
- Spaces [(808a9cd)](https://github.com/iwishiwasaneagle/jdrones/commit/808a9cdd8b1093732e5cd2147ff1902ff422640c)
- Fix issues with runners not correctly initializing due to misconfigured matrix [(5c88fca)](https://github.com/iwishiwasaneagle/jdrones/commit/5c88fca8d21d87849e53567db166c341fc027b90)
- Set correct permissions for gh pages and releases [(8aabdfd)](https://github.com/iwishiwasaneagle/jdrones/commit/8aabdfd9f0fc5faead8cfa2a1e3f3c5245a63f1e)
- Install docs/requiremnts.txt rather than tests/ [(961f940)](https://github.com/iwishiwasaneagle/jdrones/commit/961f940aa2766a6d35912ead715dc73241f5b91b)
- Revert permissions, as this is done through settings console [(07b9198)](https://github.com/iwishiwasaneagle/jdrones/commit/07b9198b677288b2a15ef73e1c315fb09af12113)

### Features

- Initial commit [(2b2928c)](https://github.com/iwishiwasaneagle/jdrones/commit/2b2928c2901d69e5954e0558ba9e2cd204b2a48c)
- Move back to pip as conda wasn't required [(071ae4b)](https://github.com/iwishiwasaneagle/jdrones/commit/071ae4b8f5ecb21e82fa4263611e0012c520271a)

### Miscellaneous Tasks

- Update changelog for v0.2.0 [skip pre-commit.ci] [(e4b25d4)](https://github.com/iwishiwasaneagle/jdrones/commit/e4b25d479669a2fa7301dd51f1090d5e378655c2)
- Update changelog for v0.1.0 [skip pre-commit.ci] [(9b40238)](https://github.com/iwishiwasaneagle/jdrones/commit/9b402385ae225dca824639154dbdb241394c1592)
- Update how conda is used in CI [(d9704d8)](https://github.com/iwishiwasaneagle/jdrones/commit/d9704d87a6442ddbc9820d6118a3545d049a3e68)
- A little import cleanup [(6cb83e9)](https://github.com/iwishiwasaneagle/jdrones/commit/6cb83e99ef333fef13b7d0e2296747652d470915)
- Update CI to also run integration tests but standalone [(aa867ce)](https://github.com/iwishiwasaneagle/jdrones/commit/aa867ce226ec9eae434881b1b0a050bb832cb484)
- Document transform funcs to show what's happening behind the scenes in PB3 [(645ce50)](https://github.com/iwishiwasaneagle/jdrones/commit/645ce50af91cb0de18737119d1f9136cf6391c80)
- Update changelog for v0.1.0 [skip pre-commit.ci] [(79d9f01)](https://github.com/iwishiwasaneagle/jdrones/commit/79d9f01083530d5b87e2d4984665ef6e77d4d8ec)
- Delete changelog for redo of v0.1.0 [skip pre-commit.ci] [(45f3fb3)](https://github.com/iwishiwasaneagle/jdrones/commit/45f3fb3e95c8062fe538e9fe10d8eb0e9602fa11)
- Update changelog for v0.1.0 [skip pre-commit.ci] [(d3dc768)](https://github.com/iwishiwasaneagle/jdrones/commit/d3dc7686ca804b005aa050f14521381a74cae501)

### Refactor

- Docs filenames and symlink names [(64aba9a)](https://github.com/iwishiwasaneagle/jdrones/commit/64aba9a8b773d72df807f4177ca34b068e4de8c0)

### Testing

- Skip integration tests [(2651c8b)](https://github.com/iwishiwasaneagle/jdrones/commit/2651c8be53738c47f3c79c7622f1ab0c15621f4b)
- Fix tests since quats and euler have multiple correct variations [(5a3a96d)](https://github.com/iwishiwasaneagle/jdrones/commit/5a3a96daf1b9374d605fba3f4dae620973b88c8e)

