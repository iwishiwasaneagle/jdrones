# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2023-02-09

### Bug Fixes

- Getting versions via setuptools_scm.get_version in docs [(9355812)](https://github.com/iwishiwasaneagle/jpdmgen/commit/93558120b3de774f16cfa618752cc2e1285158f8)
- Import errors when getting version for docs [(0366a1f)](https://github.com/iwishiwasaneagle/jpdmgen/commit/0366a1f782f987c2c9ffead3acc81f1ca7b67c95)

### Documentation

- PID trajectory env documentation to explain the control logic [(f9568f0)](https://github.com/iwishiwasaneagle/jpdmgen/commit/f9568f09250fc4ed3e027362939cbb07559ab127)
- Improve envs page readability through headings which now show up in toctree [(cedc2cc)](https://github.com/iwishiwasaneagle/jpdmgen/commit/cedc2cc379ce62b41f0f4bfe83d7af42f75ea343)

### Features

- Add GPL3 headers to all project files [(737214c)](https://github.com/iwishiwasaneagle/jpdmgen/commit/737214c61f2ee894e2109e9a36032941062253a4)

### Miscellaneous Tasks

- Bump numpy from 1.24.1 to 1.24.2 [(55af72e)](https://github.com/iwishiwasaneagle/jpdmgen/commit/55af72e03ab7d1eb0de8276b607d94dba407ce6c)
- MathJax configurations [(53fef7e)](https://github.com/iwishiwasaneagle/jpdmgen/commit/53fef7e0d971034f0dc7466405cd8179957f9652)
- Add warning about this being in alpha [(dbcb855)](https://github.com/iwishiwasaneagle/jpdmgen/commit/dbcb855046442ff874e836cc36a4c2cc47b036f5)

## [0.3.1] - 2023-02-03

### Bug Fixes

- Install pandoc in CD pipeline [(8ede886)](https://github.com/iwishiwasaneagle/jpdmgen/commit/8ede88665934c84b9854f2a9af9a9bb48be49074)

### Miscellaneous Tasks

- Update changelog for v0.3.1 [skip pre-commit.ci] [(98c3597)](https://github.com/iwishiwasaneagle/jpdmgen/commit/98c35970785ad5942496eec865da5aba022d1de9)

## [0.3.0] - 2023-02-03

### Bug Fixes

- Install pandoc which is required by sphinx [(433b146)](https://github.com/iwishiwasaneagle/jpdmgen/commit/433b1463907f526bbee2c35b872338ee26770903)
- Sudo for apt-get commands [(1ea68ea)](https://github.com/iwishiwasaneagle/jpdmgen/commit/1ea68ea0bb24ef6b9301874a777556d33dcbd49a)

### Features

- Examples [(8baabd4)](https://github.com/iwishiwasaneagle/jpdmgen/commit/8baabd41d11a2f2e997299b1db5ad19da3cd682f)

### Miscellaneous Tasks

- Fix building of docs [(4b6fffc)](https://github.com/iwishiwasaneagle/jpdmgen/commit/4b6fffc6358508d7a05e3bb7f29747a5d7535a8b)
- Mock nbtyping for docs [(9f9f88f)](https://github.com/iwishiwasaneagle/jpdmgen/commit/9f9f88f8ccbae2c965b502ddeff3f74f10d6c1e7)
- Check for empty properly [(4a6e034)](https://github.com/iwishiwasaneagle/jpdmgen/commit/4a6e034e307b3bac057715fc92dc3c3d568cf781)
- Update changelog for v0.3.0 [skip pre-commit.ci] [(40cfe19)](https://github.com/iwishiwasaneagle/jpdmgen/commit/40cfe197437323d09def72258aa7cad500489f8d)

## [0.2.0] - 2023-02-02

### Bug Fixes

- Unused variable in the URDFModel from refactoring [(95ae8b1)](https://github.com/iwishiwasaneagle/jpdmgen/commit/95ae8b184ea1c5779dae65b7b06e3164c2129328)

### Features

- Move yaw and rpy transform calculations into maths module [(99e6532)](https://github.com/iwishiwasaneagle/jpdmgen/commit/99e65324a517e07a9feff84097b432bb6f2f1950)
- Move poly traj into own module and fully test [(a727658)](https://github.com/iwishiwasaneagle/jpdmgen/commit/a727658c41fc53ddd20d58b22eb60d868efcf799)
- Use nptyping library to define shapes of data [(d824d27)](https://github.com/iwishiwasaneagle/jpdmgen/commit/d824d271a8314a3930cad0c85fafaeb0dea44607)
- Trajectory env refactored [(2aeeb96)](https://github.com/iwishiwasaneagle/jpdmgen/commit/2aeeb96af850c67a535d3dc264ac37d73d083883)
- Cache pybullet by installing in empty workflow [(15e4a88)](https://github.com/iwishiwasaneagle/jpdmgen/commit/15e4a888fc26856304c582cdd71bd77484438820)
- Properly handle asset paths [(583c600)](https://github.com/iwishiwasaneagle/jpdmgen/commit/583c6006dbf07b5f1c6b01b82ba6a4bf6949f92c)

### Miscellaneous Tasks

- Use more appropriate clip_scalar [(e288313)](https://github.com/iwishiwasaneagle/jpdmgen/commit/e288313a97f4818f7644d12cc02bfe5326a2b396)
- Deal with refactoring of drone envs [(05fe4c5)](https://github.com/iwishiwasaneagle/jpdmgen/commit/05fe4c5db17d3e626c4268efe83572d4a5eabdca)
- Give ground plane collision data in info dict [(417de31)](https://github.com/iwishiwasaneagle/jpdmgen/commit/417de315f390125fe163231629594e022538efa9)
- Deal with the various refactors [(0455adb)](https://github.com/iwishiwasaneagle/jpdmgen/commit/0455adb6254d79fef835a87562d4984494d51c78)
- Deal with the various refactors [(76d4715)](https://github.com/iwishiwasaneagle/jpdmgen/commit/76d47153ae65c6d96a9271e14f125f1d461921eb)
- Add nptyping to requirements [(c0a45f6)](https://github.com/iwishiwasaneagle/jpdmgen/commit/c0a45f62bfa0b7c34a2e72a9d0e4724a84abad58)
- Export DronePlus from envs [(7860f19)](https://github.com/iwishiwasaneagle/jpdmgen/commit/7860f19cd6a1a85fb6fb37f94ed62d93fedd4b82)
- Update changelog for v0.2.0 [skip pre-commit.ci] [(1a4ccd8)](https://github.com/iwishiwasaneagle/jpdmgen/commit/1a4ccd82bcd5854b3623c5a9da39b81ffabdf152)

### Refactor

- Remove custom action types and remodel them as types [(025ca4e)](https://github.com/iwishiwasaneagle/jpdmgen/commit/025ca4e7bef6a8099d9e1e34f9e3db88eb90a896)
- Alter way the info dict is manipulated [(a17a9a4)](https://github.com/iwishiwasaneagle/jpdmgen/commit/a17a9a4adb3c2164658f75cdf9907e0bec6e5413)
- Merge postion alt drone env into trajectory control commands [(4f931bd)](https://github.com/iwishiwasaneagle/jpdmgen/commit/4f931bdfa6e33fdcf1a0358e84b251d3ec76a44b)
- Move into own module [(1244c96)](https://github.com/iwishiwasaneagle/jpdmgen/commit/1244c967e0c8c624d3b9e3f4789a58fab5ceb2fd)
- Use apply_rpy from maths module [(b7182dc)](https://github.com/iwishiwasaneagle/jpdmgen/commit/b7182dca4ae1ccd1c49f56d1b0857a1d5dd40ba0)
- Merged with trajectory env API [(a0c30c3)](https://github.com/iwishiwasaneagle/jpdmgen/commit/a0c30c3dbd62e2f840ebb443c358d4f27c754e93)

### Testing

- Fully test PID controllers [(94862d4)](https://github.com/iwishiwasaneagle/jpdmgen/commit/94862d4d8247dbcf4b13290cfc6a511275126cad)

## [0.1.2] - 2023-01-31

### Bug Fixes

- Use different URLs for files that are hosted on gh-pages [(d84ccb3)](https://github.com/iwishiwasaneagle/jpdmgen/commit/d84ccb37136503aa81f8c1b8ec446dd29b61e6c0)
- Docstr-cov was failing whilst trying to get baseline [(49d2077)](https://github.com/iwishiwasaneagle/jpdmgen/commit/49d207739829f082ba04ad548004884753f0f2fc)
- Hierarchy error in titles [(21ea9b7)](https://github.com/iwishiwasaneagle/jpdmgen/commit/21ea9b7546144286fdbc361bf5be74535ebcc350)

### Documentation

- Add controllers to docs [(856c336)](https://github.com/iwishiwasaneagle/jpdmgen/commit/856c336ab2a64cc35594d2041ac8114ab0577654)
- Improve index.rst to act as the landing page for the hosted docs [(1091aed)](https://github.com/iwishiwasaneagle/jpdmgen/commit/1091aed4ea211d706283a5affa9643670617a9e5)

### Miscellaneous Tasks

- Mock pybullet_data [(d858099)](https://github.com/iwishiwasaneagle/jpdmgen/commit/d858099acad558b9fe0665b587fa062b6ac07ef9)
- Remove -q from checkout [(f5493c6)](https://github.com/iwishiwasaneagle/jpdmgen/commit/f5493c671f64fd533d1c04e85c68443de829af79)
- Update changelog for v0.1.2 [skip pre-commit.ci] [(0495c56)](https://github.com/iwishiwasaneagle/jpdmgen/commit/0495c5663b11d65370a22533b63f43361d7eba51)

### Refactor

- Speed up docstr build by only requireing docs/requirements.txt [(6d6cb64)](https://github.com/iwishiwasaneagle/jpdmgen/commit/6d6cb6401a33edb7ce3fa1eac218750f0e50e9ef)
- Move PyBulletIds to types module [(b3ad632)](https://github.com/iwishiwasaneagle/jpdmgen/commit/b3ad632744045e7f411f6fcf1e10f2d660b8712e)
- Remove license from docs [(e4a551d)](https://github.com/iwishiwasaneagle/jpdmgen/commit/e4a551dbe4b1ce761885c09058c3b7eb9d370a05)

## [0.1.1] - 2023-01-31

### Bug Fixes

- Running very slow because of np.asarray [(6e19344)](https://github.com/iwishiwasaneagle/jpdmgen/commit/6e193445972e9aa307bb648cf01d512ffbf3fa16)
- Reused function name [(2690f4e)](https://github.com/iwishiwasaneagle/jpdmgen/commit/2690f4ebc6f16445502014dcf6438ed3acccbdb8)
- Ignore license.rst as this is pretty much immutable [(4ea47ca)](https://github.com/iwishiwasaneagle/jpdmgen/commit/4ea47ca3c09d105832e1bb23effae80fead3c226)
- Use pyproject.toml for rstcheck config [(3985349)](https://github.com/iwishiwasaneagle/jpdmgen/commit/3985349b59fe73de1a3e1de90e55bf05e6261d26)
- If there's no coverage, set to 0 [(6949c23)](https://github.com/iwishiwasaneagle/jpdmgen/commit/6949c230201bf4b12eb6dc88ac0ef589be7b7a3a)
- Allow --accept-empty in case there's no python files [(1714bbf)](https://github.com/iwishiwasaneagle/jpdmgen/commit/1714bbfcbb494225aa0b51b6b3beb353eec4861e)

### Documentation

- Start of BaseDroneEnv docs [(7b9553f)](https://github.com/iwishiwasaneagle/jpdmgen/commit/7b9553fe5d68e92fc78e4c9b594c4cac55101db8)
- Do most of the types docs [(c3e3a18)](https://github.com/iwishiwasaneagle/jpdmgen/commit/c3e3a18e178a60e5f5ee8c85ed141d305b48e2c3)
- More docs for BaseDroneEnv [(a17a30c)](https://github.com/iwishiwasaneagle/jpdmgen/commit/a17a30cc6230177180e2f2eafcc576f74c9c1169)
- Include BaseDroneEnv in generated docs [(de0f058)](https://github.com/iwishiwasaneagle/jpdmgen/commit/de0f0585d1f734483369e133056703d41fda9812)

### Features

- Check docstr coverage isn't being reduced [(a4be21b)](https://github.com/iwishiwasaneagle/jpdmgen/commit/a4be21b1a618060c360cdff4c0f32b828117f1e2)

### Miscellaneous Tasks

- Improve readme [(f909511)](https://github.com/iwishiwasaneagle/jpdmgen/commit/f9095111c7fbc999cab9a9a45d2f8d1d3ae4ac4f)
- Add future work on motor modelling [(9307e91)](https://github.com/iwishiwasaneagle/jpdmgen/commit/9307e91da249088486242203515f0ff98a985c58)
- Mute git checkouts in docstr-cov CI [(9721eb9)](https://github.com/iwishiwasaneagle/jpdmgen/commit/9721eb9116cca9b4bb33f97a3428768292861812)
- Fix flake errors (ambiguous variable, unused imports) [(1f2b750)](https://github.com/iwishiwasaneagle/jpdmgen/commit/1f2b750848dd948d5bdd97bf785750efc76b78f5)
- Add intersphinx mapping for gymnasium [(c0316f8)](https://github.com/iwishiwasaneagle/jpdmgen/commit/c0316f8ac8fada3d1022b7e4da9474469b216d64)
- Don't upload badge as artifact. Can't currently use it [(6d9817b)](https://github.com/iwishiwasaneagle/jpdmgen/commit/6d9817b7cbea934cfb9c10a25de86693cbf35ccb)
- Upload docstr-cov badge to gh-pages [(d7e923a)](https://github.com/iwishiwasaneagle/jpdmgen/commit/d7e923aa0b44b813db5d4bb8b947f29a9d7a3e4a)
- Update changelog for v0.1.1 [skip pre-commit.ci] [(066c6c3)](https://github.com/iwishiwasaneagle/jpdmgen/commit/066c6c3a9b3a432ae86bfb8c695314d55f9f80a2)

## [0.1.0] - 2023-01-27

### Bug Fixes

- Updates to setup since there's no CPP [(6e489fb)](https://github.com/iwishiwasaneagle/jpdmgen/commit/6e489fb22f5aa2edf54887734eae587d649b61bc)
- Use pip wheel rather than cibuildwheel as it's currently a pure python package [(94a0837)](https://github.com/iwishiwasaneagle/jpdmgen/commit/94a0837ee37f26784de4231afd42b5258e44f09a)
- No need for many and musl as it's pure python for now [(3e57819)](https://github.com/iwishiwasaneagle/jpdmgen/commit/3e57819690da9d1468f14666b7ff663d5dbd06fb)
- Move to only python >=3.10 [(87570d9)](https://github.com/iwishiwasaneagle/jpdmgen/commit/87570d989c2609620579e67a1546c18c85d96c58)
- Space [(a870dfe)](https://github.com/iwishiwasaneagle/jpdmgen/commit/a870dfe5564cca971943ea753c5ae3071246aacf)
- Spaces [(808a9cd)](https://github.com/iwishiwasaneagle/jpdmgen/commit/808a9cdd8b1093732e5cd2147ff1902ff422640c)
- Fix issues with runners not correctly initializing due to misconfigured matrix [(5c88fca)](https://github.com/iwishiwasaneagle/jpdmgen/commit/5c88fca8d21d87849e53567db166c341fc027b90)
- Set correct permissions for gh pages and releases [(8aabdfd)](https://github.com/iwishiwasaneagle/jpdmgen/commit/8aabdfd9f0fc5faead8cfa2a1e3f3c5245a63f1e)
- Install docs/requiremnts.txt rather than tests/ [(961f940)](https://github.com/iwishiwasaneagle/jpdmgen/commit/961f940aa2766a6d35912ead715dc73241f5b91b)
- Revert permissions, as this is done through settings console [(07b9198)](https://github.com/iwishiwasaneagle/jpdmgen/commit/07b9198b677288b2a15ef73e1c315fb09af12113)

### Features

- Initial commit [(2b2928c)](https://github.com/iwishiwasaneagle/jpdmgen/commit/2b2928c2901d69e5954e0558ba9e2cd204b2a48c)
- Move back to pip as conda wasn't required [(071ae4b)](https://github.com/iwishiwasaneagle/jpdmgen/commit/071ae4b8f5ecb21e82fa4263611e0012c520271a)

### Miscellaneous Tasks

- Update changelog for v0.2.0 [skip pre-commit.ci] [(e4b25d4)](https://github.com/iwishiwasaneagle/jpdmgen/commit/e4b25d479669a2fa7301dd51f1090d5e378655c2)
- Update changelog for v0.1.0 [skip pre-commit.ci] [(9b40238)](https://github.com/iwishiwasaneagle/jpdmgen/commit/9b402385ae225dca824639154dbdb241394c1592)
- Update how conda is used in CI [(d9704d8)](https://github.com/iwishiwasaneagle/jpdmgen/commit/d9704d87a6442ddbc9820d6118a3545d049a3e68)
- A little import cleanup [(6cb83e9)](https://github.com/iwishiwasaneagle/jpdmgen/commit/6cb83e99ef333fef13b7d0e2296747652d470915)
- Update CI to also run integration tests but standalone [(aa867ce)](https://github.com/iwishiwasaneagle/jpdmgen/commit/aa867ce226ec9eae434881b1b0a050bb832cb484)
- Document transform funcs to show what's happening behind the scenes in PB3 [(645ce50)](https://github.com/iwishiwasaneagle/jpdmgen/commit/645ce50af91cb0de18737119d1f9136cf6391c80)
- Update changelog for v0.1.0 [skip pre-commit.ci] [(79d9f01)](https://github.com/iwishiwasaneagle/jpdmgen/commit/79d9f01083530d5b87e2d4984665ef6e77d4d8ec)
- Delete changelog for redo of v0.1.0 [skip pre-commit.ci] [(45f3fb3)](https://github.com/iwishiwasaneagle/jpdmgen/commit/45f3fb3e95c8062fe538e9fe10d8eb0e9602fa11)
- Update changelog for v0.1.0 [skip pre-commit.ci] [(d3dc768)](https://github.com/iwishiwasaneagle/jpdmgen/commit/d3dc7686ca804b005aa050f14521381a74cae501)

### Refactor

- Docs filenames and symlink names [(64aba9a)](https://github.com/iwishiwasaneagle/jpdmgen/commit/64aba9a8b773d72df807f4177ca34b068e4de8c0)

### Testing

- Skip integration tests [(2651c8b)](https://github.com/iwishiwasaneagle/jpdmgen/commit/2651c8be53738c47f3c79c7622f1ab0c15621f4b)
- Fix tests since quats and euler have multiple correct variations [(5a3a96d)](https://github.com/iwishiwasaneagle/jpdmgen/commit/5a3a96daf1b9374d605fba3f4dae620973b88c8e)

