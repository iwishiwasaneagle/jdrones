# Changelog

All notable changes to this project will be documented in this file.

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

