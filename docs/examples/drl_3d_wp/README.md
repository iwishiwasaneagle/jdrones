```bash
PYTHONPATH=$PWD python drl_3d_wp \
    learn mlp \
    -N 5000000 \
	--net_arch_mlp_width 512 \
	--net_arch_mlp_depth 4 \
	--lr 0.0002 0.00001 .5 \
	--n_envs 16 \
	--wandb_project jdrones \
	--batch_size 4096 \
	--n_steps 1024 \
	--vec_env_cls subproc \
	--n_eval 10 -T 10 \
	--clip_range 0.2 \
	--n_sub_envs 2
```
