## Setup environment
Install Mujoco backend with a bash script all at once, details at [Mujoco's official github](https://github.com/openai/mujoco-py) 
```bash
bash setup.sh
```

With python >= 3.7
```bash
pip install -r requirements.txt
```
## Config environments
All environments' file in ***environments*** folder

## Run MOPDERL
<!-- All bash script for running is in **bash** folder. (Ex: mo-swimmer-v5 environment):
**Check bash before running**
**Correct example see Swimmer environment**
```bash
bash ./bash/swimmerv2.py
```
Digging into the bash file: -->
```bash
# Run from the repository root so imports resolve correctly.
# Boundary-only mode (one-hot weights only) is enforced by the codebase.

python -m mopderl.MOPDERL.run_mo_pderl \
  -env=mo-swimmer-v5 \
  -logdir=your/dir \
  -disable_wandb \
  -seed=123 \
  -save_ckpt=0

# Continue running latest run after disconnected
# python -m mopderl.MOPDERL.run_mo_pderl -env=mo-swimmer-v5 -logdir=your/dir -disable_wandb -seed=123 -save_ckpt=0 -checkpoint

# Continue running specific run after disconnected
# python -m mopderl.MOPDERL.run_mo_pderl -env=mo-swimmer-v5 -logdir=your/dir -disable_wandb -seed=123 -save_ckpt=0 -checkpoint -checkpoint_id=10
```

Other config please seek into *run_mo_pderl.py* file:  
```python
python -m mopderl.MOPDERL.run_mo_pderl -h
```
<!-- ## Run PGMORL (Skip)
For example, running mo-swimmer-v5 environment:
```python
cd PGMORL
python scrips/swimmer-v2.py --pgmorl --savedir=../result/PGMORL/mo-swimmer-v5
``` -->

Results for each run will be at: "*your_save_dir/environment_name/run_***/archive*"


## Credit
This repository is largely based on the code of [Cris Bodnar et al](https://github.com/crisbodnar/pderl) and we would like to thank them for making their code publicly available.
