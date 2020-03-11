

## quick start

1. install mujoco: https://github.com/openai/mujoco-py/#install-mujoco
	- on ubuntu you also need to install some requirements `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
2. setup virtual environment:
```bash
git clone git@github.com:neuroevolution-ai/NeuroEvolution.git
cd NeuroEvolution
virtualenv ~/.venv/neuro --python=python3
. ~/.venv/neuro/bin/activate
pip install python3-tk scoop pybullet 'gym[all]' torch deap matplotlib

# if mujoco is needed, also run
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin pip install mujoco-py 
```

3. (optional) change configuration `nano Configuration.json`
3. run training
	- this will take some time
	- `LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin python -m scoop CTRNN_ReinforcementLearning_CMA-ES.py`
4. show results:
	- `LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin python CTRNN_Visualisierung.py`
