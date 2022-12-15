```bash
conda activate
conda create --name jp python=3.10
conda activate jp
conda install jupyterlab
conda install nb_conda_kernels
jupyter lab
```

```bash
conda activate
conda create --name final38 python=3.8
conda activate final38
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -e .
conda install nb_conda_kernels
python -c "import mujoco_py"
```

```bash
conda activate
conda create --name gluon
conda activate gluon
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install autogluon
conda install nb_conda_kernels
```

```
conda activate
conda create --name torch113 python=3.10
conda activate torch113
conda install pytorch=1.13.0 torchvision=0.14.0 torchaudio=0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install nb_conda_kernels
```
