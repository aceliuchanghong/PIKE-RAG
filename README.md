
```shell
# 初始化python环境
uv init
uv venv
source .venv/bin/activate
uv pip install .
conda create -n ep_kg python=3.10 -y
conda activate ep_kg

# win
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv/Scripts/activate

# 设置代理源
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple
vi ~/.bashrc==>export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple==>source ~/.bashrc
```