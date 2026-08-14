# 第 1 章 · 环境搭建与验证

> **本章目标**：把「驱动 → CANN → Python/torch_npu」三层装对，并且**每装一层就验证一层**。做完本章，你会得到一套通过全部自检的训练环境，以及一个以后随时可以重跑的体检脚本。

**前置条件**：能 SSH 到服务器；有 root 或 sudo（仅驱动安装需要）；机器能访问外网或有内网 pip/镜像源。

> 💡 **先跑体检再动手。** 很多机器交付时驱动甚至 CANN 已经装好。先执行本章的 [`check_env.sh`](check_env.sh)，只补缺的层，不要重复安装：
>
> ```bash
> bash 01-environment/check_env.sh
> ```

## 1.1 版本规划：装之前先想清楚

昇腾环境 90% 的问题源于版本不配套。三层的关系是：

```text
驱动/固件  ←配套→  CANN  ←配套→  torch_npu  ←严格一一对应→  PyTorch
```

规则只有三条：

1. **torch 和 torch_npu 版本必须严格对应**（如 torch 2.5.1 ↔ torch_npu 2.5.1.x），不对应会直接 import 失败或运行时报错。
2. **torch_npu 对 CANN 有最低版本要求**，CANN 对驱动有最低版本要求。新驱动通常兼容旧 CANN，反过来不一定。
3. 装之前，到官方「版本配套表」查一次，照抄一组已验证的组合。本课程基线组合与查询入口见[附录 A](../appendix/version-matrix.md)。

本章示例使用基线组合：**CANN 8.1.RC1 + Python 3.10 + torch 2.5.1 + torch_npu 2.5.1**（aarch64）。

## 1.2 第一层：驱动与固件（需要 root，装过就跳过）

从[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers)下载对应**芯片型号**（910A / 910B）和**架构**（aarch64 / x86_64）的 `.run` 包。

```bash
# 包名示意，以实际下载为准
chmod +x Ascend-hdk-910b-npu-driver_*_linux-aarch64.run
chmod +x Ascend-hdk-910b-npu-firmware_*.run

# 先驱动后固件
sudo ./Ascend-hdk-910b-npu-driver_*_linux-aarch64.run --full --install-for-all
sudo ./Ascend-hdk-910b-npu-firmware_*.run --full

# 固件升级需要重启生效
sudo reboot
```

> ✅ **验证**：重启后执行 `npu-smi info`，应列出全部 8 张卡且 `Health` 均为 `OK`。这一步不过，后面全都免谈——先去第 9 章「硬件层排查」。

## 1.3 第二层：CANN（Toolkit + Kernels）

CANN 从[昇腾社区-CANN](https://www.hiascend.com/software/cann) 下载，需要装**两个包**：

- **Toolkit**（开发套件，含运行时/编译器/HCCL）
- **Kernels**（二进制算子包，**按芯片型号选**：910B 选 `kernels-910b`）

```bash
# CANN 依赖若干系统库和 Python 包（用于算子编译），先装上
pip install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py

chmod +x Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run
chmod +x Ascend-cann-kernels-910b_8.1.RC1_linux.run

# 普通用户安装到 ~/Ascend，root 安装到 /usr/local/Ascend（推荐后者，多用户共享）
sudo ./Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --install
sudo ./Ascend-cann-kernels-910b_8.1.RC1_linux.run --install
```

装完后，**每个要用 NPU 的 shell 都必须先加载 CANN 环境变量**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

建议直接写进 `~/.bashrc`。忘记 source 是新手第一大坑，症状是 `import torch_npu` 报找不到 `libascendcl.so` 之类的动态库。

> ✅ **验证**：
>
> ```bash
> source /usr/local/Ascend/ascend-toolkit/set_env.sh
> cat $ASCEND_TOOLKIT_HOME/version.cfg      # 能看到 CANN 版本号
> ```

## 1.4 第三层：Python + PyTorch + torch_npu

推荐用 conda 隔离环境（aarch64 机器用 [Miniforge](https://github.com/conda-forge/miniforge)）：

```bash
conda create -n npu python=3.10 -y
conda activate npu

# 1) 装 PyTorch —— 注意：装 CPU 版即可，NPU 支持完全由 torch_npu 提供
#    aarch64 上 pip 默认源给的就是 CPU 版
pip install torch==2.5.1

# 2) 装配套的 torch_npu（包名带横杠，import 时是下划线）
pip install torch-npu==2.5.1

# 3) 常用基础包
pip install numpy pyyaml setuptools
```

国内网络建议配置 pip 镜像（如清华源），下载模型建议用 ModelScope 或 HF 镜像（第 4 章细讲）。

> ✅ **验证**（本章提供了更完整的 [`verify_npu.py`](verify_npu.py)）：
>
> ```bash
> python -c "
> import torch, torch_npu
> print('torch:', torch.__version__)
> print('torch_npu:', torch_npu.__version__)
> print('npu available:', torch.npu.is_available())
> print('npu count:', torch.npu.device_count())
> x = torch.randn(2, 3).npu()
> print((x @ x.T).cpu())
> "
> ```
>
> 期望：`npu available: True`、`npu count: 8`、矩阵乘出数字不报错。

## 1.5 一键体检与压测

本章附带两个脚本，以后每次拿到新机器/新环境都先跑：

```bash
# 体检：逐层检查驱动/CANN/Python 包/环境变量，输出 PASS/FAIL 清单
bash 01-environment/check_env.sh

# 功能+性能验证：8 卡逐卡做矩阵乘，报告每张卡的 TFLOPS
python 01-environment/verify_npu.py
```

`verify_npu.py` 会对每张卡跑一个 FP16 大矩阵乘并输出实测 TFLOPS。**8 张卡数值应该接近**；某张卡明显偏慢（差 20% 以上）通常意味着降频（散热/功耗）或该卡有问题，先解决再开训，否则它会拖慢整个分布式任务（木桶效应）。

示例输出（示意）：

```text
[npu:0] matmul 8192x8192x8192 fp16: 6.15 ms/iter  ≈ 178.8 TFLOPS
[npu:1] matmul 8192x8192x8192 fp16: 6.21 ms/iter  ≈ 177.1 TFLOPS
...
[npu:7] matmul 8192x8192x8192 fp16: 6.17 ms/iter  ≈ 178.2 TFLOPS
All 8 NPUs OK, max deviation 1.0%
```

## 1.6 可选：容器化环境（推荐团队使用）

裸机 conda 适合个人折腾；团队共用一台 8 卡机时，强烈建议用容器把「CANN + Python 栈」封装起来（驱动留在宿主机）：

```bash
# 方式一：官方镜像（昇腾镜像仓库 ascendhub 提供 ascend-pytorch 等镜像，
#         tag 已经把 CANN/torch/torch_npu 配好，按配套表选 tag）
docker pull <ascendhub 镜像地址>/ascend-pytorch:<tag>

# 方式二：任意镜像 + 手动挂载设备与驱动（这套挂载参数是通用的）
docker run -it --ipc=host --network host \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v $PWD:/workspace -w /workspace \
  <image> bash
```

如果集群装了 **Ascend Docker Runtime**（MindX DL 组件），可以省掉这一长串：`docker run --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0-7 ...`。

> `--ipc=host`（或足够大的 `--shm-size`）别忘了：PyTorch DataLoader 多进程要用共享内存，不加会在训练中报 shm 相关错误。

## 1.7 常见问题

| 症状 | 原因与处理 |
|---|---|
| `import torch_npu` 报 `libascendcl.so: cannot open shared object file` | 没 `source .../set_env.sh`，或 CANN 没装 |
| `import torch_npu` 报 undefined symbol / ABI 错误 | torch 与 torch_npu 版本不配套，按附录 A 重装 |
| `torch.npu.is_available()` 返回 False | 驱动没装好（先看 `npu-smi info`）；或容器里没挂载 `/dev/davinci*` |
| pip 装 torch 巨慢/装成 x86 包 | aarch64 机器 + 未配置镜像源；确认 `uname -m` 与 wheel 架构一致 |
| 首次跑模型特别慢，之后正常 | 算子在线编译的正常现象，编译结果会缓存（详见第 8 章 jit_compile 说明） |
| 只有部分卡可见 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否被设置过（对标 `CUDA_VISIBLE_DEVICES`） |

**下一章** → [第 2 章 · PyTorch NPU 上手与单卡训练](../02-single-card/README.md)：环境好了，写第一行 NPU 代码。
