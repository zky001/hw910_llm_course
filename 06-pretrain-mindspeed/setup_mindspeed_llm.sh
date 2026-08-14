#!/usr/bin/env bash
# =============================================================================
# MindSpeed-LLM 环境搭建（Megatron-LM + MindSpeed + MindSpeed-LLM 三件套）
#
# ⚠️ 版本三角必须配套：下面三个版本变量请以 MindSpeed-LLM 目标分支 README
#    中「版本配套」说明为准修改，不要各自拉 master 最新！
# 前置：第 1 章环境(torch/torch_npu/CANN)已就绪
# =============================================================================
set -e

WORKDIR=${WORKDIR:-$HOME/mindspeed_work}
# ---- 版本锁定（按官方配套说明修改这三行）----
MSLLM_BRANCH=${MSLLM_BRANCH:-master}          # 例: 2.1.0
MEGATRON_TAG=${MEGATRON_TAG:-core_r0.8.0}     # MindSpeed-LLM 文档指定的 Megatron tag
MINDSPEED_REF=${MINDSPEED_REF:-master}        # MindSpeed-LLM 文档指定的 MindSpeed commit

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo ">>> [1/4] 克隆 MindSpeed-LLM (分支: $MSLLM_BRANCH)"
[ -d MindSpeed-LLM ] || git clone -b "$MSLLM_BRANCH" https://gitee.com/ascend/MindSpeed-LLM.git

echo ">>> [2/4] 克隆 Megatron-LM 并检出配套 tag ($MEGATRON_TAG), 拷入 megatron 包"
[ -d Megatron-LM ] || git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git fetch --tags && git checkout "$MEGATRON_TAG" && cd ..
cp -r Megatron-LM/megatron MindSpeed-LLM/

echo ">>> [3/4] 克隆并安装 MindSpeed (ref: $MINDSPEED_REF)"
cd MindSpeed-LLM
[ -d MindSpeed ] || git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed && git checkout "$MINDSPEED_REF" && pip install -e . && cd ..

echo ">>> [4/4] 安装 MindSpeed-LLM 依赖, 建立工作目录"
pip install -r requirements.txt
mkdir -p logs dataset ckpt model_from_hf

echo ""
echo "✅ 完成。工作目录: $WORKDIR/MindSpeed-LLM"
echo "   下一步: 1) preprocess_data.sh 处理数据  2) 参考 examples/ 或本章模板启动预训练"
