#!/usr/bin/env bash
# =============================================================================
# 8 卡预训练 Qwen2.5-7B 结构 —— 带注释的参考模板
#
# 定位: 与 MindSpeed-LLM examples/ 下官方脚本同构, 用于"读懂每个参数"。
#       实际生产建议以你检出版本的官方脚本为底, 用本文件对照理解/修改。
# 执行位置: MindSpeed-LLM 仓库根目录
# =============================================================================
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=600

# ---------------- 路径 ----------------
TOKENIZER_PATH=${TOKENIZER_PATH:-./model_from_hf/Qwen2.5-7B}   # HF tokenizer 目录
DATA_PATH=${DATA_PATH:-./dataset/corpus_text_document}          # 预处理产物前缀
CKPT_SAVE=${CKPT_SAVE:-./ckpt/qwen25_7b}
# 续训时: CKPT_LOAD 指向 hf2mcore 转换产物; 从零预训练则留空
CKPT_LOAD=${CKPT_LOAD:-}

# ---------------- 并行策略: TP*PP*DP = 2*1*4 ----------------
TP=2
PP=1
MBS=1      # micro batch size / 卡
GBS=64     # global batch size, 必须被 DP(=8/TP/PP) * MBS 整除

DISTRIBUTED_ARGS="
    --nproc_per_node 8 --nnodes 1 --node_rank 0
    --master_addr 127.0.0.1 --master_port 29503
"

# ---------------- 模型结构: Qwen2.5-7B ----------------
MODEL_ARGS="
    --use-mcore-models
    --num-layers 28
    --hidden-size 3584
    --ffn-hidden-size 18944
    --num-attention-heads 28
    --group-query-attention
    --num-query-groups 4
    --seq-length 4096
    --max-position-embeddings 32768
    --make-vocab-size-divisible-by 1
    --padded-vocab-size 152064
    --disable-bias-linear
    --add-qkv-bias
    --position-embedding-type rope
    --rotary-base 1000000
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
"

# ---------------- 并行与显存 ----------------
PARALLEL_ARGS="
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 2
"
# 显存吃紧: recompute-num-layers 调大(上限=每个 PP stage 的层数)
# 显存富余: 调小它换吞吐; 进阶再看 --swap-attention / 自适应重计算(见 MindSpeed 文档)

# ---------------- 昇腾融合算子(910B 全开, 910A 去掉 flash-attn 相关) ----------------
FUSION_ARGS="
    --use-flash-attn
    --use-fused-rmsnorm
    --use-fused-swiglu
    --use-fused-rotary-pos-emb
"

# ---------------- 训练超参 ----------------
TRAIN_ARGS="
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters 50000
    --lr 1.0e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --lr-warmup-fraction 0.01
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.008
    --bf16
    --no-gradient-accumulation-fusion
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path $TOKENIZER_PATH
"

DATA_ARGS="
    --data-path $DATA_PATH
    --split 990,10,0
"

OUTPUT_ARGS="
    --log-interval 1
    --save-interval 2000
    --eval-interval 2000
    --eval-iters 10
    --save $CKPT_SAVE
"
[ -n "$CKPT_LOAD" ] && OUTPUT_ARGS="$OUTPUT_ARGS --load $CKPT_LOAD"

mkdir -p "$CKPT_SAVE" logs
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS $PARALLEL_ARGS $FUSION_ARGS $TRAIN_ARGS $DATA_ARGS $OUTPUT_ARGS \
    2>&1 | tee "logs/pretrain_qwen25_7b_$(date +%m%d_%H%M).log"
