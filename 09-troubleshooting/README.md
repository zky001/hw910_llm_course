# 第 9 章 · 稳定性与故障排查

> **本章目标**：把 8 卡训练中最高频的故障——HCCL 报错、OOM、NaN、卡死、设备占用——变成「见过、知道为什么、知道怎么办」。附一个一键收集诊断信息的脚本，以及长时间训练的断点续训纪律。

**前置条件**：无。建议第一次通读，之后当手册查。

## 9.1 排障方法论：两个先问

**① 先问「哪一层出的事」**，从下往上排，下层不稳上层全是幻觉：

```text
硬件/驱动(npu-smi 正常吗) → 通信(HCCL) → 框架(torch_npu/DeepSpeed/MindSpeed) → 训练动态(loss/数值)
```

**② 多卡报错先问「谁先出的事」**。集合通信的特性是**一人生病全员陪葬**：一个 rank 挂了，其他 7 个只会报「等不到它」的超时错。**满屏 HCCL timeout 里，真凶往往是唯一不报 HCCL 错、或最早打出异常栈的那个 rank。** 排查顺序：

```bash
# torchrun 每个 rank 的输出分开存的话(建议 --log-dir), 按时间戳找最早的异常
grep -rn "Traceback\|ERROR" logs/ | sort -t: -k2 | head

# 再看昇腾底层日志(见 9.2)里最早的 ERROR
```

## 9.2 日志体系：报错信息不够时去哪挖

| 日志 | 位置 | 内容 |
|---|---|---|
| 训练框架日志 | 你的 stdout/stderr（建议 `tee` 存盘） | Python 栈、loss 曲线 |
| **plog（host 侧）** | `~/ascend/log/plog/plog-*.log` | CANN/HCCL/runtime 的详细错误，**HCCL 问题必看** |
| device 日志 | `~/ascend/log/` 下 device 相关目录 | 设备侧异常 |
| 内核日志 | `dmesg` | 掉卡、复位等硬件事件 |

两个常用开关（**排查时才开，平时保持默认**，全量日志又大又拖速度）：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1        # 0=debug 1=info 2=warning 3=error(默认)
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # 底层日志直接打到屏幕
```

另一个专用于定位「到底是哪个算子出错」的开关：

```bash
export ASCEND_LAUNCH_BLOCKING=1   # 同步执行模式: 报错栈会停在真正出错的算子上
                                  # (异步模式下栈经常指向无辜的下游算子)。很慢, 只在复现问题时开
```

## 9.3 高频故障 Playbook

### ① 启动阶段 HCCL 建链失败（connect timeout）

**症状**：起训即报 `HCCL ... connect timeout / get socket timeout`，训练没开始。
**排查顺序**：
1. 残留进程占卡：`npu-smi info` 看有无上一次训练的进程 → `pkill -9 -f train`，等几秒显存释放。
2. 端口冲突：换 `--master_port`。
3. 卡数/进程数不一致：`--nproc_per_node` 与 `ASCEND_RT_VISIBLE_DEVICES` 数量对得上吗。
4. 大任务建链慢属正常：`export HCCL_CONNECT_TIMEOUT=600`。
5. （多机）NPU 网口没配好/防火墙拦了、或需 `export HCCL_WHITELIST_DISABLE=1`。

### ② 训练中途 HCCL 超时（notify wait / exec timeout）

**症状**：跑了几小时后某步集体报 `HCCL ... timeout`，常见关键词 `notify wait`。
**本质**：某集合通信操作等了 `HCCL_EXEC_TIMEOUT`（默认 1836 秒）还没等齐 8 个 rank。
**根因 Top4**：
1. **某 rank 先死了**（OOM/断言失败）→ 按 9.1 找首个异常 rank，修它的问题。
2. **rank 间执行分支不一致**：如只让 rank0 做 eval/保存，其他 rank 已进入下一步的 AllReduce 空等 → 保证所有 rank 的集合通信序列严格一致（rank0 独占的工作要么放 barrier 保护、要么放通信序列之外）。
3. **超长的合法停顿**：保存几十 GB checkpoint 到慢存储、超长 eval → 调大 `HCCL_EXEC_TIMEOUT` 与框架侧 `ddp_timeout`。
4. **慢卡/降频**：见 9.3-⑦。

### ③ 显存 OOM

**症状**：`NPU out of memory. Tried to allocate ...`。
**先分辨真假 OOM**：报错前打印 `torch.npu.memory_summary()`，若 reserved 远大于 allocated → 是**碎片**，先 `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。
**真 OOM 的处理阶梯**（代价从小到大）：
1. 降 `cutoff_len`/序列长度、降 micro-batch（用梯度累积补回全局 batch）；
2. 开梯度检查点 / 调大重计算范围；
3. 升 ZeRO 档位（2→3）或加 TP（第 5/6 章的显存账算一下该到哪档）；
4. offload（速度代价大，最后手段）。
**训练几小时后才 OOM**：多为碎片累积或 eval/保存的峰值 → expandable_segments + 把 eval batch 降下来。

### ④ Loss NaN / Inf / 突刺

**910A（FP16）**：
- 偶发 overflow + 跳步（日志有 `Gradient overflow. Skipping step, loss scaler reducing...`）是动态 loss scale 的**正常工作方式**；
- 连续跳步/最终 NaN：lr 减半、`max_grad_norm` 收紧到 0.5、warmup 拉长；还不行检查数据（超长重复、乱码、空样本）。

**910B（BF16）下仍 NaN**，基本不是精度锅，按顺序查：
1. lr 过大或 warmup 太短（尤其从零预训练）；
2. 数据里有毒（脏样本、标签错位）——记下出事 step，把那个 batch 的样本 dump 出来看；
3. 代码 bug（自定义 loss 里除零/log(0)）；
4. 需要精确定位算子时：`ASCEND_LAUNCH_BLOCKING=1` + `torch.autograd.set_detect_anomaly(True)` 复现（很慢，小数据复现用）。

**要复现实验做对比**时的确定性开关：

```bash
export HCCL_DETERMINISTIC=true          # 通信归约顺序确定
```
```python
torch.use_deterministic_algorithms(True)  # 算子确定性(性能有损, 研究用)
```

### ⑤ 卡死不动（hang，无任何报错）

**症状**：日志停住、`npu-smi` 显示利用率 100% 或 0%，进程都在。
**取证**：给每个 rank 抓 Python 栈，看大家各自停在哪一行：

```bash
pip install py-spy
for pid in $(pgrep -f your_train.py); do echo "== $pid =="; py-spy dump --pid $pid; done
```

- 全部停在同一个 collective → 又是「分支不一致/某 rank 慢」（回 ②）；
- 有 rank 停在 DataLoader → worker 死锁/共享内存不足（容器 `--shm-size` 加大、`num_workers` 降低）；
- 停在 save/文件 IO → 存储 hang（NFS 抖动）。

### ⑥ 设备占用 / 僵尸进程

**症状**：明明没在训练，`npu-smi info` 显示显存被占、新任务起不来。

```bash
npu-smi info               # 看每张卡上的进程 PID
kill -9 <pid>              # 数秒后显存自动回收
# 进程杀不掉(D 状态)或显存不回收时, 复位单颗芯片(会打断该卡所有任务, 慎用):
npu-smi set -t reset -i <NPU_ID> -c 0
# 仍不行 → 重启机器
```

### ⑦ 速度突然变慢 / 时快时慢

按嫌疑排序：**温度降频**（`npu-smi info` 看 Temp/Power，机房空调/风道）→ **别的进程混用了你的卡**（npu-smi 看进程）→ **存储变慢**（数据盘/checkpoint 盘 IO 打满）→ **碎片累积**（expandable_segments）。8 卡任务只要一张卡慢，整体就慢——用第 1 章 `verify_npu.py` 逐卡复测定位。

## 9.4 长训练的保命纪律：断点续训

超过几小时的训练，当作「一定会中断」来设计：

1. **定期保存**：按步数（如每 500~2000 步）而不是只按 epoch；`save_total_limit` 控制磁盘。
2. **原子性**：写临时目录再 rename，避免中断留下半个 checkpoint（Trainer/Megatron 已内置类似机制；自研脚本自己保证）。
3. **存全套状态**：模型 + optimizer + lr_scheduler + step 数 + RNG 状态（`Trainer --resume_from_checkpoint`、Megatron `--load` 都是全套）。
4. **验证续训正确**：恢复后前几十步的 loss 应与中断前平滑衔接；跳变说明有状态没恢复（常见：数据顺序/采样器状态）。
5. **保存别拖垮训练**：大 checkpoint 写本地 NVMe 再异步搬运到共享存储；MindSpeed 有异步保存特性可用。
6. 用 `nohup`/`tmux`/调度系统跑，别让 SSH 断连杀掉训练。

## 9.5 一键收集诊断信息

找人求助（同事/社区/工单）前，跑这个脚本把现场打包，沟通效率翻倍：

```bash
bash 09-troubleshooting/collect_diag.sh
# 产物: diag_<时间戳>.tar.gz —— 含版本信息、npu-smi、关键环境变量、最近的 plog 错误、dmesg 摘录
```

提问模板：**现象一句话 + 最小复现命令 + 首个报错 rank 的完整栈 + diag 包**。

## 9.6 练习

1. 主动制造一次 OOM（batch 调大），走一遍 9.3-③ 的阶梯把它救回来。
2. 在 DDP 训练里给 rank3 加 `if rank==3: time.sleep(10000)`，体验 ② 的超时现场，练习从日志锁定真凶。
3. kill 掉一个正在保存 checkpoint 的训练，验证你的续训流程能无损恢复（9.4-④ 的 loss 衔接检查）。

**下一章** → [第 10 章 · 推理部署](../10-inference/README.md)：把训好的模型用起来。
