# GR 模型的切分方式：现状与文献

> 问题：GR（generative recommendation）目前主要用哪些切分方式，有哪些工作提到。
>
> 一句话结论：**GR 的切分是「稀疏侧模型并行 + 稠密侧数据并行」的双轨结构，
> 加上一个新出现的第三轨——长用户序列的上下文并行（CP）。**
> 这三轨的动机各不相同：第一轨为**容量**，第二轨为**吞吐**，第三轨为**显存**。
> 没有一轨是为延迟设计的，这是 GR 与 LLM 服务的根本差别。

---

## 一、总览：三轨结构

| 轨 | 切什么 | 主要动机 | 代表实现 |
|---|---|---|---|
| **① 稀疏侧（嵌入表）** | 表按行/列/表分片 | **容量**：单卡装不下 TB 级表 | TorchRec、HugeCTR HPS、DynamicEmb、OPER、Prism |
| **② 稠密侧** | 按 batch 复制（DP） | **吞吐**：稠密层小，复制最划算 | MTGenRec、TorchRec+Megatron 集成 |
| **③ 长序列** | 按序列长度切（CP） | **显存**：激活值随 L 增长 | RecSys'25 HSTU-Jagged-CP |
| （④ 专家） | 按专家切（EP） | 容量 + 稀疏激活 | MTGenRec 的 MMoE 层 |

**特点**：①②③ 不是同一层的选择，而是**同时打开**的。NVIDIA 的 HSTU 训练示例
就是这个结构：

> This example integrates **TorchRec for embedding table sharding** and
> **Megatron-Core for dense parallelism, including data, tensor, sequence,
> pipeline, and context parallelism**. This integration ensures efficient training
> by coordinating sparse (embedding) and dense (context/data) parallelisms within
> a single model.
>
> — [NVIDIA/recsys-examples, examples/hstu/training/README.md](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/README.md)

---

## 二、稀疏侧：嵌入表怎么切

### 2.1 TorchRec 的五种分片

| 类型 | 切法 | 适用 |
|---|---|---|
| **Table-wise (TW)** | 整张表放一张卡 | 中小表（默认） |
| **Row-wise (RW)** | 按 `hash_size` 均分 | **超大表**（单卡装不下） |
| **Column-wise (CW)** | 按 `emb_dim` 切 | 嵌入维很大、要均衡负载 |
| **Table-wise-row-wise (TWRW)** | 表落一台机器，机内按行分 | 利用 NVLink 高带宽 |
| **Grid-shard (GS)** | 先按列切，每片再 TWRW | 又大又宽的表 |
| **Data parallel (DP)** | 每卡一份副本 | 小表 |

来源：[TorchRec concepts](https://github.com/meta-pytorch/torchrec/blob/main/docs/source/concepts.rst)、
[PyTorch 分片教程](https://docs.pytorch.org/tutorials/advanced/sharding.md)。

**关键点**：这是**按容量切**，不是按计算切。查表是按 token 独立的，
所以分片后每卡只查自己那份，时间上等价于「按 N 摊」——不会更快，
只是装得下。这与我们仿真的结论一致（`helps("embp","lookup")` 的时间等价性）。

### 2.2 按「访问量」而不是「容量」分片

**Understanding Capacity-Driven Scale-Out Neural Recommendation Inference**
（[arXiv:2011.02084](https://arxiv.org/abs/2011.02084)）：

> Because the expected number of lookups per table is dependent on specific feature
> and how often it appears in request inputs, **capacity-balanced sharding may
> result in imbalanced shards** that perform significantly different amounts of
> work... This aggregate imbalance can cause certain shards to become a **critical
> path bottleneck**.

→ 应**按 pooling factor（每表期望查表次数）分片**，否则某个分片成为关键路径。

### 2.3 输入相关的细粒度分片

**OPER**（USENIX ATC'24，[论文](https://www.usenix.org/system/files/atc24-wang.pdf)）：

> overlook the **input-dependent behavior** of EMTs and parallelize them in a
> coarse-grained manner, resulting in unbalanced workload distribution and
> inter-GPU communication.

它做细粒度（per-embedding）通信，减少通信量。相对 TorchRec 的收益：
推理延迟 2.4–6.4×、训练吞吐 1.9–3.5×。

### 2.4 推理侧的另一种做法：不分片，分层存

**HugeCTR HPS**（[arXiv:2210.08804](https://arxiv.org/pdf/2210.08804)）：
不做模型并行，而做**层次化参数服务器** —— GPU 显存做一级缓存、CPU 内存二级、
SSD 三级。相比 CPU 基线 5–62× 加速。

> 这条路避免了分片的通信，代价是查表要走缓存层次。与 EmbP 是**替代关系**。

### 2.5 跨节点拆资源池

**Prism**（NSDI'25，[论文](https://www.usenix.org/conference/nsdi25/presentation/yang)）：
把 DLRM 拆成 **CPU 密集子图**与 **GPU 密集子图**，分别调度到 CPU 节点与 GPU 节点，
经 RDMA 互联。生产环境 10k+ GPU。

> 它指出 DLRM 的资源画像特殊：**大量 CPU 核与内存、少量 GPU**。
> 在普通多卡机器上跑会把 CPU/内存吃光、GPU 闲置。

---

## 三、稠密侧：为什么是 DP 而不是 TP

**MTGenRec**（美团，[arXiv:2505.12663](https://arxiv.org/abs/2505.12663)）：

> We follow TorchRec's design and employ a **hybrid parallel strategy: model
> parallelism for sparse models and data parallelism for dense models**.
> This design is driven by two considerations. First, data parallelism is
> impractical for the large embedding table, as no single device can store the
> entire embedding table. Second, **dense models are relatively small, making
> data parallelism more suitable**.

这就是双轨的原始理由，而且它是**互相印证**的：稀疏侧不得不切（容量），
稠密侧不必切（小），所以前者模型并行、后者数据并行。

**这对我们很重要**：它说明「稠密侧用 DP」不是随手的选择，而是
「稠密层小 ⇒ 切了没好处，还有通信」。这与 Amdahl 分析（见 `冗余段处理-文献与GR方案.md`）一致：
`P` 小的时候切分无用。

---

## 四、GR 特有的一轨：长序列的上下文并行

这是**唯一一篇专门为 GR 设计切分的论文**，也是与本工作最相关的一篇。

**Scaling Generative Recommendations with Context Parallelism on Hierarchical
Sequential Transducers**（RecSys '25，[ACM](https://dl.acm.org/doi/full/10.1145/3705328.3748143)、
[arXiv:2508.04711](https://arxiv.org/abs/2508.04711)）

### 4.1 动机：序列长度受显存限制

> Recent studies and experiments demonstrate that attending to longer user history
> sequences yields significant metric improvements. However, **scaling sequence
> length is activation-heavy**, necessitating parallelism solutions to effectively
> shard activation memory.

基线（DDP，无激活重算）最多 3K 序列长度（89% 显存）。**加 CP 后到 16,384
（CP=8），即 5.3× 的序列长度提升。**

### 4.2 三个 GR 特有的技术问题

| 问题 | LLM 的做法 | GR 的问题 | 论文的解法 |
|---|---|---|---|
| **jagged 张量** | 序列等长，`[b, s, h]` | 用户交互长度**高度可变**，用 offsets+values 表示 | 支持 jagged 的 CP |
| **AllGather 的峰值显存** | AllGather 沿 batch，余量够 | 中间态被「拼接后的总序列长度」支配 | **换成 AllToAll**，避免每卡持有全量副本 |
| **causal mask 的负载不均** | 有成熟方案（如 TransformerEngine 的 chunk 配对） | 需要按 chunk 重排+自定义 Triton kernel | 每个 chunk 再分 2 份，`(i, 2·CP−1−i)` 配对给 rank i |

数字：AllToAll 替代 AllGather 后**峰值显存降 >60%、吞吐升 >2×**；
负载均衡再提 37%。

### 4.3 与 Amdahl 分析的对照 —— 这是最有价值的一段

论文的 Table 2（scaling factor，理想 2.00×）：

| 配置 | Scaling factor |
|---|---|
| 理想 | 2.00× |
| **只 DDP** | **1.6–1.7×** |
| **DDP + CP（cp_size=2）** | **1.33×** |
| **DDP + CP + batch 翻倍** | **1.55×** |

这三行比仿真更硬，且与 Amdahl 分析的方向一致：

1. **单独加 CP 会降 scaling factor**（1.6–1.7 → 1.33，−22%）——
   与我的 `throughput_ratio > 1`（CP 的整机吞吐不如 DP）方向一致。
2. **CP 的价值必须靠「用省下的显存换更大 batch」才能兑现**（1.55×）。
   这正是 `S21` 那条「四个参数换九成收益」的同构逻辑：**切分本身不产生收益，
   只有把释放的资源用起来才产生收益。**
3. **CP 的定位是显存，不是延迟**。5.3× 序列长度是它的真实产出；
   scaling factor 只有 1.55×，而理想是 2.00×。

**这一条可以直接写进报告的并行部分**：不是「我们用 CP 所以更快」，
而是「CP 让我们能把序列从 3K 做到 16K，多出来的显存换成 batch 才换来 1.55×」。

### 4.4 TP 在 GR 上的硬上限：受头数限制

NVIDIA 在给 HSTU 加 TP 时（2025/8 完成）明确记录了这条约束：

> **The TP size is limited by the attention heads.**
>
> — [recsys-examples issue #71](https://github.com/NVIDIA/recsys-examples/issues/71)

原因：TP 切的是 Q/K/V 的输出维，每个头是最小不可分单位。
本框架的 `max_tp_degree(shape)` 把这条约束编码成了一个可用的函数：

```python
max_tp_degree(Shape(heads=4, d_head=16))   # → 4
```

**GR 的头数通常比 LLM 少**（HSTU 类常为 4–16，LLM 常为 32–64），
所以 TP 在 GR 上的可用度远低于 LLM。这是一条容易忽略但会直接卡住方案的约束。

### 4.5 其他已集成的轴

NVIDIA 的 HSTU 训练示例在 2025 年陆续集成了：
DP、TP（2025/8）、SP（2025/11，v25.10）、PP、CP。
推理侧另有：paged KV cache、异步 host KV onload/offload、CUDA graph、
AOTInductor 部署。SID-GR 推理还引入 `ContextKV/BeamKV/BeamPath` 与连续批处理。

> **注意**：SID-GR（semantic ID 的生成式推荐）的推理有一个 LLM 没有的维度——
> **beam**。`BeamKV`/`BeamPath` 说明 beam 之间的 KV 复用是它自己的优化点。
> 这一轴在我的框架里还没有建模。

---

## 五、把文献映射到本框架的七种切分

`hetero/cost.py` 的 `Split` 现有四种：`none` / `cp_q` / `embp` / `tp`；
`执行时间预测/timepred/stage.py` 有 11 种（更细）。对应关系：

| 本框架 | 文献里的对应 | 是否有 GR 专属文献 |
|---|---|---|
| `embp`（切表） | TorchRec RW/CW/TWRW/GS；OPER 细粒度分片 | **有**（2.1–2.3） |
| `cp_q`（切 Query，KV 留整段） | HSTU Jagged CP；Ring Attention | **有**（4） |
| `tp` | Megatron TP；HSTU TP（受头数限制） | 有（4.4） |
| `pp` | GPipe / 1F1B / interleaved | 通用，无 GR 专属 |
| `ep` | GShard；MTGenRec 的 MMoE | 有（第三节 ④） |
| `shard_r` / `shard_t`（按组/组内 token 切） | **无直接对应** | **空白** |
| `none`（不切，DP 批处理） | DP；MTGenRec 稠密侧 | 有（第三节） |

**两处空白**：

1. **按组切（`shard_r`）与按组内 token 切（`shard_t`）** 是 LONGER 类
   「分组/层级注意力」特有的切法（R=⌊L/T⌋、每组长 T），文献里没有专门讨论。
   这是 GR 结构差异带来的新轴。
2. **段级切分**（同一次前向内每段用不同切分/卡数）没有任何 GR 文献涉及。

---

## 六、与本工作结论的关系

| 本工作结论 | 文献状态 |
|---|---|
| 除注意力外所有段都按 token 可切（`helps()`） | 间接支持：TorchRec 的 TP 对 linear 也切；NVIDIA 集成 TP/SP 到 HSTU 说明这些段确实可切 |
| CP 的吞吐代价 > 1（`amdahl.compare`） | **直接验证**：1.6–1.7 → 1.33（−22%） |
| 切分本身不产生收益，要靠释放的资源 | **直接验证**：CP 单独 1.33×，加 batch 才 1.55× |
| `b` 是 Amdahl 的 (1−P) | 文献未涉及（他们不测 `b`）；这是本工作的增量 |
| TP ≤ 头数 | **直接验证**（issue #71）；已编码为 `max_tp_degree` |
| `shard_r` / `shard_t` 是 GR 新轴 | **空白**，无文献 |

---

## 七、三件可以据此调整的事

1. **把「CP 只为显存」写清楚**。我们此前讨论 CP 时一直隐含「CP 更快」，
   而 RecSys'25 的数字说 CP 单独用会让 scaling factor 掉 22%。
   报告里应改成「CP 换长度，长度换 batch」。
2. **给 `embp` 加访问量维度的均衡**。文献两条独立结论（2.2 的 pooling factor、
   2.3 的 input-dependent）都指出：按容量分片会让某个分片成为关键路径。
   本框架目前按 `table_bytes` 均分，没有建模访问量偏斜。
3. **TP 的可用度要单独提示**。`max_tp_degree` 已给出上限，但 `Split` 的构造
   没有校验它。若方案里 `tp×8` 而 `heads=4`，实际不可行。
   ——这条要不要加进 `Split.__post_init__`，取决于是否接受把 `Shape` 传进去；
   目前先作为文档约束。

---

## 附：引用清单

| 主题 | 来源 |
|---|---|
| TorchRec 五种分片 | [concepts.rst](https://github.com/meta-pytorch/torchrec/blob/main/docs/source/concepts.rst)、[分片教程](https://docs.pytorch.org/tutorials/advanced/sharding.md) |
| 按访问量分片 | [arXiv:2011.02084](https://arxiv.org/abs/2011.02084) |
| 输入相关细粒度分片 | [OPER, ATC'24](https://www.usenix.org/system/files/atc24-wang.pdf) |
| 层次化参数服务器 | [HPS, arXiv:2210.08804](https://arxiv.org/pdf/2210.08804) |
| CPU/GPU 资源池拆分 | [Prism, NSDI'25](https://www.usenix.org/conference/nsdi25/presentation/yang) |
| 稀疏模型并行 + 稠密数据并行 | [MTGenRec, arXiv:2505.12663](https://arxiv.org/abs/2505.12663) |
| GR 的 jagged CP | [RecSys'25, arXiv:2508.04711](https://arxiv.org/abs/2508.04711) |
| TP 受头数限制 | [recsys-examples #71](https://github.com/NVIDIA/recsys-examples/issues/71) |
| HSTU 训练的并行集成 | [recsys-examples HSTU README](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/README.md) |
