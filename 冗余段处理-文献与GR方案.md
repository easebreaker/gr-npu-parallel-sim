# 只有 B 段可并行时，A/C/D 的冗余怎么处理

> 设定（用户提出）：一台机器只跑同类模型，一次前向分 A–B–C–D 四段，
> 只有 B 段被并行加速；用 N 卡并行 B 之后，A、C、D 要在其余 N−1 张卡上重复执行。
>
> 结论有两条，都与直觉不同：
> 1. **文献不「优化冗余」，而是让冗余变小（Megatron SP）或换到无冗余的方案（DP）。**
> 2. **在 GR 上这个冗余几乎不存在** —— 因为除注意力以外，投影、FFN、查表全都
>    按 token 可切。「只有 B 可并行」对 GR 是**悲观假设**。
>    真正不可切的是固定开销 `b`，也就是 `S18` 里那个 90%。

---

## 一、先把「冗余」这件事说清楚

同一个事实有两种解释，结论相反：

| 情形 | 其他 N−1 张卡在做什么 | A/C/D 的重复是 |
|---|---|---|
| **一次只有一个请求**（CP） | 等着 B 算完 | **纯粹浪费**，上界由 Amdahl 给出 |
| **同时有 N 个请求**（DP） | 各自算**不同**请求的 A/C/D | **不存在冗余**，总量随副本线性增长 |

文献里这一点是明说的：DP 的副本「处理不同请求，因此整体吞吐随副本数增加，
但单请求延迟不变」。

所以「如何优化冗余」这个问法本身需要修正成：**什么时候值得付这笔冗余**。
答案只取决于你要的是延迟还是吞吐。

---

## 二、已有工作怎么处理这件事

### 2.1 Megatron 序列并行：把冗余变小（消除显存，不消除计算）

**来源**：Korthikanti et al., *Reducing Activation Recomputation in Large
Transformer Models*, MLSys 2023（[arXiv:2205.05198](https://arxiv.org/abs/2205.05198)）。

它处理的正是「TP 之后还剩下一些被复制的子层」：

> Tensor parallelism ... leaves the layer-norms as well as the dropouts after
> attention and MLP blocks intact and as a result, **they are replicated across
> the tensor parallel group**. These elements **do not require a lot of compute
> but demand a considerable amount of activation memory**.
> Quantitatively, the `10sbh` part of Equation 2 is due to these replicated
> operations and as a result they are not divided by the tensor parallel size `t`.

它的做法是把这些**序列局部**的区域沿序列维切开：`[b, s, h]` → `[b, s/t, h]`，
并在进入 TP 区域前 AllGather、离开后 ReduceScatter。关键的工程细节是
**把这两个集合通信和相邻算子融合**，避免额外开销。

**能借什么**：如果 A/C/D 里有「沿某一维天然独立」的子层，就沿那一维切开，
而不是整段复制。**不能借什么**：它消除的是**显存**冗余；LN 的计算量本来就
可以忽略，所以「计算冗余」在它那里不是问题。我们的 A/C/D 若计算量可观，
这条路不够。

### 2.2 Alpa 的 intra-op pass：允许复制廉价算子

**来源**：Zheng et al., *Alpa*, OSDI 2022（[论文](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)）。

Alpa 对**每个算子**在「复制」与「切分」之间做选择，用 ILP 最小化
「计算 + 通信 + resharding」三项之和。它对复制的态度是明确的：

> For heavy operators such as matmul, we do not allow replicated computation...
> For **lightweight operators such as element-wise operators, we allow replicated
> computation of them, but their computation costs are negligible**.
> To simplify the graph, we merge computationally-trivial operators ...
> into one of their operands and propagate the sharding spec.

**能借什么**：这是判据本身 —— **贵的切、便宜的复制**。
若 A/C/D 的时间占比小，复制它们是对的，不值得为它们引入通信。
若占比大（比如 MTGR 的 `proj` 占 50%），就必须切，不能复制。

### 2.3 Shift Parallelism：动态在 TP 与 SP 之间切换

**来源**：*Shift Parallelism: Low-Latency, High-Throughput LLM Inference for
Dynamic Workloads*（[arXiv:2509.16495](https://arxiv.org/html/2509.16495)）。

它的出发点与本文的问题几乎一致：

> TP is the state-of-the-art method for reducing LLM response latency, however
> GPU communications reduces combined token throughput. On the other hand, DP
> obtains a higher throughput yet is slow in response latency.
> **Best of both worlds does not exist, and it is not possible to combine TP and
> DP because of the KV cache variance across the parallelisms.**

它用 SP（Ulysses 式序列并行）作为中间态 —— SP 有 DP 的性质（无 KV cache 差异），
再在 TP 与 SP 之间**按流量动态切换**：低流量时用 TP 压延迟，高流量时用 SP 保吞吐。
报告的收益是交互负载下响应快 1.51×、批处理吞吐高 50%。

**能借什么**：**「按负载切换并行度」是已被验证的做法**，而且它的判据就是
「延迟 vs 吞吐」。**不能借什么**：它的切换粒度是整个模型（TP↔SP），
不是逐段。

### 2.4 Amdahl 侧的定量框架

**来源**：*Scaling LLM Inference Beyond Amdahl's Limits via Eliminating
Non-Scalable Overheads*（[arXiv:2606.01927](https://arxiv.org/html/2606.01927)）。

它把 TP 的次线性归因于两件事：跨卡集合通信，与**不可并行的部分**：

> According to Amdahl's Law `A = 1/((1−P) + P/t)`, where `P` is the scalable
> portion, the overall speedup is **bounded by the non-scalable fraction (1−P)**,
> which becomes more pronounced as `t` increases.

这就是我们需要的公式。它给的实测旁证也很直接：vLLM 下 `t=8` 的吞吐
比 `t=4` 低 21%。

### 2.5 复制的交换代价

**来源**：*Model Parallelism on Distributed Infrastructure: A Literature
Review*（[arXiv:2403.03699](https://arxiv.org/html/2403.03699)）。

> Intra-operator parallelism's challenge is in its extreme communication
> requirements. **The input tensor to the parallelised operator needs to be
> scattered over the devices and the output then needs to be gathered for every
> batch.**

这条是 A/C/D 与 B 交界处的固定成本：每次进出 B 段都要 scatter/gather。
它不随 A/C/D 的时间占比变化，但对短请求可能是决定性的。

---

## 三、GR 上的事实：这个冗余几乎不存在

### 3.1 除注意力外，其余段全都按 token 可切

这是 GR 与「只有 B 段可并行」这一假设的**根本冲突**：

| 段 | 是否按 token 可切 | 依据 |
|---|---|---|
| 查表 | **是**（每个 token 独立查） | `算子复杂度-按算子手册.md` §1 |
| 线性投影 / FFN | **是**（token 维可切） | 同上 §2 |
| 自注意力 | **是**（切 Q，KV 留整段） | 同上 §3 |
| 组内注意力 | **是**（按组切） | 同上 §5 |
| 交叉注意力 | **是**（切 Q 或切 KV） | 同上 §4 |
| **固定开销 `b`** | **否**（kernel 下发与同步） | 同上 §2.4 |

**真正不可切的只有 `b`。** 而 `b` 正是 `S18` 测到的那个量：L=256 时占 92%，
L=16384 时占 12%。

### 3.2 两种假设下的数字完全不同

MTGR，L=8192，8 卡，`ρ=0.83`（实测 cp2→cp8 的有效指数）：

| 假设 | 可并行占比 P | 延迟增益（CP） | 吞吐代价（DP/CP） | Amdahl 上界 |
|---|---|---|---|---|
| **只有 attn 可并行** | 48.7% | **1.67×** | **4.80×** | 1.95× |
| **所有 token 段可并行** | 100.0% | **5.61×** | **1.43×** | — |

左列是用户的假设：花 4.8 倍吞吐买 1.67 倍延迟，**很亏**。
右列是 GR 的真实情形：花 1.43 倍吞吐买 5.61 倍延迟，**很赚**。

**所以对 GR 而言，正确做法不是「处理冗余」，而是「把本来可切的段都切了」——
冗余自然消失。**

### 3.3 三个模型分别站在哪一侧

L=8192，8 卡，只有注意力算可并行（悲观假设）：

| 模型 | 不可并行 | 可并行 | P | 延迟增益 | 吞吐代价 | 结论 |
|---|---|---|---|---|---|---|
| MTGR | 38.0 ms | 36.0 | 48.7% | 1.67× | 4.80× | 勉强，取决于是否要延迟 |
| LONGER | 10.9 ms | 0.36 | 3.2% | **1.03×** | 7.79× | **完全无意义** |
| HyFormer | 2.7 ms | 0.00 | 0.0% | **1.00×** | 8.00× | **不要切** |

LONGER 与 HyFormer 的注意力不是 L²（分别是 L·T 与几乎恒定），
所以在悲观假设下 CP 连 3% 的延迟都买不到，却要付 8 倍吞吐。
**这两个模型应当直接判死不切。**

---

## 四、迁移到 GR 的具体做法

### 4.1 判据：用 Amdahl 上界，它是预测器直接给的量

```
Amdahl 上界 = T_dp / (不可并行部分)
            = (a + b + c + d) / (b_fixed + 段间规约)
```

其中 `b_fixed` 就是 `S18` 的 `b`。**这个量我们已经能测**（层数倍比法，
见 `复杂度测定-原理方案.md` 的 E1a）。所以判据不需要新的测量：

| Amdahl 上界 | 结论 |
|---|---|
| < 1.3× | 不切。付的吞吐代价远超延迟收益 |
| 1.3–2.0× | 只在延迟确实超 SLO 时切 |
| > 2.0× | 切，且优先把其余可切段一起切掉 |

按 MTGR 的表：L=4096 时 1.48×、L=8192 时 1.95×、L=16384 时 2.90×。
**分界大约在 L≈5000。**

### 4.2 按请求决定，而不是按模型决定

这是 GR 相对 LLM 的**独有能力**。LLM 的 decode 每步都是 1 个 token，
所有请求的「可并行占比」几乎相同，所以只能按模型/按部署决定。
GR 的长度跨 64 倍（256–16384），同一个模型上：

- L=256：Amdahl 上界 1.26×，且整条请求只要 1.4 ms —— **延迟根本不是问题，不切**
- L=16384：Amdahl 上界 2.90×，请求要 220 ms —— **切**

而且这个判断**只需要预测器的输出**（每段的耗时），不需要真实执行。
**这就是把时间预测与调度决策接起来的地方。**

### 4.3 三档策略

| 档 | 条件 | 做法 |
|---|---|---|
| **不切（DP）** | Amdahl 上界 < 1.3，或单请求延迟已满足 SLO | 每卡一个请求，无冗余、无通信 |
| **全切（CP）** | Amdahl 上界 > 2.0，且所有段都可切 | 逐段用 CP，冗余只剩 `b` |
| **折中** | 中间地带 | 只切注意力和投影，查表留在本地（避免 All-to-All） |

第三档是 Alpa 的「贵切便宜复制」原则的具体化：`b` 与段间规约复制，
大段切开。

### 4.4 与我们已有结论的衔接

| 已有结论 | 与本文的关系 |
|---|---|
| `S18`：`b` 占 L=256 的 92% | **`b` 就是 Amdahl 的 (1−P)**。所以「`b` 随 N 上升」意味着「P 随 N 下降」——加卡让 CP 越来越不划算 |
| `S17`：小长度上切得越多越慢 | 同一件事的另一面：`P` 小到负数收益 |
| `段级交错`：短请求有 1.67× 空间 | 那个收益来自「短请求用不满 8 卡」；本文说明**短请求本就不该用 8 卡** |
| `共置配对`：一算力一访存可互补 | 若一台机器只跑同类模型，配对不成立；**本文的 DP 批处理是它的替代** |

---

## 五、结论

1. **文献没有「优化 A/C/D 冗余」的方案**，因为它们不认为那是冗余：
   Megatron SP 把冗余（显存）变小；Alpa 规定「廉价算子可以复制」；
   Shift Parallelism 干脆换到 DP，让冗余不存在。
2. **GR 上这个冗余本来就很小**：除注意力外所有段都按 token 可切。
   真正不可切的只有固定开销 `b`，而 `b` 是 `S18` 已测的量。
3. **所以正确的动作不是「处理冗余」，是「把可切的全切了」**。
   只有注意力可切时，MTGR 是 1.67× 延迟换 4.80× 吞吐（亏）；
   全切之后是 5.61× 延迟换 1.43× 吞吐（赚）。
4. **LONGER 与 HyFormer 应直接判死不切**（P 为 3.2% 与 0%）。
5. **判据是 Amdahl 上界，它由预测器直接给出**，因此可以**按请求**决定切不切 ——
   这是 GR 相对 LLM 的独有能力，也是本文最值得写的一点。

---

## 附：可复现

```bash
cd 异构并行仿真
python3 -m unittest discover -s tests
```

模块：`hetero/amdahl.py`
（`stages_of` / `compare` / `breakeven_cards` / `serves_latency`）。

`compare()` 返回的 `latency_gain` 与 `throughput_ratio` 是一对可比的数：
前者是 CP 相对 DP 的单请求加速，后者是 DP 相对 CP 的整机吞吐倍数。
两个数一起看，才能判断这笔冗余值不值得付。
