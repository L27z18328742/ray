# Ray 与 slime 源码导读


---

源码提交：`5bae5bb7928d65906e24e99f5b5d3f99a4daaf93`。全部课程已展开。


---

# 先看一轮训练怎样发生

> 阅读目标：先能说出“谁指挥、谁生成、谁训练、谁传数据”，再打开具体类。状态：已展开。

你已经见过 `@ray.remote`，却未必实际启动过多个进程。直接读 `slime/ray`，很容易把 GPU 调度、Python 对象、分布式训练和异步请求混成一件事。这本教程从一个普通函数开始，每章解决一个具体问题，最后回到完整的 RL 训练循环。

## 我们到底要读懂什么

想象你有四张 GPU：两张负责让模型回答问题，另两张负责用这些回答训练模型。一轮工作是：取一批问题 → 生成回答并打分 → 整理训练数据 → 更新训练模型 → 把新权重交给推理模型。下一轮才能用更新后的策略继续回答。

GPU 不会自己协调这条流程。slime 使用 Ray 把有状态的训练进程和推理控制进程安排到合适位置，并由 driver 发出命令。训练进程内部的前向、反向和梯度通信仍由 PyTorch / Megatron 执行；生成请求则交给 SGLang 服务。

**读懂设计的标准**：看到一次调用，你能说出调用者在哪个进程，参数是什么，返回的是句柄、引用还是具体值，哪一行在等待，以及真正的计算发生在哪里。

## 全书使用的源码版本

本次实际读取的是 THUDM/slime 的提交 **`5bae5bb7928d65906e24e99f5b5d3f99a4daaf93`**，提交日期 2026-09-19，读取日期 2026-09-20。`main` 会继续变动，书内源码链接和本地快照都固定到这个提交。本地读取的 checkout 无改动。

全书使用三种证据标记：

| 标记 | 含义 |
|---|---|
| **源码事实** | 已实际读取该提交的相应实现，链接可定位到文件与符号 |
| **设计理解** | 根据实现解释动机与取舍，不冒充作者声明或性能结论 |
| **教学简化** | 为学习而写的小例子或图示，不是 slime 的原实现 |

浏览器交互是教学模型。CPU 脚本的运行结果见[实战章](webapp/chapters/ch13.html)和[验证记录](webapp/chapters/sources.html)；本教程没有在 GPU 集群上运行完整 slime，也没有测量训练性能。

## 先把分支固定下来

第一遍只看一条容易跟踪的路线：Megatron 训练后端、默认 SGLang rollout、训练与推理使用不同 GPU、没有 critic、没有 offload/release、没有外部推理服务。样本使用 `object-store`，权重使用默认的 `full + nccl` 路线。

这是**阅读配置**，不是可直接复制启动大模型的完整命令。模型结构、checkpoint、数据路径和分布式参数仍需实际配置。第 10、11 章再把省显存和异步分支放回来。

## 一张进程地图

```text
driver：train.py
 ├─ RayTrainGroup（driver 内普通对象）
 │    ├─ MegatronTrainRayActor rank 0 ─┐
 │    └─ MegatronTrainRayActor rank 1 ─┴─ PyTorch/Megatron 通信
 └─ RolloutManager（Ray Actor，CPU）
      ├─ 数据源与生成函数 → HTTP router → SGLang server
      ├─ SGLangEngine（Ray Actor）→ 启动并控制服务子进程
      └─ Lock（Ray Actor）→ 权重更新时协调访问

样本：manager → Ray 对象存储 → 训练 worker
权重：训练 worker → 独立的权重传输路径 → SGLang
```

图中箭头有三种含义：Python 直接调用、Ray 远程方法、HTTP / 分布式通信。后面读代码时，要先确认箭头类型。`SGLangEngine` 这个包装类不等于服务内部的所有 GPU worker；图中也没有把每个子进程画全。

## 主链与必读顺序

整条主链是：**启动 driver → 预留资源 → 创建 rollout manager → 创建训练 worker → 初次同步权重 → 生成 → 训练 → 同步 → 下一轮 → dispose**。

1. [train.py → train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)：输入 `args`；把资源、模型与 manager 连起来；驱动训练循环。
2. [create_placement_groups](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L120)：输入规模与部署开关；输出每个角色的 PG 与 GPU 索引映射；交给创建流程。
3. [RayTrainGroup.create](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L188)：创建 actor handles，等待初始化，连接 rollout manager；输出恢复后的起始 rollout 编号。
4. [RolloutManager.generate](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L176)：输入轮次；生成样本、转训练格式、分 DP；返回引用包装的列表。
5. [MegatronTrainRayActor.train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L386)：输入这批引用；取数据并训练；策略训练分支返回 `None`，driver 用完成引用等待它结束。

选读：[权重更新选择器](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/__init__.py#L10)。暂缓阅读模型损失公式、MoE 路由回放和多模态处理；本教程只追踪它们与编排层的接口。

## 怎么读这本书

第 1–3 章让 `remote / get / wait / Actor` 变成实际经验；第 4–9 章沿源码走通一轮；第 10–12 章解释分支和失败；第 13 章把它们组装成一个小系统。单章约 20–30 分钟，学完先尝试回答章末问题，再继续。

**停下来想一想：如果两张训练 GPU 上的梯度需要 all-reduce，直接负责这项计算的是 Ray，还是训练后端？你会去哪个文件继续追踪？**


---

# 第一次 remote：任务与结果引用

> 阅读目标：分清“任务已提交”“结果已准备好”和“结果已取回”。状态：已展开。

## 从一个本地函数开始

普通 `square(3)` 在当前 Python 进程执行，返回 `9` 后才继续下一行。加上 `@ray.remote`，调用方式变成 `square.remote(3)`：Ray 安排 worker 执行，调用方拿到的是 **ObjectRef**。它是远程结果的引用，不是结果本身，也不是新开出来的线程。

下面是完整的 CPU 教学例子，不需要安装 slime：

```python
import os
import ray

@ray.remote
def square(x):
    return {"value": x * x, "worker_pid": os.getpid()}

if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False)
    try:
        refs = [square.remote(x) for x in [2, 3]]
        print("driver:", os.getpid())
        print("引用类型:", type(refs[0]).__name__)
        print("结果:", ray.get(refs))
    finally:
        ray.shutdown()
```

`.remote()` 返回并不要求 worker 已经开始运行。资源不足时，任务可能还在等调度。`ray.get(ref)` 会等待对应结果准备好，并把结果交给当前进程；失败也可能在这里以异常形式出现。

## 按时间走一遍

```text
driver       提交 square(2) → 得到 ref0
             提交 square(3) → 得到 ref1
             ray.get([ref0, ref1]) ────────── 等待 → [结果0, 结果1]
worker A            计算 2² → 完成
worker B                计算 3² → 完成
```

这是示意时间线，任务何时开始、是否使用两个 worker 由资源与调度决定。`ray.get([ref0, ref1])` 的结果顺序与输入引用顺序一致，不按谁先完成排列。

`num_cpus=2` 声明给本地 Ray 的逻辑 CPU 资源。它不是“只允许 Python 使用两个操作系统线程”的隔离规则。这个小程序只验证远程执行与引用语义，不用启动时间推算分布式训练加速比。

## ObjectRef 可以表示成功，也可以表示失败

如果远程函数抛错，`.remote()` 通常已经返回引用；到 `ray.get()` 时调用方才拿到任务异常。因此日志中的“submitted”只能证明提交动作，不证明业务完成。

```python
@ray.remote
def fail():
    raise ValueError("这是教学用的异常")

ref = fail.remote()
# ray.get(ref) 将抛出 Ray 包装的任务异常
```

如果需要“有结果再继续”，必须建立依赖或等待。只把变量命名为 `result` 并不会改变它的类型。

## 回到 slime：同样的语法，不同的工作量

主链：**driver 调用 → manager 的远程方法 → ObjectRef → driver 等待 → 方法返回值**。

1. [train.py → train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)：找到 `rollout_manager.generate.remote(rollout_id)`。输入轮次，执行者是另一个进程中的 manager。
2. [RolloutManager.generate](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L176)：看方法末尾，返回 `_split_train_data_by_dp(data)` 的结果。
3. 回到 `train.py` 的 `ray.get(...)`：它等待 **generate 这次方法调用** 完成。第 7 章会解释为什么这还不等于 driver 取回整批训练张量。

选读：[Ray 官方 Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)，用于核对任务与远程执行的基础接口。暂缓深入 `generate()` 的数据清洗；先认出方法返回值和那层包在外面的 ObjectRef。

## 动手观察

第 13 章给出了统一环境准备。运行 [01_tasks.py](examples/01_tasks.py)，查看 driver 与 worker 的 PID、引用类型和计算结果。不要预设两个任务一定使用不同 worker；**worker PID 与 driver PID 不同**才是此实验的核心观察点。

接着把“先提交全部任务，再 get”改成“循环里逐个提交并 get”，预测时间线如何变化。第 3 章会解释其中的等待边界。

**理解题：拿到一个 ObjectRef 时，能否确定任务已经开始？能否确定计算成功？分别还缺什么证据？**


---

# Actor：让模型留在一个进程里

> 阅读目标：理解“远程对象长期保留状态”，以及为什么训练模型适合放进 Actor。状态：已展开。

## 每次都加载模型，代价太大

假设普通远程任务每调用一次就加载 checkpoint、构建 optimizer，然后训练一个 batch。这会反复付出初始化成本。我们更希望初始化一次，之后把多个命令送到同一个持有模型状态的进程。

Ray Actor 提供的就是这种有状态的远程对象。`Counter.remote()` 返回 **ActorHandle**；`counter.add.remote()` 返回这次方法调用的 **ObjectRef**。两者用途不同：一个用来继续发命令，一个用来等待某次结果。

```python
@ray.remote(num_cpus=1)
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n):
        self.value += n
        return self.value

a = Counter.remote()       # ActorHandle，不是本地 Counter 实例
refs = [a.add.remote(2), a.add.remote(3)]
assert ray.get(refs) == [2, 5]
```

这段代码是[02_actors.py](examples/02_actors.py)中的核心。它假设同一调用方对默认同步 Actor 顺序提交方法，没有启用并发或异步 Actor 的特殊设置。不要把这个顺序结论套到所有 Actor 类型和多提交者场景。

## 谁拥有状态

```text
driver 里的 a（句柄） ── add.remote(2) ──→ Actor 进程
                                          self.value: 0 → 2
                     ←── ObjectRef ────── 本次调用的返回值
```

driver 不直接读取远端 `self.value`。它通过方法取得值。再创建 `b = Counter.remote()`，`b` 有自己独立的计数器状态；`a` 增加到 5，不会让 `b` 也变成 5。

**设计理解**：常驻训练 Actor 适合保存模型、optimizer、并行组等昂贵状态。代价是生命周期更长，进程故障与显存释放需要明确处理。Actor 重建也不自动恢复内存中的 optimizer；恢复语义要看 checkpoint 和业务实现。

## slime 中有三个容易混淆的 actor

| 名称 | 实际含义 | 位于哪里 |
|---|---|---|
| RL 中的 actor | 被训练的策略模型角色，与 critic 相对 | 业务概念 |
| `RayTrainGroup` | 持有多个训练 ActorHandle 的普通 Python 对象 | driver 进程 |
| `MegatronTrainRayActor` | 实际保存模型并执行训练的 Ray Actor 实现 | 各训练 worker 进程 |

`actor_model.async_train(...)` 是一次本地方法调用。这个方法再遍历 `_actor_handlers`，逐个执行 `actor.train.remote(...)`。认出这个两层关系，后面就不会误以为每一层都是远程进程。

## 主链与阅读顺序

触发：需要创建训练模型。主链是 **create_actor_model → allocate_train_group → RayTrainGroup.create → 包装远程类 → 创建 handles → init.remote**。

1. [allocate_train_group](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L140)：输入规模、PG 与模型角色，直接返回一个 `RayTrainGroup`。
2. [RayTrainGroup.__init__](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L32)：保存配置与空 handle 列表，这里尚未加载模型。
3. [RayTrainGroup._allocate_gpus_for_actor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L57)：把实现类交给 `ray.remote(...)`，再 `.options(...).remote(...)` 创建每个 worker。
4. [RayTrainGroup.async_train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L131)：把同一轮训练命令发送给所有 handles，返回引用列表给 driver。

选读：[Ray 官方 Actors](https://docs.ray.io/en/latest/ray-core/actors.html)。资源与 rank 的细节暂缓到第 4、5 章；本章只追“本地对象如何持有远端状态的入口”。

## 默认串行和多个 Actor 并行

对同一个默认同步 Actor 连续提交两个耗时方法，它们通常排队执行。创建两个 Actor 再分别提交，则有机会并行执行，但还受可用资源、共享服务和实际阻塞影响。

因此“用了 Actor”不等于“方法都并发执行”。slime 中多个训练 worker 要一起工作，是因为它创建了多个 Actor，并由训练后端进行分布式协作，而不是把单个 Actor 变成无数 GPU 线程。

**理解题：`actor_model`、`actor_model._actor_handlers[0]`、`actor_model.async_train(...)` 的返回值分别是什么类型的对象？哪个对象真正持有 GPU 上的模型？**


---

# 等待、依赖与背压

> 阅读目标：看到“async”时，能沿调用方找到实际等待位置。状态：已展开。

## 先提交，再等待

下面两个循环结果相同，但调度机会不同。它们是教学简化：

```python
# A：提交一个，就等待一个
values = [ray.get(work.remote(i)) for i in range(4)]

# B：先提交所有工作，再等待所有结果
refs = [work.remote(i) for i in range(4)]
values = ray.get(refs)
```

A 中下一个任务要等前一个结束才被提交。B 让 Ray 提前知道四项工作，资源允许时可以同时运行。B 并不保证固定加速比：任务太小、CPU 资源不足或都调用同一个串行服务，都可能限制并行。

## get 和 wait 问的问题不同

| 操作 | 提出的问题 | 返回 |
|---|---|---|
| `ray.get(ref)` | 这次调用的结果是什么？ | 具体值，或异常 |
| `ray.get(refs)` | 这些调用的结果分别是什么？ | 按输入顺序排列的值 |
| `ray.wait(refs, num_returns=1)` | 哪些引用已经就绪？ | `(ready_refs, remaining_refs)` |
| `ray.put(value)` | 如何把已有数据变成可传递的引用？ | ObjectRef |

`wait` 不替你处理远程计算异常。一般还要对就绪引用执行 `get`。带 `timeout` 的 `wait` 可能返回空的就绪列表；超时不会自动取消剩余任务。

## 引用还能表达依赖

```python
@ray.remote
def double(x):
    return x * 2

@ray.remote
def plus_one(x):
    return x + 1

x_ref = double.remote(10)
y_ref = plus_one.remote(x_ref)
assert ray.get(y_ref) == 21
```

作为**顶层参数**传入的 ObjectRef 会在执行前被 Ray 解引用。`plus_one` 收到的是 `20`，调度器会等依赖就绪；driver 无需先取回 `20` 再转发。

如果引用藏在列表或包装对象里，规则不同：被调用者可以收到引用本身，自行选择何时取数。第 7 章的 `Box` 就与这种“传引用而非立即取值”的边界有关。[03_objects.py](examples/03_objects.py)会同时验证顶层引用与嵌套引用。

## slime 中的 async 是命名约定

主链：**driver → RayTrainGroup.async_train → 多个 actor.train.remote → 引用列表 → driver 的 ray.get**。

1. 读 [RayTrainGroup.async_train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L131)：它是 `def`，不是 `async def`；返回 Python 列表，每项是一个远程方法引用。
2. 回到 [train.py → train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)：无 critic 分支立即执行 `ray.get(actor_model.async_train(...))`；driver 等所有训练 worker 的调用完成。
3. 往后看 `actor_model.update_weights()`：这个普通方法内部也有 `ray.get`。外面没有显式 `get`，不代表它不等待。

选读 [RayTrainGroup.update_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L162)。暂缓 fully-async rollout，那是第 11 章的另一层并发机制。

## 为什么还需要背压

假设生成任务每秒提交 100 个，处理端每秒只能完成 10 个。CPU 数量虽然限制了正在执行的任务，待执行任务和结果仍可能不断堆积。背压的做法是在未完成数量达到上限时暂停提交，先消费完成结果。

```python
pending = []
for item in inputs:
    if len(pending) >= limit:
        ready, pending = ray.wait(pending, num_returns=1)
        consume(ray.get(ready))
    pending.append(work.remote(item))
consume(ray.get(pending))
```

这是对“在途任务数量”的限制，不是内存字节数保证，也不是精确的吞吐率限制。运行 [06_backpressure.py](examples/06_backpressure.py)可以观察记录的最大 pending 数量。对应 Ray 官方说明是[限制待处理任务](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html)。

**理解题：如果把函数改名成 `async_train`，但调用方仍然立刻 `ray.get(...)`，能否说明 rollout 与训练重叠了？你还需要查看哪一层？**


---

# 先安排座位：Placement Group

> 阅读目标：解释为什么“训练两卡、rollout 两卡”在只有三卡可用时，会卡在资源预留阶段。状态：已展开。

## 四个进程不该各自抢座位

假设训练需要两张卡，推理也需要两张卡。如果两个角色各自启动并占住部分资源，可能出现“各拿到一点，但谁都凑不齐”的困境。Placement Group（下文简称 PG）允许先声明一组资源 bundles，让调度器整体安排。

一个 bundle 是一份资源需求，例如 `{"GPU": 1, "CPU": 1}`。它是调度上的资源包，既不是一个进程，也不是一个 CUDA stream。一组 PG 有多个 bundle，后续 Actor 可以指定使用其中哪一个。

## 从配置算出整组需求

本章固定：`actor_num_nodes=1`、`actor_num_gpus_per_node=2`、`rollout_num_gpus=2`，不启用 colocate / external / debug。

在 [_get_placement_group_layout](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L100) 中：

```python
actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
# 简化：只保留普通、分卡部署的最后一条分支
return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus
```

因此总数为 4，rollout 的逻辑起点是 2。在 [create_placement_groups](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L120) 中，`actor` 与 `rollout` 都携带**同一个 PG**，只是 rollout 的索引数组从 offset 开始切片。`actor` 保存的索引数组没有在这里裁成两项；训练组按自己的 `world_size=2` 只消费前两项。

## PACK、ready 与一个重要等待点

主链：**train → create_placement_groups → _create_placement_group → pg.ready → InfoActor 探测 → 返回映射**。

1. [train.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)：先申请 PG；函数成功返回后，才开始调用创建 rollout manager 的代码。
2. [_create_placement_group](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L42)：输入 GPU 总数，构造四个 1 GPU + 1 CPU bundle，调用 `placement_group(..., strategy="PACK")`。
3. 同一函数等待 `pg.ready()`。资源未满足时每 30 秒记录总量和可用 GPU 数，然后继续等待。
4. 资源满足后创建临时探测 Actor；返回 `(pg, reordered_bundle_indices, reordered_gpu_ids)`，给后续创建流程使用。

`PACK` 表示尽量集中到较少节点；**不要求整个 PG 必须放在同一个节点**。单个 bundle 要能放入一个节点，整组还必须满足 CPU 等资源要求。四张 GPU 的总量只是必要条件之一。

```python
ready_ref = pg.ready()
while not ray.wait([ready_ref], timeout=30)[0]:
    # 源码在这里记录资源与等待时长
    ...
```

这里 30 秒是观察间隔，不是总超时。若四卡 PG 当前无法调度，driver 会留在这里；不会先跳出去开始两卡训练。Ray 官方的[Placement Groups 文档](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)说明了初次创建的整组调度语义；已创建后节点失效的恢复阶段则有不同边界，不能笼统说任何时候都是原子操作。

## 为什么先创建 InfoActor 又立刻杀掉

bundle 编号并不等于“节点上的物理 GPU 编号”。Ray 的实际调度结果可能与列表顺序不同。slime 在每个 bundle 上启动 [InfoActor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L16)，查询节点 IP 和 `ray.get_gpu_ids()[0]`，随后 `ray.kill` 这些临时 Actor。

接着按节点标识和 GPU ID 排序，形成逻辑顺序。假设探测结果如下：

| 原始 bundle | 探测位置 | 排序后的逻辑位置 |
|---|---|---|
| 0 | 节点 B / GPU 1 | 3 |
| 1 | 节点 A / GPU 0 | 0 |
| 2 | 节点 B / GPU 0 | 2 |
| 3 | 节点 A / GPU 1 | 1 |

那么 `reordered_bundle_indices = [1, 3, 2, 0]`。训练 rank 0、1 使用原始 bundle 1、3；rollout 用排序后的后两项。这个表是教学示例，实际放置要看探测日志。

**设计理解**：排序让上层能按节点和卡号解释 rank / engine 的分布。但排序只是在已调度结果上建立映射，不会重新搬动 GPU，也不是对网络拓扑的性能优化证明。

## 资源 ready 不等于模型 ready

把四种状态分开：PG 资源预留完成 → Actor 进程可接受调用 → 分布式进程组初始化完成 → checkpoint / optimizer 初始化完成。`ready_ref` 只证明第一项；后面还需要模型 `init.remote()` 与等待。

选读 [sort_key](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L21)。本地可运行 [04_placement.py](examples/04_placement.py)，用 CPU bundles 验证可调度与不可调度两种情况；这不能验证 GPU 编号、跨节点网络或 NCCL。

**理解题：第 4 张 GPU 注册成功，`pg.ready()` 就绪了。这一刻是否能说模型已加载？沿 `train.py` 指出还没执行的步骤。**


---

# 把训练 worker 一个个启动起来

> 阅读目标：追踪 rank 0 的地址如何到达其他 worker，并分清 Ray PG 与 PyTorch process group。状态：已展开。

## 已经有资源，下一步才是创建进程

继续两张训练卡的例子。`world_size = num_nodes × num_gpus_per_node = 2`，所以要启动两个训练 Ray Actor。每个 Actor 长期持有自己的模型分片或副本、optimizer 状态和分布式通信状态。

本章主链：**create_training_models → create_actor_model → RayTrainGroup.create → _allocate_gpus_for_actor → 构造所有 Actor → init 所有 Actor → 连接 manager**。

## 先看四个初始化参数

| 参数 | 两 worker 示例 | 作用 |
|---|---|---|
| `world_size` | 都是 2 | 训练进程组总进程数 |
| `rank` | 0、1 | 当前进程在训练组中的全局编号 |
| `master_addr` | rank 0 所在节点地址 | 进程组初始化的会合地址 |
| `master_port` | rank 0 查找的空闲端口 | 配合地址完成会合 |

这里的 master 与 Ray head 不是同一个概念。Ray 集群的控制节点负责 Ray 服务；训练的 `MASTER_ADDR` 是此组训练 worker 建立 PyTorch 通信时使用的地址。

## rank 0 为什么先被询问

阅读 [RayTrainGroup._allocate_gpus_for_actor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L57)。先选择默认 `MegatronTrainRayActor` 实现，再包装成远程类。创建循环的核心可以简化为：

```python
master_addr, master_port = None, None
for rank in range(world_size):
    actor = TrainRayActor.options(...).remote(
        world_size, rank, master_addr, master_port
    )
    if rank == 0:
        master_addr, master_port = ray.get(
            actor.get_master_addr_and_port.remote()
        )
    self._actor_handlers.append(actor)
```

rank 0 的构造函数发现没有 master 地址，就选择自己节点的 IP 与空闲端口。driver 等待拿到它，再把同样的地址端口传给 rank 1。**此时还没有让 rank 0 单独完成全部分布式初始化。**否则它可能一直等尚未创建的同伴。

随后 `create()` 才对所有 worker 提交 `init.remote()`，一次性 `ray.get` 这组引用。这种“先创建所有成员，再同时初始化”对需要互相等待的分布式系统很重要。

## 从 Ray 的 GPU ID 到训练设备

[TrainRayActor.__init__](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/train_actor.py#L26)设置 `MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK / LOCAL_RANK`。本版本通过 `get_local_gpu_id()` 调用 accelerator 的可见设备映射，不应简单把 `rank` 当本机 GPU ID。

训练 Actor 的 runtime env 设置了一组 `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` 开关。读这个项目时，要同时查看 Ray 分配的 GPU ID、进程可见设备和显式设备选择，不能直接套“每个 Actor 都只能看到一个 cuda:0”的直觉。

资源也有两层：PG bundle 预留的是 1 GPU；[allocate_train_group](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L140)实际传入 `num_gpus_per_actor=0.4`，worker 的 `.options()` 覆盖了远程类上初始的 1 GPU 声明。这是逻辑资源记账，不是显存的 40% 硬上限。第 10 章专门解释。

## 进入真正的分布式训练初始化

1. [RayTrainGroup.create](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L188)：输入 manager handle，创建 worker 并等待 `init`，返回起始轮次列表。
2. [MegatronTrainRayActor.init](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L64)：调用父类，再初始化 Megatron 和模型 / optimizer，得到 checkpoint 对应的起始轮次。
3. [TrainRayActor.init](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/train_actor.py#L47)：设置实际设备，调用 `dist.init_process_group`，并初始化 Gloo 辅助组。
4. [TrainRayActor.set_rollout_manager](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/train_actor.py#L123)：保存 manager handle；训练 rank 0 把 `train_parallel_config` 发给 manager，告诉它后续按多少个 DP 分片准备数据。
5. [create_training_models](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L186)：要求返回的起始轮次一致，必要时恢复数据源进度，最终交还模型组给 driver。

## 两种 group 不是同一个东西

| 名称 | 管理什么 | 代表操作 |
|---|---|---|
| Ray Placement Group | 一组资源的放置与预留 | `placement_group(...)` |
| PyTorch Process Group | 一组训练进程之间的通信 | `dist.init_process_group(...)` |
| slime RayTrainGroup | driver 里的 worker handle 集合 | `async_train / update_weights` |

选读 [RayActor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/ray_actor.py#L4)，它只提供地址、端口等小工具，本身没有 `@ray.remote`。暂缓 Megatron 并行组内部算法；这里以“初始化完成后可以一起训练”为接口边界。

**理解题：为什么源码先逐个创建 Actor，再对所有 Actor 提交 `init.remote()`？如果每创建一个就立刻等待它的 `dist.init_process_group()` 完成，会有什么风险？**


---

# RolloutManager 如何组织生成

> 阅读目标：从 driver 的一次 generate 调用，追到 SGLang 收到 HTTP 请求。状态：已展开。

## manager 不是用 CPU 执行大模型推理

[create_rollout_manager](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L227)给 `RolloutManager` 设置 `num_cpus=1, num_gpus=0`。它负责读取数据、组织请求、等待结果和转换训练格式；GPU 推理由它管理的服务执行。

manager handle 返回也不意味着服务已初始化。Actor 构造过程包含启动服务和等待初始化；后续需要构造完成的远程方法会受到这个顺序约束。若要从日志判断状态，应区分“handle 已拿到”与“引擎健康检查已通过”。

## 启动服务：Ray 之外还有子进程

主链：**RolloutManager.__init__ → start_rollout_servers → ServerGroup.start_engines → SGLangEngine.init → 服务子进程 → 健康检查与路由注册**。

1. [RolloutManager.__init__](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L45)：输入配置和 rollout PG，调用服务部署函数，保存 pending init handles。
2. [start_rollout_servers](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/deployment.py#L79)：解析模型 / server group 配置，启动 router，并逐组创建引擎，返回服务描述与初始化引用。
3. [ServerGroup.start_engines](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/engine_group.py#L63)：按 PG 映射选择 base GPU，为 SGLangEngine 创建 Ray Actor、分配端口并提交 `init.remote()`。
4. [SGLangEngine.init](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/sglang_engine.py#L105)与[launch_server_process](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/sglang_engine.py#L31)：配置服务参数并启动真正的 SGLang server 子进程。普通 node 0 路径会等待服务健康。
5. manager 对初始化引用执行 `ray.get`，再建立锁与可选健康监控等状态。

同节点的一个多 GPU 推理引擎可以由一个 Ray 控制 Actor 启动服务并管理多张 GPU。跨节点引擎又会有每节点控制实例。**训练的“一卡一个训练 rank”与推理的“一个 engine 几张卡”不能机械地一一对应。**

`placement_group_capture_child_tasks=True`针对 Ray 子任务 / 子 Actor 的放置传播；它不是操作系统 `multiprocessing.Process` 的 GPU 隔离器。实际服务设备配置仍由 SGLang 启动参数决定。

## 生成函数可以换，先确认默认绑定

manager 通过 `load_function(args.rollout_function_path)`载入生成函数。不是看到 `generate_rollout` 这个成员名就能判断执行目标，必须继续找参数默认值。

本版本[参数声明](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/arguments.py#L339)默认绑定到 `slime.rollout.sglang_rollout.generate_rollout`，数据源默认是 `RolloutDataSourceWithBuffer`。因此同步主线可以确切追到下面的实现；如果用户配置覆盖了路径，应重新读取自定义函数。

## 一批问题怎样变成 HTTP 请求

主链：**driver generate.remote(id) → manager.generate → _get_rollout_data → call_rollout_fn → generate_rollout → generate_rollout_async → generate_and_rm_group → generate → HTTP /generate**。

1. [RolloutManager._get_rollout_data](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L286)：调用已绑定函数，取得 samples 与 metrics，再展开嵌套样本。
2. [generate_rollout](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/sglang_rollout.py#L668)：普通训练分支运行异步生成过程，必要时把中断样本放回数据 buffer。
3. [generate_rollout_async](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/sglang_rollout.py#L415)：从数据源取 prompt groups，提交生成任务，用 `asyncio.wait(...FIRST_COMPLETED)`收集完成的组，达到目标后处理剩余请求。
4. [generate_and_rm_group](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/sglang_rollout.py#L329)：组织同一 prompt 的多个 sample，以及生成 / reward 路径。
5. [generate](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/sglang_rollout.py#L154)：准备 token、采样参数和 payload，向 router 的 `/generate` 发 HTTP 请求，再把 token 与 log probability 等结果写回 Sample。

注意第三步是 **Python asyncio**，不是 `ray.wait`。manager 是同步 Ray Actor，不妨碍它在某次方法内部运行 asyncio 来并发等待许多 HTTP 请求。默认一个批次内部请求并发，也不等于这个批次与训练阶段重叠。

## 生成完成后还没开始训练

`RolloutFnTrainOutput`包含 samples 和 metrics；manager 后面还要记录信息、转换训练字段、按 DP 分片。driver 得到的是下一章解释的 `list[Box]`。

选读 [call_rollout_fn](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/base_types.py#L19)，它还兼容旧格式输出。暂缓 external rollout、多模型 router、PD / EPD 分离；本章解释的是普通本地管理的 SGLang 部署。

**理解题：`RolloutManager`没有申请 GPU，为什么它仍然能触发 GPU 上的生成？请分别画出 Ray 调用、普通 Python 调用和 HTTP 请求。**


---

# 一批数据如何到达训练 GPU

> 阅读目标：解释 `ray.get(generate.remote(...))`之后，为什么 worker 里还有一次 `ray.get`。状态：已展开。

## 先把变量名里的陷阱拿掉

driver 中变量叫 `rollout_data_ref`，但在默认路线下，它不是单独一个 ObjectRef，而是 **`list[Box]`**。每个 Box 里面存着一个 DP 分片的 ObjectRef。外层方法结果与内层训练数据是两个层次。

```text
generate.remote(id)
   │ 返回外层 ObjectRef（代表 generate 调用）
   ▼ ray.get
driver 得到 [Box(ref_dp0), Box(ref_dp1), ...]
   │ 作为列表传给每个训练 worker
   ▼
worker 按自己的 dp_rank 选择一个 Box
   ▼ ray.get(box.inner)
CPU 上的训练数据 → 显式搬到训练 GPU
```

图示表达引用传递关系，不声称底层数据零复制或永远不经网络。

## 从 Sample 到训练字段

主链：**generate → _convert_samples_to_train_data → _split_train_data_by_dp → ray.put + Box → worker.process_rollout_data → 搬到 GPU**。

1. [RolloutManager.generate](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L176)：完成生成后调用转换函数，再返回 DP 分片结果。
2. [_convert_samples_to_train_data](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L348)：把 Sample 转成字典，核心字段包括 `tokens`、`response_lengths`、`rewards`、`loss_masks`、`rollout_ids`；可选还有 rollout log probs、多模态输入等。
3. [_split_train_data_by_dp](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L513)：使用训练侧提供的 `dp_size`等并行配置，计算分区和 microbatch 调度，再打包各 DP rank 的数据。

这里要区分训练循环中的 `rollout_id`（这一轮编号）与数据中的 `rollout_ids`（用于样本归组和训练步调度的轨迹身份）。在一条轨迹拆出多个训练 sample 的场景里，它们不能按“样本位置”随意改写。

## DP 分片不是按训练 Actor 数量随便平分

例如训练 world size 为 4，只有 TP=2、DP=2、PP=CP=1。四个训练进程只对应两个 DP 分片；同一 DP rank 的 TP 成员需要使用对应的数据，而不是给四个 rank 各发互不相同的 prompt。

更简单的教学例子：四条样本、DP=2、每个 microbatch 一条、关闭均衡时，可能形成 DP0 `[0,2]` 与 DP1 `[1,3]`。这只是该简化条件下的示意；真实 [build_dp_schedule](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/dp_schedule.py#L82)会考虑轨迹归组、global batch、microbatch、动态长度、均衡以及 VPP 对齐，不能把固定切片当实现。

manager 的分片字典里，`tokens`等字段按 partition 选择；`raw_reward`和 `total_lengths`仍保留全量，其中部分字段在训练侧再次处理。因此“分片”也不意味着字典每一项都已经变成局部数据。

## Box 具体做了什么

[Box](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/misc.py#L143)只有一个 `_inner`字段和 `inner`属性，没有网络传输算法。manager 的默认分支可以概括为：

```python
# 源码中的核心表达式；外围按 DP rank 循环
rollout_data_refs.append(Box(ray.put(rollout_data)))
```

如果把 ObjectRef 直接作为远程方法的顶层参数，Ray 会在调用执行前解引用。包装 / 嵌套结构让程序能够把引用交到目标 worker，由它选择需要的分片再取数。**这里列表本身也是嵌套容器；不能声称只有 Box 才能实现延迟解引用。**

**设计理解**：driver 只需要操纵引用清单，不必先把所有数据物化到自己进程再转交。实际吞吐、跨节点复制、共享内存行为仍需 profiling，不能从用了 `ray.put`就推导出零拷贝或性能收益。

## 最后一次 get 在训练进程

4. [RayTrainGroup.async_train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L131)：把整个引用包装列表传给所有 worker。
5. [process_rollout_data](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/data.py#L304)：检查列表长度等于 DP size，执行 `ray.get(rollout_data_ref[dp_rank].inner)`；按 partition 处理长度与局部 reward。
6. [MegatronTrainRayActor._get_rollout_data](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L216)：用 Megatron 提供的 DP rank 调用取数函数，然后显式把 token、mask 等 tensor 移到当前加速器。

因此 worker 的 get 是获取训练数据，driver 前面的 get 是等 manager 的方法返回。这两次等待各自有意义。

## 另一个数据传输分支

`rollout_data_transport`默认是 `object-store`，也支持 `nixl`。本版本会为相关 Actors 启用 tensor transport，并在 `ray.put`时选择 `_tensor_transport="nixl"`。这里涉及额外环境与 Ray 接口要求；本教程没有实测该分支。它与第 9 章的**模型权重同步**是不同的数据流。

选读 [参数声明](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/arguments.py#L565)。暂缓具体 packing 算法、CP 切分数学和 NIXL 运行部署；本章到“worker 得到正确 DP 分片并搬到设备”为止。

**理解题：假设四个训练 worker 中 TP=2、DP=2，manager 会创建几个 Box？driver 第一次 get 后拿到的内容是什么？**


---

# 读通 train.py 的一整个循环

> 阅读目标：从一次生成走到下一次生成，逐项标出“提交”与“已完成”。状态：已展开。

## 先把入口压缩成一条能记住的线

下面是无 critic、无 offload / release 的**教学简化**，不是完整原文：

```python
pgs = create_placement_groups(args)
manager, per_epoch = create_rollout_manager(args, pgs["rollout"])
actor, _ = create_training_models(args, pgs, manager)
actor.update_weights()  # 第一次生成前，也要先同步

for rollout_id in range(start, end):
    data_refs = ray.get(manager.generate.remote(rollout_id))
    ray.get(actor.async_train(rollout_id, data_refs))
    if should_save:
        actor.save_model(rollout_id, force_sync=is_last)
    ray.get(manager.cleanup_rollout_data.remote(rollout_id))
    actor.clear_memory()
    actor.update_weights()
    if should_eval:
        ray.get(manager.eval.remote(rollout_id))

ray.get(manager.dispose.remote())
```

真实 [train.py → train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)还包含初始 evaluation、critic 与多种资源分支。第一遍先检查简化主线中每次调用如何结束。

## 一个 batch 经过哪些执行位置

| driver 的动作 | 真正执行位置 | 等到什么 |
|---|---|---|
| `generate.remote(id)`后 get | manager、HTTP 服务和推理 GPU | 样本生成与训练分片打包完成 |
| `actor.async_train(...)`后 get | 所有训练 Ray Actors | 所有本次训练方法返回 |
| `actor.save_model(...)` | worker 与 checkpoint 后端 | 取决于保存模式与 `force_sync` |
| `cleanup_rollout_data.remote(id)`后 get | manager | 该轮 routed-experts 等专用清理完成 |
| `actor.update_weights()` | worker、推理控制 Actor 与通信后端 | 这条更新路径的同步调用完成 |

`cleanup_rollout_data`不是“清空整个 Ray object store”；它调用的是 routed-experts 数据清理，移除对应活动记录。普通 ObjectRef 的生命周期还受持有引用与 Ray 管理机制影响。

## 训练方法内部到底在哪里更新参数

主链：**RayTrainGroup.async_train → MegatronTrainRayActor.train → _get_rollout_data → train_actor → model.train → train_one_step**。

1. [RayTrainGroup.async_train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L131)：输入轮次和 DP 引用清单，提交所有 worker；输出引用列表。
2. [MegatronTrainRayActor.train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L386)：取数据；按角色调用 `train_actor`或 `train_critic`；actor 路线完成后返回 `None`。
3. [train_actor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L436)：准备迭代器，按配置计算 log probability、advantage 等，调用后端训练；之后更新 actor 权重备份。
4. [model.train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/model.py#L710)：一轮 rollout 数据可能对应多个 optimizer step，按调度逐步执行。
5. [train_one_step](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/model.py#L512)：执行 Megatron 的 forward/backward 路线并调用 `optimizer.step()`；这才是数值训练的核心边界。

“一次 `train.remote`”不必然等于“一个 optimizer step”。本版本根据每轮样本对应的轨迹与 global batch 规划多个 step，并进一步分为 microbatches。

## 当前主线的时间线

```text
driver      等生成 ────── 等训练 ────── 保存/清理 ── 等权重同步 ── 下一轮
rollout     [ 生成 batch k ]                         [更新权重]
train                        [ 训练 batch k ]        [发送权重]
```

各训练 worker 之间可能并行，生成请求内部也可能并发，但默认这条主线没有显式预提交下一批生成。不要根据两个独立 GPU 池就断言训练与 rollout 一定重叠。浏览器实验会按源码顺序逐步显示这些等待点。

## 保存完成也要分层看

`RayTrainGroup.save_model()`会等待远程方法返回，但后端支持 `async_save`。[MegatronTrainRayActor.save_model](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L587)在特定位置 finalize 异步保存，`force_sync`控制是否等待当前保存结束。driver 对最后一次保存或 release 路线设置相应条件。

因此验证 checkpoint 时，要区分“已提交保存”“远程保存函数返回”“异步写入已完成”和“文件可用于恢复”。不能把其中一项日志当成全部证据。

## critic 分支怎么读

有 critic 时，先取得 `critic_model.async_train(...)`的引用列表，再通过 `external_data`交给策略训练。`RayTrainGroup.async_train`逐项把它作为远程方法参数传入，Ray 的顶层引用依赖会影响策略 worker 何时执行。这条实现不应简单描述成 actor / critic 无条件同时计算。

如果还没到 `num_critic_only_steps`，本轮只等待 critic。选读 [train_critic](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L408)，暂缓 PPO / GRPO 的损失公式；本章只要求读懂依赖与等待。

**理解题：请指出一条严格发生在“本轮所有训练 worker 完成之后、下一轮默认生成开始之前”的操作，并说明等待证据在哪里。**


---

# 训练完，推理模型怎么拿到新权重

> 阅读目标：把“训练样本传输”和“模型权重同步”两条数据流分开。状态：已展开。

## 训练模型和推理模型是两份状态

训练 worker 更新了自己的参数，不会自动修改 SGLang 进程里的模型。第一次 rollout 前也需要同步：训练侧可能从恢复 checkpoint 加载了与推理初始 checkpoint 不同的状态。

[train.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)在创建模型后先调用一次 `actor_model.update_weights()`，之后每轮训练结束再调用。这是显式维护两份模型状态一致性的编排动作。

## 先选对更新器

[create_weight_updater](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/__init__.py#L10)按配置选择实现：

| 条件 | 实现 | 本教程覆盖 |
|---|---|---|
| full + nccl + 非 colocate | `UpdateWeightFromDistributed` | 本章主线 |
| 非 disk、colocate | `UpdateWeightFromTensor` | 识别接口，不深入 IPC 细节 |
| full + disk | `UpdateWeightFromDisk` | 第 10 章解释生命周期连接 |
| delta + disk | `UpdateWeightFromDiskDelta` | 仅解释选择条件与限制 |

表中分支有顺序：delta 与 disk 会先被判断，不能只看 colocate 开关。本教程主线选择普通 full / nccl，在 NVIDIA GPU 环境下追分布式权重广播。

## 从 driver 到更新器

主链：**RayTrainGroup.update_weights → 各训练 Actor.update_weights → 获取可更新 engines 和 Lock → 必要时建立通信组 → updater.update_weights**。

1. [RayTrainGroup.update_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L162)：普通分支提交每个 worker 的更新方法并等待。不要仅按其简短 docstring 把整个实现说成“全局 rank 0 发全部权重”。
2. [MegatronTrainRayActor.update_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L612)：按需恢复故障引擎，向 manager 询问 engines、锁、GPU 数等信息；必要时重新连接更新通信组。
3. [get_updatable_engines_and_lock](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L157)：本版本选择第一个 `update_weights=True`的模型服务，冻结模型不在其中；并非同时更新所有模型。
4. [UpdateWeightFromDistributed.connect_rollout_engines](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L58)：每个 PP stage 中符合 DP / TP 源 rank 条件的训练进程建立对应更新组。暂时忽略 PP 时，可把它看成一个源训练进程连接多个推理 GPU。

## 两条线：控制消息与张量

```text
训练 worker ── Ray 方法(names, dtypes, shapes, version) ──→ SGLangEngine
     │                                                       │ HTTP 控制请求
     │                                                       ▼
     └──────── 分布式 broadcast(参数张量) ─────────────→ SGLang 接收侧
```

在 [update_weights_from_distributed](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L326)里，先对引擎提交接收命令，再调用 `dist.broadcast`传实际 tensor，等待 broadcast handles，最后上层 `ray.get(refs)`确认接收方法结束。

这里 Ray 参数主要携带名字、类型、形状、group 名和版本等控制信息。权重张量没有通过第 7 章的 `Box(ray.put(...))`路线走。一个是 manager 到训练侧的样本流，一个是训练侧到推理侧的模型状态流。

为什么先提交接收命令再等待？因为接收端可能需要等待广播发送端参与。如果提交后立刻阻塞等待“接收完成”，而发送还没发生，就可能互等。源码在两边都被启动后才等待完成。

## 更新协议是一组有顺序的动作

[UpdateWeightFromDistributed.update_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L102)的大体顺序是：增加版本 → 暂停生成 → flush cache → barrier → 按块整理与传权重 → 必要的量化后处理 → 恢复生成 → barrier。

转换与 gather 由训练后端处理：训练格式中的分片参数需要转换成推理侧需要的名字与张量布局。不要期待“在 Ray 里广播一下 Python 模型对象”就解决并行布局差异。

**设计理解**：暂停与 cache 处理用于协调权重切换边界；锁用于避免多个源并发操作接收端造成通信顺序冲突。它们不能证明任何异常下都具备事务回滚或完全一致的恢复语义。

## 这个 Lock 不是阻塞式锁

[Lock.acquire](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/utils.py#L63)检查一个布尔值，获取成功返回 True，失败返回 False。调用方循环尝试并 `sleep(0.1)`；远程锁 Actor 不在自己的 acquire 方法里一直等。

这样默认同步 Actor 仍能处理后续 release 方法。若 acquire 在锁 Actor 内阻塞，release 又在同一个串行队列后面，就可能永远排不到。源码里的锁也没有 lease 或 owner identity；不要据此推导调用者死亡后的自动解锁。

选读 [SGLangEngine.update_weights_from_distributed](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/sglang_engine.py#L397)。暂缓具体 MoE 权重转换、量化与 IPC 算法；本章到双方确认更新结束为止。

**理解题：为什么“rollout 样本使用 object-store”与“权重通过 NCCL 广播”可以同时成立？请分别说出发送方、接收方和载荷。**


---

# 同卡部署、卸载与进程释放

> 阅读目标：分清“放在哪里”“显存里留什么”“进程还在不在”。状态：已展开。

## 三个开关回答三个问题

| 机制 | 回答的问题 | 可观察到的动作 |
|---|---|---|
| `colocate` | 训练和 rollout 是否使用重叠 GPU 布局？ | PG 数量与索引起点改变 |
| `offload_train / offload_rollout` | 某阶段是否卸载 / 恢复部分设备内存状态？ | sleep / wake、release / resume memory |
| `release_train` | rollout 阶段是否销毁训练 Ray Actors？ | `ray.kill`，随后重新 create 与加载 |

这些不是同一个动作的三种叫法。开启某种资源布局也不等于已经实现显存切换；要沿入口和参数校验读完整组合。

## 同卡时，PG 怎么算

在 [_get_placement_group_layout](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L100)的 colocate 分支，总数为 `max(actor_num_gpus, rollout_num_gpus)`，rollout offset 为 0。训练两卡、rollout 两卡时，逻辑布局从四卡变为两卡共用。

资源主链：**布局改变 → 相同 PG 区域 → 训练和推理控制 Actor 使用分数 GPU 配额 → 后端显式处理设备与内存**。

训练 Actor 通常申请 `0.4 GPU / 0.4 CPU`，SGLang 控制 Actor 申请 `0.2 GPU / 0.2 CPU`。在它们实际绑定的 bundle 上，这些数字参与调度记账。**0.4 GPU 不表示 Ray 给这个进程硬划了 40% 显存、40% SM 或固定算力。**实际服务可能使用多张卡，不能只把控制 Actor 的配额相加推导真实显存峰值。

这是 CPU 实验无法验证的边界；GPU 显存是否足够要结合模型、KV cache、optimizer、通信 buffer 和峰值测量判断。

## offload：对象还活着，内存状态暂时撤走

1. [train.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)：生成之后可让 rollout offload；更新前先恢复推理权重所需状态，更新后再恢复 KV cache 等。
2. [MegatronTrainRayActor.train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L386)：`offload_train`时训练前 wake_up，训练后 sleep。
3. [MegatronTrainRayActor.sleep](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L175)：配合 memory saver 释放设备占用，并处理进程组生命周期；对应 wake_up 恢复。
4. [RolloutServer.onload_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/engine_group.py#L316)与[onload_kv](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/engine_group.py#L325)：按阶段恢复权重、KV cache / CUDA graphs 等状态。

设计上把权重恢复与 KV 恢复分开，可让权重更新发生在推理完整工作集重新就绪之前。具体节省多少显存、切换多久，本教程没有测量。

注意 `clear_memory()`只是另一个清理接口，不应等同于杀进程、卸载模型或释放 PG。

## release：需要保存，再销毁，再重建

本版本的参数校验要求 `release_train`有保存路径、使用 `full + disk`权重更新，并且不支持 critic。见[校验位置](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/arguments.py#L2083)。

主链是：**训练 → 强制同步保存训练 checkpoint → 导出更新权重 → release 训练 Actors → 推理从磁盘加载权重 → rollout → 重新 create 训练 Actors → 恢复训练状态**。

[RayTrainGroup.save_model](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L151)在 release 路线更新恢复参数；[update_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L162)的 full disk 分支在导出后调用 release，再让 engines 重载；[release](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L181)清空 handles 并 `ray.kill(..., no_restart=True)`。

训练 checkpoint 与推理可加载的权重目录用途不同：前者用于恢复训练状态，后者让推理服务加载模型参数。不能认为“导出了推理权重”就一定保存了 optimizer / RNG 状态。

销毁训练 Actor 也不等于删除 PG。PG 可能继续由作业持有，给重建或共用流程使用；Ray 的“可用资源”与进程显存占用是两个观察维度。

## 选择分支时如何思考

**设计理解**：分卡路线较容易隔离训练与推理的资源需求；同卡与 offload 减少同时驻留的要求，但增加切换管理；release 更彻底地移除训练进程，却带来保存、重建与重新加载成本。这是机制上的取舍，不是已测量的性能排名。

选读 [RayTrainGroup._reload_rollout_weights_from_disk](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L227)。暂缓磁盘 delta 编码、memory saver 底层分配器和不同 GPU 型号的兼容性验证。

**理解题：看到训练 Actor 已被 kill，你能否断言它对应的 GPU 已归还给整个 Ray 集群？还要检查什么资源对象？**


---

# 真正的异步发生在哪里

> 阅读目标：解释为什么默认 train.py 是顺序驱动，而 fully-async rollout 仍能持续在后台生成。状态：已展开。

## 不要寻找一个并不存在的入口

在本教程固定的提交中，根目录没有 `train_async.py`。fully-async 示例仍使用 `train.py`，通过替换 rollout 函数启用后台持续生成。因此不能拿旧版本或其他框架的异步入口来解释当前实现。

绑定位置是：

```text
--rollout-function-path
slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

本章属于进阶分支，前提是已经理解默认的“生成一批 → 训练一批 → 同步权重”。这里只做源码解释，没有运行完整异步 GPU 训练。

## 前台还是一轮一轮，后台一直有工作

主链：**manager 调用替换后的函数 → 获取进程级 worker → 后台线程运行 asyncio loop → 完成组进入队列 → 前台收够一批返回 → driver 开始训练 → 后台继续工作**。

```text
driver         get batch k ─── train k ─── update ─── get batch k+1
manager 前台   [收集完成组]                            [收集完成组]
后台线程       [生成任务池 → 完成队列 → 再补新任务 ……]
推理服务                             更新窗口可能暂停 / 恢复
```

这是结构示意，非实测时间线。权重更新仍可能暂停生成；“fully async”不表示生成完全不受训练与同步影响，也不保证零等待。

## 顺着线程与队列读

1. [generate_rollout_fully_async](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/fully_async_rollout.py#L317)：作为 manager 绑定的公开入口，收集本轮数据；该版本明确拒绝 evaluation 模式。
2. [_get_global_worker](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/fully_async_rollout.py#L59)：复用进程级 `_global_worker`；首次调用创建 `AsyncRolloutWorker`并启动线程。配置并发容量由 server concurrency 与 engine 数，再按 samples per prompt 折算为 group 并发。
3. [AsyncRolloutWorker._thread_main](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/fully_async_rollout.py#L134)：在线程里运行 `asyncio.run(self._loop())`，与 manager 前台方法区分开。
4. [_loop](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/fully_async_rollout.py#L137)：回收完成任务，在额度允许时从 data buffer 取新 prompt groups，调用 `generate_and_rm_group`，把结果放入队列。
5. [_generate_rollout_async](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/fully_async_rollout.py#L221)：前台反复获取完成组，收够 rollout batch；按 sample index 排序，再交回 manager 的公共转换链路。

这里使用的是 Python 线程、asyncio Task 和 `queue.Queue`，不是又创建了一组 Ray 训练 Actor。不同层的“任务”必须标明所在系统。

## 队列没有 maxsize，为什么仍有背压

源码给 output queue 使用无界 `queue.Queue()`，注释解释是避免 done callback 在事件循环线程里被阻塞的 put 卡住。它把背压放在补任务条件上：

```python
while (
    len(active_tasks) < max_concurrent
    and self.output_queue.qsize() < max_concurrent
    and self.running
):
    # 从数据源取组，提交新生成
    ...
```

也就是说：结果堆到一个池规模时停止补新工作，等训练侧取走结果再补。已在途的任务还可能继续完成，所以它不是“队列长度任何时刻绝不超过这个数”的严格断言，更不是字节级内存上限。

这个观察也解释了为什么只检查 `Queue(maxsize=...)`不足以判断系统有没有背压：要找到生产者的补充条件和消费者的取数位置。

## 异步之后，样本会变旧

假设某组样本在策略 v3 时开始生成，训练已把推理侧推进到 v4；队列中可能仍有来自旧版本的样本。异步系统需要明确允许多旧的数据、如何记录版本、如何进行算法修正。

本章只从源码确认队列跨 rollout 调用保留，以及 `rollout.py`在相应配置下查询权重版本用于度量。不要把“有版本字段”解释成“所有 staleness 都已自动消除”。想进一步研究要读 loss、off-policy 修正与测量路径，本教程不证明其训练收敛效果。

## 退出与失败的边界

全局 worker 通过 `atexit`注册停止函数；stop 设置 running 并 join 线程，源码带有等待上限。这不是异常退出或强杀进程时的持久化队列协议。某些 ABORTED 组会被重新放回 buffer，也不等于任意任务都自动恰好执行一次。

选读[官方仓库的 fully-async 示例说明](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/examples/fully_async/README.md#L1)。暂缓 rollout filter、复杂 agent 轨迹和异步算法效果；先确认并发发生在哪条线程及哪个边界。

**理解题：如果 driver 仍在 `ray.get(actor.async_train(...))`等待训练，后台生成为什么还能继续？是哪一段状态没有随着上一轮 generate 返回而销毁？**


---

# 卡住与失败时，沿边界排查

> 阅读目标：先定位“哪一步尚未完成”，再选日志与检查工具。状态：已展开。

## 一句“Ray 卡住了”还不够

同一个 `ray.get`等待可能是资源没凑齐、Actor 构造出错、服务没健康、HTTP 请求没返回，或某个训练 rank 卡在 collective。先写清楚最后一条成功事件与下一条预期事件，比盲目重启更有用。

本章主链是：**找到 driver 等待点 → 定位执行进程 → 检查该层依赖 → 确认失败 / 继续等待 → 按该层真实恢复路径处理**。

## 按阶段观察

| 现象 | 首先核对 | 源码入口 |
|---|---|---|
| 一直打印 Waiting for placement group | 注册 / 可用 GPU、CPU、bundle 能否放下、PG 是否 pending | `_create_placement_group` |
| PG ready 后没有模型 ready | Actor 构造日志、环境依赖、rank/master 地址、checkpoint | `RayTrainGroup.create`与 worker.init |
| rollout 初始化不返回 | server 进程是否还活着、health endpoint、端口与模型加载 | `launch_server_process` |
| 生成阶段不返回 | 数据源、HTTP 服务、reward 函数、有效样本目标 | `generate_rollout_async` |
| 训练 get 不返回 | 哪个 rank 未进入 / 退出通信，数据引用是否能取回 | `train`与训练后端 |
| 权重更新不返回 | pause、锁、通信组、收发是否都已启动、engine 健康 | `update_weights` |

仅看 driver 的栈只能知道“我在等”；还要看被等待的 worker 与服务端。

## 资源总量为什么不等于可用量

`ray.cluster_resources()`显示集群逻辑总量，`ray.available_resources()`显示当时可用量。PG 已预留的资源可能不再算作其他任务可用；Actor 占有调度资源也不代表此刻 GPU kernel 正在运行。显存占用和 GPU utilization 要从设备与进程层面另看。

只读排查命令示例：

```bash
ray status
ray list placement-groups --detail
ray list actors --detail
ray list tasks --detail
```

State CLI 需要相应 Ray dashboard / state 依赖与可访问地址；本教程的轻量 CPU 环境不保证安装了它们。不要把 CLI 连接失败直接解释成训练失败。Python 基础资源查询可在已连接的 driver 中进行。

## 本版本真的实现了怎样的引擎恢复

1. [RolloutManager.__init__](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L45)在 `use_fault_tolerance`开启时，为 server groups 启动健康监控线程；普通主线未开启时不能假设它存在。
2. [RolloutHealthMonitor._check_engine_health](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/health_monitor.py#L136)调用引擎健康方法；出错后进入 `_kill_engine`，尝试 shutdown / kill，并把相应 engine slot 设为 None。
3. [MegatronTrainRayActor.update_weights](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L612)的 rank 0 在更新前请求 manager 恢复。
4. [RolloutManager.recover_updatable_engines](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L256)与[RolloutServer.recover](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/engine_group.py#L263)重启可更新服务的缺失引擎；后续新引擎数量驱动训练侧重建权重通信连接。

所以“健康监控发现问题”“重启 engine”“重新连接权重组”“同步最新权重”是分开的阶段。也不能把 rollout engine 的恢复直接推广成训练 rank 故障后的完整 optimizer 恢复。

## 一处容易误读的 timeout

`engine.health_generate.remote(timeout=...)`把 timeout 传进远程方法，不等于给外面的 `ray.get(...)`加了超时。本版本 monitor 在这一处没有 `ray.get(..., timeout=...)`。底层 HTTP 请求 timeout 与 Ray 调用等待时间是两层接口。

同样，启动阶段的 `_wait_server_healthy`轮询与 PG 的 30 秒轮询也不自动形成整场作业的总截止时间。教程只解释源码已有行为，不补造 watchdog 或重试保证。

## 正常结束，不等于验证过强杀清理

[train.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)正常循环结束后调用 `rollout_manager.dispose()`；该方法停止健康监控、清理活跃 routed-experts 数据并结束 tracking。它没有在该方法里逐个显式 shutdown 所有 SGLang engines，也没有调用 remove PG。

Ray 所属作业的资源生命周期、子进程退出机制、外部服务及异常退出时的清理需要分别验证。入口主线也没有包住全流程的 `try/finally`。因此本教程不会把“正常 dispose 路径存在”写成“任何故障后都保证无孤儿进程”。

我们自己的 CPU 实验使用 `try/finally`，显式 kill 教学 Actors、remove 教学 PG，再 `ray.shutdown()`，便于反复练习。它是教学脚本的清理策略，不是对 slime 源码的描述。

## 写一份有用的故障记录

最小记录包含：源码 SHA、Ray / CUDA / 后端版本、配置分支、最后完成阶段、当前等待调用、每个相关进程的状态，以及一个最小复现。GPU 集群上再分别记录资源 ready、模型 ready、首个 rollout、optimizer 更新、权重版本和 checkpoint 恢复证据。

选读 [SGLangEngine.shutdown](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/sglang_engine.py#L296)。暂缓 Ray 集群控制平面的底层容错和完整故障注入；本章没有宣称跑过 driver kill 验证。

**理解题：PG ready、Actor alive、服务 health 200、optimizer.step 返回、checkpoint 可恢复，这五项能互相替代吗？任选两项说出差别。**


---

# 亲手做一个 CPU 版迷你 slime

> 阅读目标：在无需模型和 GPU 的情况下，把资源、Actor、数据引用和权重版本串成一轮可检查的流程。状态：已展开。

## 这个实验验证什么

这是我们为教程编写的 **Ray 编排教学模型**。它不用 Megatron、SGLang、PyTorch、NCCL，也不会训练语言模型。它用整数模拟模型版本，用短列表模拟 batch，让你清楚观察引用和状态何时变化。

真实源码中的训练 worker 会依靠 collective 保持模型一致；本实验里各训练 Actor 独立维护相同的版本计数，仅模拟同步屏障。不要把它的正确运行当成 GPU 通信或 slime 可部署性的证据。

## 准备独立环境

站点构建只依赖 Python 标准库。CPU 实验使用 **Python 3.12 与 Ray 2.49.2**，这是教程实验的固定版本；slime 的 requirements 本身没有把 Ray 精确锁定到该版本。下面在教程目录创建局部虚拟环境：

```bash
cd /Users/lezewei/project/study/blog/ray
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r examples/requirements.txt
.venv/bin/python examples/01_tasks.py
```

如果本目录已有 `.venv`，不必重新创建；可先运行 `.venv/bin/python -c 'import ray; print(ray.__version__)'`检查。没有 uv 时，也可用已有 Python 3.12 的 `python3.12 -m venv .venv`和 `.venv/bin/python -m pip install -r examples/requirements.txt`。

脚本显式使用 `address="local"`启动独立本地 Ray，避免意外连接你的训练集群。每个脚本结束时会清理自己启动的资源。实验需要至少可供声明的少量 CPU 资源与本地内存；不需要 Ray 集群或 CUDA。

## 六个实验，按顺序运行

| 文件 | 操作 | 成功时应观察到什么 |
|---|---|---|
| [01_tasks.py](examples/01_tasks.py) | 提交平方任务，获取结果 | ObjectRef；返回 4、9；worker PID 与 driver 不同 |
| [02_actors.py](examples/02_actors.py) | 在两个 Counter 上调用方法 | 同一 Counter 累加到 5，另一实例保持独立 |
| [03_objects.py](examples/03_objects.py) | 顶层引用依赖、Box 包装与按分片取数 | 顶层参数收到值；Box 内仍是 ObjectRef |
| [04_placement.py](examples/04_placement.py) | 两 CPU bundle 就绪；再申请超过本地容量的 PG | 小组成功；大组在限时 wait 内未就绪，随后被移除 |
| [05_mini_slime.py](examples/05_mini_slime.py) | 生成 → DP 取数 → 训练屏障 → 同步版本 | 连续三轮的 serving / train 版本按预期递增 |
| [06_backpressure.py](examples/06_backpressure.py) | 有上限地提交任务、逐批消费 | 所有结果完整，记录的 pending 峰值不超过设定上限 |

可以逐个执行，例如：

```bash
.venv/bin/python examples/04_placement.py
.venv/bin/python examples/05_mini_slime.py
.venv/bin/python examples/06_backpressure.py
```

PID、启动耗时和任务完成顺序每次都可能变化。脚本验证不依赖固定 PID 或“必须加速几倍”；它检查值、引用类型、分片与版本顺序等关键性质。

## mini slime 的主链

**预留两个 CPU bundles → 创建两个训练 Actor 和一个 manager → 初次版本同步 → manager 生成两个数据分片 → driver 拿到 Box 列表 → worker 按 rank 获取对应分片 → 等两个训练调用完成 → 更新 manager 版本 → 下一轮**。

| 教学对象 | 对应的 slime 角色 | 被省略的实现 |
|---|---|---|
| 普通 Python 主程序 | `train.py` driver | 参数系统、保存与 evaluation |
| `TrainGroup` | `RayTrainGroup` | 多节点 rank/master 初始化 |
| `Trainer` Actor | `MegatronTrainRayActor` | tensor、梯度、optimizer、collective |
| `Manager` Actor | `RolloutManager`与推理侧状态的简化合体 | HTTP、SGLang server、reward |
| `Box(ray.put(shard))` | 默认 rollout 数据路径 | 长度均衡与真实 tensor |
| `set_version` | 推理侧拿到新权重的完成点 | 权重转换与传输协议 |

输出中的 `serving=0 → trained=[1,1] → synced=1`表示第一轮用版本 0 生成，两个 worker 都训练至版本 1，再把版本 1 同步给生成侧。它不是 slime 自带日志格式。

## 三个小改动，检验你是否理解

1. 注释掉每轮末尾的 `set_version`同步，保留脚本中的版本断言。预测哪一轮会报错，解释为什么 GPU 再多也无法修复这个状态问题。
2. 在 `Trainer.train`里打印实际取到的数据片。两个 worker 是否取到了不同分片？把选片 rank 改成常量 0，会在哪个断言被发现？
3. 把 `TrainGroup.async_train`改成每提交一个就 get，再让某个 worker 人为变慢。画出新旧时间线。这里用 `sleep`只能演示等待结构，不能模拟真实 GPU 性能。

修改前可以复制实验文件保留原版。实验退出后若发现未结束的本地 Ray 进程，先确认它属于本次本地 session；不要对共用训练机器直接执行无差别 stop / kill。

## 回到真实源码做最后一次对照

1. 打开 [train.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9)，用教学程序标出 driver 的等待点。
2. 打开 [RayTrainGroup.async_train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L131)，确认提交所有 worker 的方式。
3. 打开 [RolloutManager._split_train_data_by_dp](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L513)，确认 Box 的生产位置。
4. 打开 [process_rollout_data](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/data.py#L304)，确认真正取数的位置。
5. 打开[权重更新选择器](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/__init__.py#L10)，说出真实系统比 `set_version`多出的工作。

实际运行环境、退出码和结果记录统一保存在 [验证与来源](webapp/chapters/sources.html)。下一步上 GPU 前，还需要独立准备模型、数据、后端依赖、网络与资源配置；这些不包含在本 CPU 模型的验证范围里。

**结课任务：不看图，口述一批数据从 driver 发出 generate 命令，到下一轮推理使用新权重的全过程。至少标出三个进程边界、两次 get，以及一个不由 Ray 执行的计算或通信操作。**


---

# 术语与对象速查

## 先查对象类型

| 名称或表达式 | 类型 / 所在位置 | 应该怎么使用 |
|---|---|---|
| `args` | 普通配置对象 | 决定创建规模和分支 |
| `bundles` | Python list of dict | 传给 PG 创建接口 |
| `pg` | Ray PlacementGroup 句柄 | 调度策略引用它 |
| `pg.ready()` | ObjectRef | 等资源组准备好 |
| `actor_model` | driver 中的 RayTrainGroup | 普通方法组织远程调用 |
| `_actor_handlers[i]` | ActorHandle | 用 `.method.remote()`发命令 |
| `actor.train.remote(...)` | ObjectRef | 等一次训练方法结果 |
| `async_train(...)` | list[ObjectRef] | 等所有训练 worker |
| `rollout_manager` | ActorHandle | 调用生成、评估、服务管理接口 |
| driver 的 `rollout_data_ref` | 默认是 list[Box] | 传给训练组，名称不能代替类型检查 |
| `rollout_data_ref[dp_rank].inner` | ObjectRef | 在训练 worker 中获取对应分片 |
| `rollout_data` | 具体训练字典 | token、mask、reward 与调度元数据 |

## 再查系统名词

| 术语 | 在本教程里的意思 | 别与什么混淆 |
|---|---|---|
| driver | 执行主编排程序的进程 | 不是所有 GPU 计算都在这里 |
| Ray head | 提供 Ray 集群控制服务的节点角色 | 不等于训练 rank 0 |
| task | 一次远程函数调用或 actor method 调用的工作 | 不等于 CUDA kernel |
| Actor | 有状态的远程对象与对应 worker | 不等于 RL 的策略角色 |
| ObjectRef | 指向远程结果的引用 | 不等于已取回的结果 |
| bundle | 一份逻辑资源需求 | 不等于硬件隔离分区 |
| PG | Placement Group，资源预留 / 放置 | 不等于 PyTorch Process Group |
| rank | 某通信组中的进程编号 | 不一定是物理 GPU ID |
| world size | 某进程组的成员数量 | 不一定是 DP size |
| DP | 不同副本处理不同的数据分片 | 不等于一切训练 rank |
| TP / PP / CP | 参数计算 / 模型层 / 序列上下文的并行划分 | Ray 不负责其数学实现 |
| rollout | 用策略生成轨迹 / 样本的过程，也可指某条轨迹 | 轮次编号与 sample 数量不同 |
| policy / actor | RL 中生成动作的策略模型 | 不等于 Ray Actor 类本身 |
| critic | 价值模型角色 | 主线入门可以先不启用 |
| router | 为生成 HTTP 请求选择服务端 | 不是 Ray scheduler |
| engine | 一套推理服务实例，可含多个 GPU worker | 不总是一张卡一个 |
| microbatch | 一个训练 step 内更小的批次 | 不等于整个 rollout batch |
| backpressure | 消费跟不上时抑制继续生产 | 不等于无条件取消在途任务 |
| offload | 转移 / 释放部分设备驻留状态 | 不等于销毁进程 |
| release | 本教程分支中指销毁训练 Actors | 不等于移除 PG |
| staleness | 样本生成策略相对当前训练策略的陈旧程度 | 不只是网络延迟 |

## 三个“就绪”问题

问“资源是否就绪”，看 PG。问“方法是否完成”，看对应引用与异常。问“模型状态是否可用于下一阶段”，看初始化、权重版本、通信完成与保存恢复证据。它们分别回答不同的问题。

需要继续找实现时，使用[源码索引](webapp/chapters/sources.html)中的固定版本链接。


---

# 理解题与源码寻路任务

这份练习只覆盖已经展开的内容，不附标准答案。建议每题用 3–5 句话回答，再附一个源码位置。无法解释时回到对应章节，不需要先运行 GPU 训练。

## 每章一个核心问题

| 编号 | 对应章节 | 你要回答的问题 |
|---|---|---|
| 00 | [全景](webapp/chapters/ch00.html) | Ray、Megatron、SGLang 分别负责哪一层？ |
| 01 | [任务](webapp/chapters/ch01.html) | ObjectRef 出现时，哪些状态还不能确定？ |
| 02 | [Actor](webapp/chapters/ch02.html) | RayTrainGroup、ActorHandle 和 ObjectRef 如何区分？ |
| 03 | [等待](webapp/chapters/ch03.html) | `async_train`为什么不能证明端到端异步？ |
| 04 | [PG](webapp/chapters/ch04.html) | 四 bundle 只够三张 GPU 时卡在哪？第 4 张到来后是否已加载模型？ |
| 05 | [训练进程](webapp/chapters/ch05.html) | 为什么先创建所有 Actor，再等待所有 init？ |
| 06 | [生成](webapp/chapters/ch06.html) | 一个不申请 GPU 的 manager 如何触发 GPU 推理？ |
| 07 | [数据](webapp/chapters/ch07.html) | 两层 ObjectRef 的两个 get 分别获取什么？ |
| 08 | [循环](webapp/chapters/ch08.html) | 一轮训练结束与 checkpoint 可恢复是同一件事吗？ |
| 09 | [权重](webapp/chapters/ch09.html) | 样本路径和权重路径有哪些不同？ |
| 10 | [资源切换](webapp/chapters/ch10.html) | colocate、offload、release 分别改变什么状态？ |
| 11 | [异步](webapp/chapters/ch11.html) | 无界 output queue 的背压具体在哪一行生效？ |
| 12 | [排障](webapp/chapters/ch12.html) | 一个 timeout 参数到底限制哪一层等待？ |
| 13 | [实战](webapp/chapters/ch13.html) | CPU 模型验证了什么，又遗漏了什么？ |

## 五个源码寻路任务

**任务 A：训练 2 卡 + rollout 2 卡。** 写出总 PG GPU 数、rollout offset、训练 world size。自行编一个打乱的 bundle → 节点 / GPU 映射，算出排序结果。说明哪些数由配置决定，哪些只能从实际调度探测得到。

**任务 B：一批样本的物主。** 从 `train.py`的一次 generate 开始，列出 manager、driver、训练 worker 三处的变量类型。指出哪里是 ActorHandle、哪里是外层 ObjectRef、哪里是 list[Box]、哪里才有具体 tensor。

**任务 C：源 rank 的通信顺序。** 在权重更新路径中找到先发接收命令、后 broadcast、再等待接收完成的证据。解释把第一步改成“立即等待接收返回”可能造成什么依赖环。

**任务 D：只剩一项没有完成。** 假设一个训练 rank 没有退出 `train`。用 driver 等待引用的位置解释：其他 rank 完成能否让整个训练屏障通过？下一轮默认 generate 是否会提前被 driver 发出？

**任务 E：进阶分支。** 替换成 fully-async rollout 后，哪些后台状态跨轮次保留？一批生成结果的排序是否保证它们来自完全相同的权重版本？若要证明，需要再查什么证据？

## 自评方式

给每题分别标记：能找到文件、能按调用顺序解释、能画出进程 / 引用边界、能说出未验证的条件。能背出术语但不能指出调用方和等待位置，说明还需要回到小实验。

最终交付给自己的学习笔记可以只有一张图：上半部分画进程和资源，下半部分画数据与权重两条流，旁边标出每个等待点。图上的每个箭头都要能找到实际调用证据。


---

# 版本、证据与源码索引

## 本书使用的版本

| 项目 | 已确认内容 |
|---|---|
| 源码仓库 | [THUDM/slime](https://github.com/THUDM/slime) |
| 固定提交 | `5bae5bb7928d65906e24e99f5b5d3f99a4daaf93` |
| 提交日期与说明 | 2026-09-19；Don't use reloadable pg when not offloading train (#2394) |
| 获取方式 | 查询 main SHA，克隆 checkout 并读取源码；保留教程范围的原始文件快照 |
| 读取日期 | 2026-09-20 |
| 本地源码变更 | 读取的 checkout 无改动；`source/`是选定文件的只读副本，不是完整可运行安装 |
| CPU 实验版本 | Python 3.12.14 / Ray 2.49.2；与 slime 运行环境版本要求分别管理 |

正文链接打开的是本地带行号源码页；源码页顶部提供同一 SHA 的 GitHub 链接。这样即使上游 main 变动或离线阅读，仍能找到本书正在解释的实现。

`source/manifest.json`记录每个文件的 SHA-256；生成器会检查源码链接中的文件与类 / 函数是否真实存在。原始代码保持不变；教程中的省略代码均标注为教学简化。

## 核心目录逐个看

| 文件 | 职责 | 对应章节 |
|---|---|---|
| [train.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/train.py#L9) | driver 的端到端编排 | 00、08 |
| [placement_group.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/placement_group.py#L1) | 资源布局、探测、创建角色对象 | 04、05 |
| [actor_group.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/actor_group.py#L1) | 本地训练组、批量 RPC、创建 / 释放 | 02、05、08、10 |
| [rollout.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/rollout.py#L1) | manager、样本转换、DP 包装、服务接口 | 06、07、12 |
| [train_actor.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/train_actor.py#L1) | worker 基类、环境、进程组初始化 | 05 |
| [ray_actor.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/ray_actor.py#L1) | 地址和端口辅助方法 | 05 |
| [utils.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/utils.py#L1) | runtime env、设备开关与 Lock Actor | 05、09 |
| [__init__.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/ray/__init__.py#L1) | 当前版本为空 | 目录完整性说明 |

`slime/ray`的六个非空 Python 文件均已读取。正文覆盖它们的编排职责；诸如多模态字段、路由回放、量化和复杂恢复条件只在影响接口时提及，不代表每个分支都经过运行验证。

## 追出目录才完整的调用链

| 路线 | 必读证据 | 本书停止的位置 |
|---|---|---|
| 训练 | [MegatronTrainRayActor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/actor.py#L62) → [model.train](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/model.py#L710) → [train_one_step](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/model.py#L512) | 训练调度与 optimizer 接口；不推导损失公式 |
| 推理部署 | [start_rollout_servers](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/deployment.py#L79) → [ServerGroup.start_engines](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/engine_group.py#L63) → [SGLangEngine.init](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/sglang_engine.py#L105) | 服务启动和控制接口；不读取 SGLang 内部全部实现 |
| 默认生成 | [generate_rollout](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/sglang_rollout.py#L668) → [generate](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/sglang_rollout.py#L154) | HTTP 请求与 Sample 回填 |
| 数据源 | [data_source.py](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/data_source.py#L1) | manager 配置绑定与调用接口 |
| DP 数据 | [build_dp_schedule](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/dp_schedule.py#L82) → [Box](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/misc.py#L143) → [process_rollout_data](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/data.py#L304) | 分片、引用与 worker 取数 |
| 权重 | [create_weight_updater](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/__init__.py#L10) → [UpdateWeightFromDistributed](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L24) | full / nccl 控制与广播；其他实现仅定位选择分支 |
| 异步 | [AsyncRolloutWorker](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/rollout/fully_async_rollout.py#L82) | 线程、队列、背压与跨批次生命周期 |
| 健康监控 | [RolloutHealthMonitor](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/utils/health_monitor.py#L9) → [RolloutServer.recover](https://github.com/THUDM/slime/blob/5bae5bb7928d65906e24e99f5b5d3f99a4daaf93/slime/backends/sglang_utils/engine_group.py#L263) | 已实现的检测与引擎重建路径 |

## 验证记录与边界

构建、示例执行和浏览器检查的实际结果会写入教程根目录 `validation/report.md`。源码读取、CPU 示例验证与真实 GPU 训练是三个证据层级。

**2026-09-20 实际执行**：macOS arm64、Python 3.12.14、Ray 2.49.2；六个 CPU 示例均退出码 0。mini slime 三轮的推理版本依次为 0、1、2，训练 / 同步后依次为 1、2、3；背压示例消费了全部 12 项结果，pending 峰值为 3。

站点已生成 17 篇页面与首页。32 份源码快照的 SHA-256 与固定提交的 checkout 一致；构建时已验证正文引用的源码符号。网页与交互的检查记录另见 `validation/report.md`。

**没有执行的验证**：完整 slime GPU 训练、跨节点通信、NCCL 权重传输、NIXL、colocate 显存切换、release 重建恢复、GPU 故障注入和 driver 强杀清理。教学模拟的图表不提供这些运行证据。

## Ray 官方接口参考

这些是理解基础接口的补充，源码行为以本书固定提交为准。官方 `/latest/`文档会更新；本次读取的页面标注 Ray 2.58.0，不能把它当成本教程 CPU 实验使用的 2.49.2 或 slime 的锁定版本。

- [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)：远程函数与调用方式。
- [Actors](https://docs.ray.io/en/latest/ray-core/actors.html)：有状态 worker。
- [Objects](https://docs.ray.io/en/latest/ray-core/objects.html)：结果引用与对象。
- [Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)：bundle、整组预留和策略。
- [Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)：逻辑资源声明与物理使用的区别。
- [限制 pending tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html)：基于 wait 的提交背压。
- [Actor 方法执行顺序](https://docs.ray.io/en/latest/ray-core/actors/task-orders.html)：不同 Actor 配置下的顺序边界。

## 来源与许可

本教程为独立中文源码导学，不是 slime 官方文档，也不是 Ultra-Scale Playbook 的译文。教程结构按学习依赖重新编排，正文、CPU 实验和交互组件为本次任务编写。

页面配色、卡片、目录与阅读排版改编自用户指定的 [Twinkle AI Community / Ultra-Scale Playbook 中文版](https://github.com/ai-twinkle/ultrascale-playbook-zh-tw)。复用的基础 CSS 按 Apache 2.0 保留来源与许可；本站没有复用该书的正文与插图。

slime 选定源码文件保留其 Apache 2.0 许可；本地 `marked.min.js`为 Marked v12.0.2（MIT），文件头保留版权声明。完整第三方许可与修改说明见教程根目录 `THIRD_PARTY_NOTICES.md`及 `licenses/`。
