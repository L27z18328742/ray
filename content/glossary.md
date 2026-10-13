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

需要继续找实现时，使用[源码索引](sources.html)中的固定版本链接。
