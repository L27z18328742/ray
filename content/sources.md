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
| [train.py](source:train.py#train) | driver 的端到端编排 | 00、08 |
| [placement_group.py](source:slime/ray/placement_group.py) | 资源布局、探测、创建角色对象 | 04、05 |
| [actor_group.py](source:slime/ray/actor_group.py) | 本地训练组、批量 RPC、创建 / 释放 | 02、05、08、10 |
| [rollout.py](source:slime/ray/rollout.py) | manager、样本转换、DP 包装、服务接口 | 06、07、12 |
| [train_actor.py](source:slime/ray/train_actor.py) | worker 基类、环境、进程组初始化 | 05 |
| [ray_actor.py](source:slime/ray/ray_actor.py) | 地址和端口辅助方法 | 05 |
| [utils.py](source:slime/ray/utils.py) | runtime env、设备开关与 Lock Actor | 05、09 |
| [__init__.py](source:slime/ray/__init__.py) | 当前版本为空 | 目录完整性说明 |

`slime/ray`的六个非空 Python 文件均已读取。正文覆盖它们的编排职责；诸如多模态字段、路由回放、量化和复杂恢复条件只在影响接口时提及，不代表每个分支都经过运行验证。

## 追出目录才完整的调用链

| 路线 | 必读证据 | 本书停止的位置 |
|---|---|---|
| 训练 | [MegatronTrainRayActor](source:slime/backends/megatron_utils/actor.py#MegatronTrainRayActor) → [model.train](source:slime/backends/megatron_utils/model.py#train) → [train_one_step](source:slime/backends/megatron_utils/model.py#train_one_step) | 训练调度与 optimizer 接口；不推导损失公式 |
| 推理部署 | [start_rollout_servers](source:slime/backends/sglang_utils/deployment.py#start_rollout_servers) → [ServerGroup.start_engines](source:slime/backends/sglang_utils/engine_group.py#ServerGroup.start_engines) → [SGLangEngine.init](source:slime/backends/sglang_utils/sglang_engine.py#SGLangEngine.init) | 服务启动和控制接口；不读取 SGLang 内部全部实现 |
| 默认生成 | [generate_rollout](source:slime/rollout/sglang_rollout.py#generate_rollout) → [generate](source:slime/rollout/sglang_rollout.py#generate) | HTTP 请求与 Sample 回填 |
| 数据源 | [data_source.py](source:slime/rollout/data_source.py) | manager 配置绑定与调用接口 |
| DP 数据 | [build_dp_schedule](source:slime/utils/dp_schedule.py#build_dp_schedule) → [Box](source:slime/utils/misc.py#Box) → [process_rollout_data](source:slime/utils/data.py#process_rollout_data) | 分片、引用与 worker 取数 |
| 权重 | [create_weight_updater](source:slime/backends/megatron_utils/update_weight/__init__.py#create_weight_updater) → [UpdateWeightFromDistributed](source:slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#UpdateWeightFromDistributed) | full / nccl 控制与广播；其他实现仅定位选择分支 |
| 异步 | [AsyncRolloutWorker](source:slime/rollout/fully_async_rollout.py#AsyncRolloutWorker) | 线程、队列、背压与跨批次生命周期 |
| 健康监控 | [RolloutHealthMonitor](source:slime/utils/health_monitor.py#RolloutHealthMonitor) → [RolloutServer.recover](source:slime/backends/sglang_utils/engine_group.py#RolloutServer.recover) | 已实现的检测与引擎重建路径 |

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
