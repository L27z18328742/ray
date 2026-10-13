# Ray 与 slime 源码导读

面向了解过 Ray、尚未实战的读者。14 章完整中文教程、3 篇附录、6 个浏览器交互实验、6 个真实 Ray CPU 示例。视觉样式参考同级 `ultrascale-playbook-zh-tw`。

源码固定为 THUDM/slime `5bae5bb7928d65906e24e99f5b5d3f99a4daaf93`（2026-09-19），2026-09-20 读取。不是完整 slime 运行环境，不需要 GPU 就能阅读全站。

## 阅读

```bash
cd /Users/lezewei/project/study/blog/ray
bash run.sh
```

打开 <http://127.0.0.1:8650>。端口可用 `PORT=8651 bash run.sh`覆盖。也可以直接打开 `webapp/index.html`：正文、目录、源码与交互不依赖联网；已读进度与复制功能受浏览器对 file URL 的限制。

构建只需系统 Python 3，无 npm 或第三方 Python 依赖：`python3 build.py`。生成的 `webapp/`可单独作为静态站点，且所有资源均为本地文件。

## 文件

- `content/`：逐章 Markdown，主要编辑入口。
- `tutorial.md`：构建生成的全书 Markdown，源码链接指向固定 SHA。
- `course.json`：章节顺序、标题和交互类型。
- `webapp/`：生成后的阅读网站与带行号源码页面。
- `source/`：选定源码原样快照、SHA-256 清单；不是完整仓库。
- `examples/`：CPU 实验，Python 3.12 / Ray 2.49.2。
- `validation/`：实际验证记录及脚本输出。
- `THIRD_PARTY_NOTICES.md`、`licenses/`：第三方来源与许可。

## 实验

```bash
uv venv --python 3.12 .venv  # 已有环境时跳过
uv pip install --python .venv/bin/python -r examples/requirements.txt
.venv/bin/python examples/05_mini_slime.py
```

脚本各自启动独立本地 Ray，完成后清理。`05_mini_slime.py`只是编排教学模型；没有模型训练、NCCL、SGLang、GPU 权重同步。完整 GPU 系统的适配与运行结论不能从这些 CPU 实验推出。

## 更新文章

正文源码链接使用 `[文字](source:相对路径#类.方法)`；构建器解析实际 AST，验证符号存在并链接到准确行号。也支持 `#L123`行号。章节间使用 `ch07.html`等链接。修改后运行 `python3 build.py`。

升级源码时必须重新核对调用链与分支，不要只替换 SHA。阅读进度只保存在本地浏览器，站点没有跟踪或外部分析服务。
