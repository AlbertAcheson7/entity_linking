# Agent 指南

本仓库是一个 ICD-10 到 ICD-11 MMS 实体链接的研究工作区。

## 工作规范

- 请勿修改 `data/source/` 目录下的源数据。
- 除非明确要求，否则请勿提交来自 `data/prepared/`、`chroma/`、`models/`、`logs/` 的生成产物或大型实验输出。
- 保持可复现性：由配置驱动的行为应当保留在 `configs/*.yaml` 中。
- 优先进行小巧、聚焦的改动。除非被明确要求，否则请勿重写整个流水线（Pipeline）。
- 仓库中可能会存在用户现有的改动。请勿回滚不相关的编辑。

## 重要入口点

- 命令行界面（CLI）：`src/icd_linker/cli.py`
- 数据准备：`src/icd_linker/prepare.py`
- 文本视图：`src/icd_linker/text_views.py`
- 检索后端：`src/icd_linker/retrieval.py`
- 模型适配器：`src/icd_linker/models.py`
- 评估指标：`src/icd_linker/metrics.py`
- 测试：`tests/test_core.py`

## 当前研究方向

旧的冒烟测试（Smoke-test）系统已归档至 `docs/smoke-test-notes.md`。

当前工作旨在验证一个新的模型/检索框架：
- 矩阵检索后端（Matrix retrieval backend）
- 多种嵌入适配器（Multiple embedding adapters）
- 目标视图扩展：`name_text`、`context_text`、`path_text`
- 查询变体：name / context / path
- 视图记录排序 vs 实体级聚合（View-record ranking vs entity-level aggregation）
- 针对一对多映射的评估指标

## 验证

对于仅包含代码的改动，请运行：

```bash
python -m unittest