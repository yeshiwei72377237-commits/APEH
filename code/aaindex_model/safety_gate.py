#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fitness_model 的人工安全门控词表（单一事实源）。

背景（缺陷修复，2026-09-10）
--------------------------
`review_status` 历史上存在**两套词表**，且下游门控只匹配了其中一套：

- 自动审核 `ProteinMPNN/proteinmpnn_wrapper.py::audit_candidates_csv` 写出
  `HOLD_FOR_STRUCTURE_REVIEW` / `PASS_TO_STRUCTURE_MODEL`；
- 人工复核（见 `wiki/AI-Ethics-Safety.md` §2.2）写出 `HOLD`（待定）/ `BLOCK`（否决）。

`score.py` 与 `select_top_mutants.py` 原先只判断 `{"HOLD", "BLOCK"}`，
与自动审核的字符串不一致 → **门控分支从未触发**，被标记的候选照常被打分、
照常进入 TopN。仓库内第三方审计报告 `APEH项目参赛代码审计报告(1).md` 已指出该问题。

修复方式：把词表集中到这里，两个下游模块统一引用，避免再次漂移。

语义
----
凡 `review_status` 命中 `BLOCKED_STATUSES` 的候选，**禁止打分**（`score_invalid=2`）
且**禁止进入 TopN**。
"""
from __future__ import annotations

#: 人工复核词表 + 自动审核词表 —— 两者任一命中即视为「须人工/结构复核通过后方可下游」
HOLD_STATUSES = frozenset({"HOLD", "HOLD_FOR_STRUCTURE_REVIEW"})
BLOCK_STATUSES = frozenset({"BLOCK"})

#: 全部「不得进入下游」的 review_status 取值
BLOCKED_STATUSES = HOLD_STATUSES | BLOCK_STATUSES

#: 未命中任何风险标记、可直接进入下游
PASS_STATUSES = frozenset({"PASS_TO_STRUCTURE_MODEL", "PASS", "OK", ""})


def normalize(status) -> str:
    """把 review_status 归一化：去空白、转大写、缺失值视作空串。"""
    if status is None:
        return ""
    s = str(status).strip().upper()
    return "" if s in {"NAN", "NONE"} else s


def is_blocked(status) -> bool:
    """该候选是否被门控拦下（HOLD / BLOCK / HOLD_FOR_STRUCTURE_REVIEW）。"""
    return normalize(status) in BLOCKED_STATUSES
