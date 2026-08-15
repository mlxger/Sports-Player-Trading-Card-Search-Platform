"""
Stage 执行时间与字段分组记录器。

记录内容：
  ① 哪些字段被合并成了同一个 Stage（字段分组）
  ② 每个 Stage 的执行耗时
  ③ 程序总体耗时
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# ── 单条记录 ──────────────────────────────────────────────────────────────────


@dataclass
class StageRecord:
    """单个 Stage 的执行记录。"""

    stage_index: int
    fields: list[str]
    elapsed: float  # 秒
    success: bool
    timed_out: bool = False
    error: str | None = None

    @property
    def status_label(self) -> str:
        if self.timed_out:
            return "TIMEOUT"
        if not self.success:
            return "ERROR  "
        return "OK     "

    @property
    def time_per_field(self) -> float:
        """字段均摊时间（Stage 内所有字段共享同一次模型调用）。"""
        return self.elapsed / max(len(self.fields), 1)

    @property
    def is_merged(self) -> bool:
        return len(self.fields) > 1


# ── 计时器主体 ─────────────────────────────────────────────────────────────────


class StageTimer:
    """
    记录整个抽取流程的 Stage 分组与用时。

    典型用法
    --------
    >>> timer = StageTimer().start()
    >>> # ... 各 Stage 执行，stage_runner 内部自动调用 timer.record_stage() ...
    >>> timer.stop().print_summary()
    """

    def __init__(self) -> None:
        self._records: list[StageRecord] = []
        self._total_start: float | None = None
        self._total_elapsed: float | None = None

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def start(self) -> StageTimer:
        """开始整体计时。"""
        self._total_start = time.perf_counter()
        return self

    def stop(self) -> StageTimer:
        """停止整体计时。"""
        if self._total_start is not None:
            self._total_elapsed = time.perf_counter() - self._total_start
        return self

    # ── 记录接口（由 stage_runner 自动调用）──────────────────────────────────

    def record_stage(
        self,
        fields: list[str],
        elapsed: float,
        success: bool,
        timed_out: bool = False,
        error: str | None = None,
    ) -> None:
        self._records.append(
            StageRecord(
                stage_index=len(self._records) + 1,
                fields=list(fields),
                elapsed=elapsed,
                success=success,
                timed_out=timed_out,
                error=error,
            )
        )

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    @property
    def records(self) -> list[StageRecord]:
        return list(self._records)

    @property
    def total_elapsed(self) -> float | None:
        return self._total_elapsed

    def get_field_elapsed(self, field_name: str) -> float | None:
        """查询某字段所在 Stage 的实际耗时。"""
        for rec in self._records:
            if field_name in rec.fields:
                return rec.elapsed
        return None

    def get_stage_groupings(self) -> list[list[str]]:
        """返回所有 Stage 的字段分组，顺序与执行顺序一致。"""
        return [rec.fields for rec in self._records]

    def get_merged_stages(self) -> list[StageRecord]:
        """返回所有合并了多个字段的 Stage。"""
        return [rec for rec in self._records if rec.is_merged]

    # ── 报告生成 ──────────────────────────────────────────────────────────────

    def summary_text(self) -> str:
        W = 66
        sep = "─" * W
        lines: list[str] = []

        def center_banner(text: str) -> str:
            inner = W - 2
            padded = text.center(inner)
            return f"║{padded}║"

        """
        lines += [
            "",
            "╔" + "═" * W + "╗",
            center_banner("Stage 执行报告  ·  时间统计  &  字段分组"),
            "╚" + "═" * W + "╝",
            sep,
            "",
        ]
        """
        """
        # ── ① 字段分组 ────────────────────────────────────────────────────────
        lines.append("【① 字段分组  ——  哪些字段合并成了同一个 Stage】")
        lines.append("")

        merged_stages = self.get_merged_stages()
        if merged_stages:
            lines.append(f"  共有 {len(merged_stages)} 个 Stage 做了字段合并：")
        else:
            lines.append("  所有字段均以独立 Stage 运行，未做合并。")
        lines.append("")

        for rec in self._records:
            tag = "合并 ▶" if rec.is_merged else "独立  "
            field_display = "  +  ".join(rec.fields)
            lines.append(f"  Stage {rec.stage_index:>2}  [{tag}]  {field_display}")

        lines += ["", sep, ""]
        """
        # ── ② 用时明细 ───────────────────────────────────────────────────────
        # lines.append("【② 各 Stage 用时明细】")
        # lines.append("")

        col_fields = 32
        lines.append(
            f"  {'#':>3}  {'字段':^{col_fields}}  {'状态':^9}  {'耗时(s)':>8}  {'字段均摊(s)':>10}"
        )

        # lines.append(f"  {'─'*3}  {'─'*col_fields}  {'─'*9}  {'─'*8}  {'─'*10}")

        for rec in self._records:
            field_str = " + ".join(rec.fields)
            if len(field_str) > col_fields:
                field_str = field_str[: col_fields - 3] + "..."
            lines.append(
                f"  {rec.stage_index:>3}  {field_str:<{col_fields}}  "
                f"{rec.status_label:^9}  {rec.elapsed:>8.2f}  "
                f"{rec.time_per_field:>10.2f}"
            )
            if rec.error and not rec.timed_out:
                lines.append(f"       ↳ 错误：{rec.error}")

        lines += ["", sep, ""]
        """
        # ── ③ 汇总 ───────────────────────────────────────────────────────────
        lines.append("【③ 汇总】")
        lines.append("")

        stage_total = sum(r.elapsed for r in self._records)
        overall     = self._total_elapsed if self._total_elapsed is not None else stage_total
        ok_count    = sum(1 for r in self._records if r.success)
        fail_count  = len(self._records) - ok_count
        total_fields = sum(len(r.fields) for r in self._records)

        lines += [
            f"  程序总体用时      : {overall:>8.2f} s",
            f"  Stage 累计用时    : {stage_total:>8.2f} s",
            f"  其他开销          : {max(overall - stage_total, 0):>8.2f} s",
            f"  Stage 数量        : {len(self._records):>8} 个",
            f"  字段总数          : {total_fields:>8} 个",
            f"  成功 / 失败       : {ok_count:>4} / {fail_count}",
            "",
        ]
        """
        # 字段分组一行总览
        lines.append("【字段分组快照（一行总览）】")
        groups = ["[" + ", ".join(r.fields) + "]" for r in self._records]
        # 自动折行
        chunk, current = [], ""
        for g in groups:
            if current and len(current) + len("  →  ") + len(g) > W:
                chunk.append(current)
                current = g
            else:
                current = current + ("  →  " if current else "") + g
        if current:
            chunk.append(current)
        for c in chunk:
            lines.append("  " + c)

        lines.append("")
        return "\n".join(lines)

    def print_summary(self, enabled: bool = True) -> None:
        """按需打印汇总报告到 stdout。"""
        if not enabled:
            return
        print(self.summary_text())
