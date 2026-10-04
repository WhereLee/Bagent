"""记忆生命周期纯规则（不依赖 DB/LLM，便于单测）。

三类关注点：
- trust 门控：只有 >=min_trust 的记忆参与作答（draft 默认不作答，防自毒）。
- 双时态：valid_at/invalid_at 决定某时刻是否"为真"（联网知识会过期）。
- 去重/更新/促升：按相似度决定 ADD/UPDATE/NOOP；佐证数达阈值促为 verified。
"""
from __future__ import annotations

from datetime import datetime, timezone

TRUST_RANK = {"draft": 0, "verified": 1, "curated": 2}


def trust_ok(candidate_trust: str, min_trust: str) -> bool:
    return TRUST_RANK.get(candidate_trust, 0) >= TRUST_RANK.get(min_trust, 0)


def is_active(valid_at: datetime | None, invalid_at: datetime | None,
              now: datetime | None = None) -> bool:
    """此刻是否为真：valid_at<=now 且 (invalid_at 为空或 now<invalid_at)。"""
    now = now or datetime.now(timezone.utc)
    if valid_at is not None and now < valid_at:
        return False
    if invalid_at is not None and now >= invalid_at:
        return False
    return True


def promote_trust(support: int, threshold: int, current: str = "draft") -> str:
    """佐证数达阈值则从 draft 升 verified；curated 不回退。"""
    if current == "curated":
        return "curated"
    return "verified" if support >= threshold else current


# P3-C：经验靠"用过之后成没成"治理（弥补"多源一致≠真相"）
_NEXT_TRUST = {"draft": "verified", "verified": "curated", "curated": "curated"}


def outcome_action(ok: bool, current_trust: str, fail_total: int, fail_threshold: int = 2) -> tuple[str, str | None]:
    """结果反馈决策：成功→促升一级(经验被验证)；失败累计达阈值→作废；否则不动。返回 (action, new_trust)。"""
    if ok:
        return ("promote", _NEXT_TRUST.get(current_trust, "curated"))
    if fail_total >= fail_threshold:
        return ("invalidate", None)
    return ("noop", None)


def decide_op(similar: list[tuple[int, float]], dedup_sim: float,
              exact_hash_match: bool = False) -> tuple[str, int | None]:
    """给定相似命中 [(id, similarity)]，决定 ADD/UPDATE/NOOP。

    - 完全相同(内容 hash 命中) -> NOOP。
    - 最相近相似度 >= dedup_sim -> UPDATE 该条（改写而非重复堆叠）。
    - 否则 -> ADD。
    """
    if exact_hash_match:
        return ("NOOP", None)
    best = max(similar, key=lambda x: x[1], default=None)
    if best is not None and best[1] >= dedup_sim:
        return ("UPDATE", best[0])
    return ("ADD", None)
