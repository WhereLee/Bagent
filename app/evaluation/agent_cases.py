"""P1 评测用 mock 任务；不含真实业务数据，只验证轨迹/审批/工具边界。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentCase:
    name: str
    goal: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    requires_approval: bool = False
    expected_status: str = "completed"
    max_steps: int = 4


CASES: list[AgentCase] = [
    AgentCase("kb-basic", "查手册", "kb_search", {"query": "手册"},
              ["kb_search"], ["research"]),
    AgentCase("web-basic", "查网页", "web_search", {"query": "网页"},
              ["web_search"], ["research"]),
    AgentCase("kb-two-queries", "查两个问题", "kb_search", {"query": "问题A"},
              ["kb_search"], ["research"]),
    AgentCase("kb-answer", "直接回答", "kb_answer", {"query": "答案"},
              ["kb_answer"], ["research"]),
    AgentCase("memory-basic", "查记忆", "memory_search", {"query": "偏好"},
              ["memory_search"], ["research"]),
    AgentCase("case-six", "查故障", "kb_search", {"query": "故障"},
              ["kb_search"], ["research"]),
    AgentCase("case-seven", "查政策", "web_search", {"query": "政策"},
              ["web_search"], ["research"]),
    AgentCase("case-eight", "查型号", "kb_search", {"query": "型号"},
              ["kb_search"], ["research"]),
    AgentCase("case-nine", "查库存", "kb_answer", {"query": "库存"},
              ["kb_answer"], ["research"]),
    AgentCase("case-ten", "查历史", "memory_search", {"query": "历史"},
              ["memory_search"], ["research"]),
    AgentCase("invalid-args", "参数错误", "kb_search", {},
              [], ["research"], expected_status="completed"),
    AgentCase("duplicate", "重复检索", "kb_search", {"query": "重复"},
              ["kb_search"], ["research"]),
    AgentCase("research-await", "需要研究", "research", {"topic": "主题"},
              [], [], requires_approval=True, expected_status="awaiting_approval"),
    AgentCase("research-reject", "拒绝研究", "research", {"topic": "主题"},
              [], [], requires_approval=True, expected_status="rejected"),
    AgentCase("budget", "预算耗尽", "kb_search", {"query": "循环"},
              ["kb_search"], ["research"], expected_status="budget_exceeded", max_steps=2),
    AgentCase("final-only", "直接回答", None, {},
              [], ["research"]),
    AgentCase("case-sixteen", "查产品", "kb_search", {"query": "产品"},
              ["kb_search"], ["research"]),
    AgentCase("case-seventeen", "查售后", "web_search", {"query": "售后"},
              ["web_search"], ["research"]),
    AgentCase("case-eighteen", "查订单", "kb_answer", {"query": "订单"},
              ["kb_answer"], ["research"]),
    AgentCase("case-nineteen", "查经验", "memory_search", {"query": "经验"},
              ["memory_search"], ["research"]),
    AgentCase("timeout", "超时恢复", "slow", {"query": "慢查询"},
              [], ["research"]),
    AgentCase("retry", "瞬时失败重试", "flaky", {"query": "重试"},
              ["flaky"], ["research"]),
    AgentCase("low-confidence", "无证据回答", None, {}, [], ["research"]),
]
