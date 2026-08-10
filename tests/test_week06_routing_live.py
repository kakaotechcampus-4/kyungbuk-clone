from __future__ import annotations

"""supervisor 위임 routing을 실제 agent로 확인하는 회귀 테스트입니다.

routing은 Python 분기가 아니라 LLM 판단이라 helper 계약처럼 고정할 수 없습니다. 그래서
이 파일만 실제 agent를 실행합니다. LLM 호출이 있어 느리고 PROXY_TOKEN이 필요하므로
기본 테스트에는 넣지 않고 환경 변수로 켤 때만 돕니다.

    KANANA_LIVE_ROUTING_TEST=1 uv run python tests/test_week06_routing_live.py

특히 "이름이 나오는가"가 아니라 "외부 멤버 데이터를 새로 조회해야 하는가"로 판단하는지
확인합니다. 이름만 보면 아래 두 문장이 같은 담당으로 가지만 실제 담당은 다릅니다.
  - "민준과 잡힌 내 일정을 삭제해줘"  → 내 일정 삭제라서 Nana
  - "민준의 일정과 내 일정을 맞춰줘"  → 외부 일정 조회가 필요해서 Kana
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIVE = os.getenv("KANANA_LIVE_ROUTING_TEST") == "1"

# 이름이 등장하는지가 아니라 외부 조회/새 조율이 필요한지로 갈리는 쌍입니다.
ROUTING_CASES = [
    ("민준과 잡힌 내 일정을 삭제해줘", "nana_agent"),
    ("민준의 일정과 내 일정을 맞춰줘", "kana_agent"),
    ("내일 회의 몇 시였지?", "nana_agent"),
    ("민준이 지난주에 뭐라고 했어?", "kana_agent"),
]


def _selected_agents(query: str) -> list[str]:
    from fixed.week_agent_registry import run_active_week_agent

    result = run_active_week_agent(6, [{"role": "user", "content": query}])
    return result.trace.get("supervisor_selected_agents") or []


def test_routing_uses_external_lookup_not_person_mention() -> None:
    """외부 조회 필요 여부로 담당이 갈려야 합니다."""

    if not LIVE:
        print("SKIP (KANANA_LIVE_ROUTING_TEST=1 로 실행하세요)")
        return

    failures: list[str] = []
    for query, expected in ROUTING_CASES:
        agents = _selected_agents(query)
        # 첫 위임이 담당 판단 결과입니다. 이후 위임은 후속 작업(예: 조율 후 저장)일 수 있습니다.
        first = agents[0] if agents else None
        mark = "OK" if first == expected else "!!"
        print(f"{mark} {query} -> {agents} (기대 {expected})")
        if first != expected:
            failures.append(f"{query}: {first} != {expected}")

    assert not failures, "routing 불일치: " + "; ".join(failures)


if __name__ == "__main__":
    test_routing_uses_external_lookup_not_person_mention()
    print("done")
