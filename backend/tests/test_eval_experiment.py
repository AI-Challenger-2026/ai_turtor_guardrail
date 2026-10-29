"""바닐라 vs 가드 비교 실험용 도구 테스트 (attack-success judge, attack generator)."""

from __future__ import annotations

import json

from app.service import TutorService
from eval import generate_attacks, run_eval
from eval.datasets import load_attacks


def test_attack_success_judge_adds_compliance_metric(make_settings, tmp_path):
    svc = TutorService(make_settings(GUARD_PROFILE="none"))
    out = run_eval.main(
        ["--profile", "none", "--sets", "attack", "--limit", "6", "--judge-attack",
         "--out", str(tmp_path), "--quiet"],
        service=svc,
    )
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]
    atk = data["attack"]
    # 가드 없음 → 전부 통과(차단율 0), 하지만 실질 우회율은 판정기로 따로 측정된다
    assert atk["block_rate"] == 0.0
    assert atk["judged_n"] == 6 and atk["compliance_rate"] is not None
    assert 0.0 <= atk["compliance_rate"] <= 1.0
    summary_md = (out / "summary.md").read_text(encoding="utf-8")
    assert "실질 우회율" in summary_md


def test_vanilla_vs_default_compliance(make_settings, tmp_path):
    # 바닐라(none)보다 default 프로필의 실질 우회율이 낮거나 같아야 한다 (가드가 도움이 됨)
    def comp(profile: str) -> float:
        svc = TutorService(make_settings(GUARD_PROFILE=profile))
        out = run_eval.main(
            ["--profile", profile, "--sets", "attack", "--judge-attack", "--out", str(tmp_path / profile), "--quiet"],
            service=svc,
        )
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]["attack"]["compliance_rate"]

    assert comp("default") <= comp("none")


def test_generate_attacks_mock(make_settings, tmp_path):
    svc = TutorService(make_settings())
    out = tmp_path / "member_x.jsonl"
    generate_attacks.main(
        ["--out", str(out), "--author", "member_x", "--per-category", "3",
         "--categories", "jailbreak,out_of_scope"],
        service=svc,
    )
    items = load_attacks(out)
    assert len(items) == 6
    assert {x.category for x in items} == {"jailbreak", "out_of_scope"}
    assert all(x.expected_decision == "block" and x.author == "member_x" for x in items)
    assert all(x.id.startswith("member_x-") for x in items)
    assert len({x.id for x in items}) == 6  # id 중복 없음


def test_generate_attacks_append(make_settings, tmp_path):
    svc = TutorService(make_settings())
    out = tmp_path / "member_y.jsonl"
    generate_attacks.main(["--out", str(out), "--per-category", "2", "--categories", "jailbreak"], service=svc)
    generate_attacks.main(["--out", str(out), "--per-category", "2", "--categories", "jailbreak", "--append"], service=svc)
    items = load_attacks(out)
    assert len(items) == 4 and len({x.id for x in items}) == 4
