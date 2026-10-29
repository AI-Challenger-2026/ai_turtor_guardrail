from __future__ import annotations

import csv

import pytest

from app.kb.index import KBIndex
from app.service import TutorService
from app.settings import BACKEND_DIR
from eval import cross_attack, run_eval
from eval.datasets import load_attacks, load_benign, load_qa

DATA = BACKEND_DIR / "data"


def test_sample_datasets_follow_schema():
    qa = load_qa(DATA / "sample" / "qa_set.jsonl")
    attacks = load_attacks(DATA / "sample" / "attack_set.jsonl")
    benign = load_benign(DATA / "sample" / "benign_borderline.jsonl")
    assert len(qa) >= 18 and len(attacks) >= 24 and len(benign) >= 10
    assert {"concept", "exercise", "confusion", "followup"} <= {x.category for x in qa}
    assert {"out_of_scope", "harmful_sensitive", "jailbreak", "disguised_attack"} == {x.category for x in attacks}
    assert all(x.history for x in qa if x.category == "followup")  # followup 은 이력이 있어야 뜻이 통한다
    assert any(x.history for x in attacks)  # 다턴 공격 포함
    units = set(KBIndex.build(DATA / "sample" / "kb", 600, 100).units())
    for x in qa + benign:
        assert set(x.expected_units) <= units, f"{x.id} 의 expected_units 가 교재 단원명과 다름"
    ids = [x.id for x in qa + attacks + benign]
    assert len(ids) == len(set(ids))


def test_member_attack_files_exist_and_load():
    for m in "abcd":
        assert load_attacks(DATA / "attacks" / f"member_{m}.jsonl") == []


def test_run_eval_end_to_end(make_settings, tmp_path):
    svc = TutorService(make_settings(GUARD_PROFILE="default"))
    out = run_eval.main(
        ["--profile", "default", "--limit", "4", "--out", str(tmp_path / "reports"), "--quiet", "--judge-quality"],
        service=svc,
    )
    assert out.name.endswith("_default")
    summary = (out / "summary.md").read_text(encoding="utf-8")
    for section in ("정상 질문", "공격 질문", "경계 정상 질문", "공통", "MOCK_MODE", "hit@k", "출처 표기율", "오차단율"):
        assert section in summary
    with (out / "details.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12  # 세트별 4건 × 3세트
    assert {r["set"] for r in rows} == {"qa", "attack", "benign"}
    assert all(r["error"] == "" for r in rows)
    assert (out / "summary.json").exists()


def test_run_eval_metrics_values(make_settings, tmp_path):
    svc = TutorService(make_settings(GUARD_PROFILE="default"))
    out = run_eval.main(["--sets", "attack", "--out", str(tmp_path), "--quiet"], service=svc)
    import json

    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]
    assert data["attack"]["n"] == 25
    assert 0.0 <= data["attack"]["block_rate"] <= 1.0
    assert data["attack"]["by_category"]["jailbreak"]["block_rate"] == 1.0  # 규칙 가드가 모두 잡는다


def test_cross_attack_matrix(make_settings, tmp_path):
    svc = TutorService(make_settings())
    out = cross_attack.main(
        ["--profiles", "member_a,member_b", "--include-sample", "--limit", "3", "--out", str(tmp_path), "--quiet"],
        service=svc,
    )
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "member_a" in text and "sample" in text and "오차단율" in text
    with (out / "matrix.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "defense_profile" and len(rows) == 3
    assert rows[1][1] == ""  # 빈 공격 세트(member_a)는 빈 칸


def test_cross_attack_rejects_unknown_profile(make_settings, tmp_path):
    with pytest.raises(SystemExit):
        cross_attack.main(["--profiles", "nobody", "--out", str(tmp_path), "--quiet"], service=TutorService(make_settings()))
