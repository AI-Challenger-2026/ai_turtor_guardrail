import asyncio, json, re, sys
sys.path.insert(0, "/home/elicer/competition")
from app.classify import classify
from app import classify_feat
S = sys.argv[1]
SUBMIT = re.compile(r"수행\s*평가|숙제|과제|제출|답안|시험|보고서|assignment|homework|submi|graded|exam", re.I)
rows = []
for e in ("tutor", "guard"):
    rows += [json.loads(l) for l in open(f"public/{e}.jsonl") if l.strip()]
sem = asyncio.Semaphore(32)
async def one(r):
    async with sem:
        return await classify(r["messages"])
async def main():
    for run in (1, 2):
        out = await asyncio.gather(*[one(r) for r in rows])
        with open(f"{S}/pii_flags{run}.jsonl", "w") as f:
            for r, o in zip(rows, out):
                f.write(json.dumps({"id": r["id"], "flags": o.get("flags")}) + "\n")
    fixed = broke = touched = 0
    for run in (1, 2):
        fl = {json.loads(l)["id"]: json.loads(l)["flags"] for l in open(f"{S}/pii_flags{run}.jsonl")}
        for r in rows:
            f = fl[r["id"]]
            if not f: continue
            old = classify_feat.decide(f)["category"]
            text = " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
            new = old
            if old == "academic_dishonesty" and f["self_pii"] and not SUBMIT.search(text):
                new = "pii_disclosure"
            if new != old:
                touched += 1
                g = r["labels"]["category"]
                fixed += g == new; broke += g == old
                print(f"run{run} {r['id']} 정답 {g}: {old} -> {new}")
    print(f"바뀐 판정 {touched}, 고침 {fixed}, 망침 {broke}")
asyncio.run(main())
