"""Split the public set into a few-shot half and a dev half.

Near-duplicate items are grouped so that a group never straddles the two halves.

    python3 eval/make_split.py
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ('tutor', 'guard')


def user_turns(item):
    return [m['content'] for m in item['messages'] if m['role'] == 'user']


def char_grams(text, n=3):
    s = ''.join(text.split())
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def load(public_dir):
    items, lines = [], []
    for ep in ENDPOINTS:
        for line in (public_dir / f'{ep}.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                items.append(json.loads(line))
                lines.append(line)
    return items, lines


def group_items(items, threshold):
    g_all = [char_grams('\n'.join(user_turns(it))) for it in items]
    g_last = [char_grams(user_turns(it)[-1]) for it in items]
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if jaccard(g_all[i], g_all[j]) >= threshold or jaccard(g_last[i], g_last[j]) >= threshold:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0]), g_all, g_last


def assign(items, groups, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for gi, members in enumerate(groups):
        eps = sorted({items[i]['endpoint'] for i in members})
        cats = sorted({items[i]['labels']['category'] for i in members})
        strata[(tuple(eps), tuple(cats))].append(gi)
    side_of = {}
    total = Counter()
    for key in sorted(strata):
        gids = strata[key][:]
        rng.shuffle(gids)
        gids.sort(key=lambda g: -len(groups[g]))
        local = Counter()
        for g in gids:
            n = len(groups[g])
            if local['fewshot'] != local['dev']:
                side = 'fewshot' if local['fewshot'] < local['dev'] else 'dev'
            elif total['fewshot'] != total['dev']:
                side = 'fewshot' if total['fewshot'] < total['dev'] else 'dev'
            else:
                side = rng.choice(['fewshot', 'dev'])
            side_of[g] = side
            local[side] += n
            total[side] += n
    return side_of


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--public', default=str(ROOT / 'public'))
    ap.add_argument('--out', default=str(ROOT / 'eval'))
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20261110)
    a = ap.parse_args()

    items, lines = load(Path(a.public))
    groups, g_all, g_last = group_items(items, a.threshold)
    side_of = assign(items, groups, a.seed)

    out_groups = []
    split_of_item = {}
    for gi, members in enumerate(groups):
        side = side_of[gi]
        for i in members:
            split_of_item[i] = side
        out_groups.append({
            'group': gi,
            'split': side,
            'endpoint': sorted({items[i]['endpoint'] for i in members}),
            'categories': sorted({items[i]['labels']['category'] for i in members}),
            'ids': [items[i]['id'] for i in members],
        })

    ids = [it['id'] for it in items]
    assert len(set(ids)) == len(ids), 'duplicate ids in public set'
    fewshot = [items[i]['id'] for i in range(len(items)) if split_of_item[i] == 'fewshot']
    dev = [items[i]['id'] for i in range(len(items)) if split_of_item[i] == 'dev']
    assert not set(fewshot) & set(dev) and len(fewshot) + len(dev) == len(items)
    max_cross = 0.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if split_of_item[i] != split_of_item[j]:
                max_cross = max(max_cross, jaccard(g_all[i], g_all[j]), jaccard(g_last[i], g_last[j]))
    assert max_cross < a.threshold, f'cross-split similarity {max_cross:.3f} >= threshold'

    out = Path(a.out)
    (out / 'public_dev').mkdir(parents=True, exist_ok=True)
    split = {
        'method': {
            'text': 'user turns; whitespace removed; char 3-grams',
            'link': 'jaccard(all user turns joined) >= t OR jaccard(last user turn) >= t, union-find',
            'threshold': a.threshold,
            'stratify': '(endpoint, sorted category set of group), greedy balance',
            'seed': a.seed,
        },
        'counts': {
            'items': len(items), 'groups': len(groups),
            'fewshot': len(fewshot), 'dev': len(dev),
            'group_sizes': dict(sorted(Counter(len(g) for g in groups).items())),
            'max_cross_split_jaccard': round(max_cross, 4),
        },
        'fewshot': fewshot,
        'dev': dev,
        'groups': out_groups,
    }
    (out / 'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for ep in ENDPOINTS:
        dev_lines = [lines[i] for i in range(len(items))
                     if items[i]['endpoint'] == ep and split_of_item[i] == 'dev']
        (out / 'public_dev' / f'{ep}.jsonl').write_text('\n'.join(dev_lines) + '\n', encoding='utf-8')

    print(f"items {len(items)}, groups {len(groups)}, sizes {split['counts']['group_sizes']}, "
          f"fewshot {len(fewshot)}, dev {len(dev)}, max cross-split jaccard {max_cross:.3f}")
    table = defaultdict(Counter)
    for i, it in enumerate(items):
        table[(it['endpoint'], it['labels']['category'])][split_of_item[i]] += 1
    print(f"{'endpoint':8s} {'category':22s} {'fewshot':>7s} {'dev':>5s}")
    for (ep, cat), c in sorted(table.items()):
        print(f"{ep:8s} {cat:22s} {c['fewshot']:7d} {c['dev']:5d}")
    pairs = Counter(g['split'] for g in out_groups if len(g['ids']) > 1)
    print('multi-item groups by split:', dict(pairs))


if __name__ == '__main__':
    main()
