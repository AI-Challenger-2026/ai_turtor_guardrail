// 가드레일 판정 기록 패널 (guard trace panel)
import type { Decision, GuardResult, GuardTrace, TurnMeta, Usage } from '../types';
import { DECISION_LABELS, FINAL_ACTION_LABELS, formatNumber, labelOf } from '../labels';

const DECISION_CLASSES: Record<Decision, string> = {
  allow: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  revise: 'border-amber-200 bg-amber-50 text-amber-800',
  block: 'border-rose-200 bg-rose-50 text-rose-700',
};

function DecisionChip({ decision }: { decision: string }) {
  const cls =
    decision in DECISION_CLASSES
      ? DECISION_CLASSES[decision as Decision]
      : 'border-slate-200 bg-slate-100 text-slate-600';
  return (
    <span className={`inline-flex whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium ${cls}`}>
      {labelOf(DECISION_LABELS, decision)}
    </span>
  );
}

/** 출력 가드 시도 번호: 0 = 최초 답변, 1 이상 = 재생성 */
function attemptLabel(attempt: number): string {
  if (attempt <= 0) return '최초';
  if (attempt === 1) return '재생성';
  return `재생성 ${attempt}`;
}

function detailsTitle(details: Record<string, unknown> | undefined): string | undefined {
  if (!details || Object.keys(details).length === 0) return undefined;
  try {
    return JSON.stringify(details, null, 2);
  } catch {
    return undefined;
  }
}

interface StageTableProps {
  title: string;
  results: GuardResult[];
  showAttempt: boolean;
}

function StageTable({ title, results, showAttempt }: StageTableProps) {
  return (
    <section>
      <h4 className="mb-1.5 text-xs font-semibold text-slate-600">
        {title} <span className="font-normal text-slate-400">({results.length})</span>
      </h4>
      {results.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-200 bg-white px-3 py-2 text-xs text-slate-400">
          실행된 가드가 없습니다.
        </p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="w-full min-w-[600px] text-left text-xs">
            <thead className="bg-slate-50 text-[11px] text-slate-500">
              <tr>
                <th className="px-2.5 py-1.5 font-medium">가드 이름</th>
                <th className="px-2.5 py-1.5 font-medium">판정</th>
                <th className="px-2.5 py-1.5 font-medium">카테고리</th>
                <th className="px-2.5 py-1.5 font-medium">사유</th>
                <th className="px-2.5 py-1.5 text-right font-medium">점수</th>
                <th className="px-2.5 py-1.5 text-right font-medium">지연(ms)</th>
                {showAttempt && <th className="px-2.5 py-1.5 font-medium">시도</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {results.map((g, i) => (
                <tr key={`${g.name}-${g.attempt}-${i}`} className="align-top">
                  <td className="whitespace-nowrap px-2.5 py-1.5 font-mono text-[11px] text-slate-800">{g.name}</td>
                  <td className="px-2.5 py-1.5">
                    <DecisionChip decision={g.decision} />
                  </td>
                  <td className="px-2.5 py-1.5 text-slate-600">{g.category || '-'}</td>
                  <td className="px-2.5 py-1.5 text-slate-600" title={detailsTitle(g.details)}>
                    {g.reason || '-'}
                  </td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 text-right tabular-nums">{formatNumber(g.score, 2)}</td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 text-right tabular-nums">
                    {Number.isFinite(g.latency_ms) ? Math.round(g.latency_ms) : '-'}
                  </td>
                  {showAttempt && (
                    <td className="whitespace-nowrap px-2.5 py-1.5 text-slate-600">{attemptLabel(g.attempt)}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

interface GuardTracePanelProps {
  trace: GuardTrace;
  meta?: TurnMeta | null;
  usage?: Usage | null;
  traceId?: string | null;
}

export default function GuardTracePanel({ trace, meta, usage, traceId }: GuardTracePanelProps) {
  const metaItems: { label: string; value: string }[] = [];
  if (meta) {
    if (meta.rewrite_applied && meta.rewritten_query) {
      metaItems.push({ label: '재작성된 질의', value: meta.rewritten_query });
    }
    if (meta.question_type) metaItems.push({ label: '질문 유형', value: meta.question_type });
    if (meta.learner_level) metaItems.push({ label: '학습자 수준', value: meta.learner_level });
    if (typeof meta.retrieved_count === 'number') {
      metaItems.push({ label: '검색된 청크', value: `${meta.retrieved_count}개` });
    }
  }

  return (
    <div className="mt-3 space-y-3 rounded-xl border border-slate-200 bg-slate-50/70 p-3">
      <StageTable title="입력 가드" results={trace.input ?? []} showAttempt={false} />
      <StageTable title="출력 가드" results={trace.output ?? []} showAttempt />

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-200 pt-2.5 text-xs text-slate-600">
        <span>
          프로필 <span className="font-mono font-medium text-slate-800">{trace.profile}</span>
        </span>
        <span aria-hidden="true" className="text-slate-300">
          |
        </span>
        <span>재생성 {trace.regenerations}회</span>
        <span aria-hidden="true" className="text-slate-300">
          |
        </span>
        <span>
          최종 판정{' '}
          <span className="font-medium text-slate-800">{labelOf(FINAL_ACTION_LABELS, trace.final_action)}</span>
          <span className="ml-1 font-mono text-[11px] text-slate-400">({trace.final_action})</span>
        </span>
      </div>

      {(metaItems.length > 0 || usage || traceId) && (
        <div className="space-y-1 text-[11px] leading-5 text-slate-400">
          {metaItems.length > 0 && (
            <p className="[overflow-wrap:anywhere]">
              {metaItems.map((item, i) => (
                <span key={item.label}>
                  {i > 0 && <span className="mx-1.5 text-slate-300">·</span>}
                  {item.label}: <span className="text-slate-500">{item.value}</span>
                </span>
              ))}
            </p>
          )}
          {usage && (
            <p>
              LLM 호출 {usage.llm_calls}회
              <span className="mx-1.5 text-slate-300">·</span>
              토큰 {Number.isFinite(usage.total_tokens) ? usage.total_tokens.toLocaleString('ko-KR') : '-'}
              <span className="mx-1.5 text-slate-300">·</span>
              예상 비용 ${formatNumber(usage.estimated_cost_usd, 4)}
            </p>
          )}
          {traceId && (
            <p className="[overflow-wrap:anywhere]">
              trace_id: <span className="font-mono">{traceId}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
