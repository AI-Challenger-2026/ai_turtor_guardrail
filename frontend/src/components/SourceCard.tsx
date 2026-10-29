// 출처 카드 (source citation card) — 클릭하면 원문 청크를 펼친다
import type { Citation } from '../types';
import { formatNumber } from '../labels';
import { citeAnchorId } from './Markdown';
import { ChevronDownIcon } from './icons';

interface SourceCardProps {
  messageId: string;
  citation: Citation;
  expanded: boolean;
  onToggle: (index: number) => void;
  /** 본문 [n] 클릭 시 증가 → 강조 애니메이션 재시작 (flash restart key) */
  flashSeq: number;
}

export default function SourceCard({ messageId, citation, expanded, onToggle, flashSeq }: SourceCardProps) {
  const anchorId = citeAnchorId(messageId, citation.index);
  const panelId = `${anchorId}-body`;
  const pathParts = [citation.doc_title, citation.unit, citation.section].filter(
    (part) => typeof part === 'string' && part.trim() !== '',
  );

  return (
    <div
      id={anchorId}
      className={`relative scroll-mt-24 rounded-xl border bg-white transition-colors ${
        expanded ? 'border-indigo-200' : 'border-slate-200 hover:border-slate-300'
      }`}
    >
      {flashSeq > 0 && (
        <span key={flashSeq} className="cite-flash pointer-events-none absolute inset-0 rounded-xl" aria-hidden="true" />
      )}
      <button
        type="button"
        onClick={() => onToggle(citation.index)}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="flex w-full items-start gap-2.5 rounded-xl px-3 py-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300"
      >
        <span className="mt-0.5 flex-none rounded-md bg-indigo-50 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-indigo-700">
          [{citation.index}]
        </span>
        <span className="min-w-0 flex-1 text-[13px] leading-5 text-slate-700 [overflow-wrap:anywhere]">
          {pathParts.length > 0 ? (
            pathParts.map((part, i) => (
              <span key={i}>
                {i > 0 && <span className="mx-1 text-slate-300">›</span>}
                <span className={i === 0 ? 'font-medium text-slate-800' : ''}>{part}</span>
              </span>
            ))
          ) : (
            <span className="text-slate-500">{citation.doc_id}</span>
          )}
        </span>
        <ChevronDownIcon
          className={`mt-0.5 h-4 w-4 flex-none text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
        />
      </button>
      {expanded && (
        <div id={panelId} className="border-t border-slate-100 px-3 pb-3 pt-2">
          <div className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg bg-slate-50 px-3 py-2 text-[13px] leading-6 text-slate-700 [overflow-wrap:anywhere]">
            {citation.text}
          </div>
          <p className="mt-2 text-[11px] text-slate-400 [overflow-wrap:anywhere]">
            <span className="font-mono">{citation.chunk_id}</span>
            {citation.score !== null && citation.score !== undefined && (
              <span> · 점수 {formatNumber(citation.score, 3)}</span>
            )}
          </p>
        </div>
      )}
    </div>
  );
}
