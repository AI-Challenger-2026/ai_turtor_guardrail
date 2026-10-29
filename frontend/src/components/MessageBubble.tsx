// 메시지 말풍선 (message bubble): 사용자 / 튜터 답변 / 오류 / 입력 중 표시
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { AssistantUiMessage, ErrorUiMessage, UiMessage, UserUiMessage } from '../uiTypes';
import { ANSWER_MODE_LABELS, formatLatency, labelOf } from '../labels';
import GuardTracePanel from './GuardTracePanel';
import Markdown, { citeAnchorId } from './Markdown';
import SourceCard from './SourceCard';
import StatusBadge from './StatusBadge';
import { AlertCircleIcon, BookIcon, ChevronDownIcon, FileTextIcon, RefreshIcon, ShieldAlertIcon, ShieldCheckIcon } from './icons';

function TutorAvatar({ tone = 'accent' }: { tone?: 'accent' | 'amber' | 'red' }) {
  const cls =
    tone === 'amber'
      ? 'bg-amber-100 text-amber-700'
      : tone === 'red'
        ? 'bg-rose-100 text-rose-600'
        : 'bg-indigo-600 text-white';
  return (
    <div
      className={`mt-0.5 hidden h-8 w-8 flex-none items-center justify-center rounded-full sm:flex ${cls}`}
      aria-hidden="true"
    >
      <BookIcon className="h-4 w-4" />
    </div>
  );
}

/* ---------------- 사용자 말풍선 (user bubble) ---------------- */

function UserBubble({ message }: { message: UserUiMessage }) {
  return (
    <div className="flex flex-col items-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2.5 text-[15px] leading-7 text-white shadow-sm [overflow-wrap:anywhere] sm:max-w-[75%]">
        {message.content}
      </div>
      {message.state === 'failed' && (
        <p className="mt-1 text-[11px] text-rose-500">전송 실패 · 대화 기록에서 제외됨</p>
      )}
    </div>
  );
}

/* ---------------- 튜터 답변 (assistant bubble) ---------------- */

function AssistantBubble({ message }: { message: AssistantUiMessage }) {
  const r = message.response;
  const isBlocked = r.blocked || r.status === 'blocked';
  // 차단된 답변에는 근거 없음 표시를 생략 (blocked 우선)
  const noEvidence = !isBlocked && (r.status === 'no_evidence' || (r.grounding === 'kb' && r.grounded === false));
  const citations = r.citations;

  const [showTrace, setShowTrace] = useState(false);
  const [openCites, setOpenCites] = useState<ReadonlySet<number>>(() => new Set<number>());
  const [flash, setFlash] = useState<{ index: number; seq: number } | null>(null);

  const citationIndexes = useMemo(() => citations.map((c) => c.index), [citations]);

  // 본문 [n] 클릭: 해당 출처 카드를 펼치고 강조 (expand + flash)
  const handleCite = useCallback((index: number) => {
    setOpenCites((prev) => {
      if (prev.has(index)) return prev;
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    setFlash((prev) => ({ index, seq: (prev?.seq ?? 0) + 1 }));
  }, []);

  const toggleCite = useCallback((index: number) => {
    setOpenCites((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  // 펼쳐진 뒤(렌더 후) 스크롤 이동 (scrollIntoView after render)
  useEffect(() => {
    if (!flash) return;
    const el = document.getElementById(citeAnchorId(message.id, flash.index));
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.querySelector('button')?.focus({ preventScroll: true });
  }, [flash, message.id]);

  const providerText = r.provider === 'mock' ? 'mock' : `${r.provider} · ${r.model}`;

  const statusBadges = (
    <>
      {noEvidence && (
        <StatusBadge tone="amber" size="xs" dot>
          자료에서 근거를 찾지 못함
        </StatusBadge>
      )}
      {r.status === 'revised' && (
        <StatusBadge tone="sky" size="xs" dot>
          근거 검증 후 재생성됨
        </StatusBadge>
      )}
      {r.status === 'fallback' && (
        <StatusBadge tone="orange" size="xs" dot>
          안전 대체 답변
        </StatusBadge>
      )}
    </>
  );
  const hasStatusBadges = noEvidence || r.status === 'revised' || r.status === 'fallback';

  return (
    <div className="flex gap-3">
      <TutorAvatar tone={isBlocked ? 'amber' : 'accent'} />
      <div className="min-w-0 flex-1">
        <div
          className={`rounded-2xl rounded-tl-md border px-4 py-3 shadow-sm sm:px-5 sm:py-4 ${
            isBlocked ? 'border-amber-200 bg-amber-50/80' : 'border-slate-200 bg-white'
          }`}
        >
          {isBlocked && (
            <div className="mb-3 flex items-start gap-2.5 border-b border-amber-200/70 pb-3">
              <span className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-amber-100 text-amber-700">
                <ShieldAlertIcon className="h-[18px] w-[18px]" />
              </span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-amber-900">안전 정책에 따라 답변이 제한되었습니다</p>
                {r.block_category && (
                  <p className="mt-0.5 text-xs text-amber-700 [overflow-wrap:anywhere]">
                    분류: <span className="font-mono">{r.block_category}</span>
                  </p>
                )}
              </div>
            </div>
          )}

          {hasStatusBadges && <div className="mb-2.5 flex flex-wrap gap-1.5">{statusBadges}</div>}

          {r.answer.trim() !== '' ? (
            <Markdown
              content={r.answer}
              messageId={message.id}
              citationIndexes={citationIndexes}
              onCite={handleCite}
            />
          ) : (
            <p className="text-sm text-slate-400">(빈 답변)</p>
          )}

          {citations.length > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-slate-500">
                <FileTextIcon className="h-3.5 w-3.5" />
                참고 자료 <span className="font-normal text-slate-400">({citations.length})</span>
              </p>
              <div className="space-y-2">
                {citations.map((c) => (
                  <SourceCard
                    key={`${c.index}-${c.chunk_id}`}
                    messageId={message.id}
                    citation={c}
                    expanded={openCites.has(c.index)}
                    onToggle={toggleCite}
                    flashSeq={flash && flash.index === c.index ? flash.seq : 0}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 하단 메타 정보 (provider · latency · 판정 보기) */}
        <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 px-1 text-[11px] text-slate-400">
          <StatusBadge tone="gray" size="xs" title={`제공자: ${r.provider} / 모델: ${r.model}`}>
            {providerText}
          </StatusBadge>
          <span title={`${r.latency_ms}ms`}>{formatLatency(r.latency_ms)}</span>
          <span aria-hidden="true">·</span>
          <span>{labelOf(ANSWER_MODE_LABELS, r.answer_mode)}</span>
          {r.guard_trace !== null && (
            <button
              type="button"
              onClick={() => setShowTrace((v) => !v)}
              aria-expanded={showTrace}
              className="ml-auto inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
            >
              <ShieldCheckIcon className="h-3.5 w-3.5" />
              {showTrace ? '판정 닫기' : '판정 보기'}
              <ChevronDownIcon className={`h-3 w-3 transition-transform ${showTrace ? 'rotate-180' : ''}`} />
            </button>
          )}
        </div>

        {showTrace && r.guard_trace !== null && (
          <GuardTracePanel trace={r.guard_trace} meta={r.meta} usage={r.usage} traceId={r.trace_id} />
        )}
      </div>
    </div>
  );
}

/* ---------------- 오류 말풍선 (error bubble) ---------------- */

interface ErrorBubbleProps {
  message: ErrorUiMessage;
  onRetry: (errorId: string) => void;
  retryDisabled: boolean;
}

function ErrorBubble({ message, onRetry, retryDisabled }: ErrorBubbleProps) {
  const metaParts: string[] = [];
  if (message.status > 0) metaParts.push(`HTTP ${message.status}`);
  if (message.code) metaParts.push(message.code);
  if (message.traceId) metaParts.push(`trace ${message.traceId}`);

  return (
    <div className="flex gap-3">
      <TutorAvatar tone="red" />
      <div className="min-w-0 flex-1">
        <div className="rounded-2xl rounded-tl-md border border-rose-200 bg-rose-50 px-4 py-3" role="alert">
          <div className="flex items-start gap-2.5">
            <AlertCircleIcon className="mt-0.5 h-[18px] w-[18px] flex-none text-rose-500" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium leading-6 text-rose-800">{message.message}</p>
              {metaParts.length > 0 && (
                <p className="mt-0.5 font-mono text-[11px] text-rose-400 [overflow-wrap:anywhere]">
                  {metaParts.join(' · ')}
                </p>
              )}
            </div>
          </div>
          <div className="mt-2.5 flex justify-end">
            <button
              type="button"
              onClick={() => onRetry(message.id)}
              disabled={retryDisabled}
              className="inline-flex items-center gap-1.5 rounded-lg border border-rose-200 bg-white px-3 py-1.5 text-xs font-medium text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshIcon className="h-3.5 w-3.5" />
              다시 시도
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- 입력 중 표시 (typing indicator) ---------------- */

export function TypingBubble() {
  return (
    <div className="flex gap-3" aria-live="polite">
      <TutorAvatar />
      <div className="inline-flex items-center gap-3 rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <span className="flex items-center gap-1" aria-hidden="true">
          <span className="typing-dot h-1.5 w-1.5 rounded-full bg-indigo-500" />
          <span className="typing-dot h-1.5 w-1.5 rounded-full bg-indigo-500" />
          <span className="typing-dot h-1.5 w-1.5 rounded-full bg-indigo-500" />
        </span>
        <span className="text-sm text-slate-500">답변을 준비하고 있어요…</span>
      </div>
    </div>
  );
}

/* ---------------- 진입점 (entry) ---------------- */

interface MessageBubbleProps {
  message: UiMessage;
  onRetry: (errorId: string) => void;
  retryDisabled: boolean;
}

export default function MessageBubble({ message, onRetry, retryDisabled }: MessageBubbleProps) {
  switch (message.kind) {
    case 'user':
      return <UserBubble message={message} />;
    case 'assistant':
      return <AssistantBubble message={message} />;
    case 'error':
      return <ErrorBubble message={message} onRetry={onRetry} retryDisabled={retryDisabled} />;
  }
}
