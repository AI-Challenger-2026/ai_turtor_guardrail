// 대화 목록 (chat window): 메시지 스크롤 영역 + 빈 화면 예시 질문
import { useEffect, useRef } from 'react';
import type { UiMessage } from '../uiTypes';
import MessageBubble, { TypingBubble } from './MessageBubble';
import { BookIcon } from './icons';

const EXAMPLE_QUESTIONS = [
  '리스트와 튜플의 차이가 뭐야?',
  'for문이랑 while문 언제 써?',
  '함수에서 return은 왜 써?',
  '변수 이름 규칙 알려줘',
];

interface ChatWindowProps {
  messages: UiMessage[];
  pending: boolean;
  onRetry: (errorId: string) => void;
  onExample: (question: string) => void;
}

function EmptyState({ onExample, disabled }: { onExample: (q: string) => void; disabled: boolean }) {
  return (
    <div className="flex min-h-full flex-col items-center justify-center px-2 py-10 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-sm">
        <BookIcon className="h-6 w-6" />
      </div>
      <h2 className="text-xl font-semibold text-slate-800 sm:text-2xl">무엇을 배워 볼까요?</h2>
      <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
        궁금한 개념을 편하게 물어보세요. 차근차근 함께 이해해 볼게요.
      </p>
      <div className="mt-7 grid w-full max-w-xl grid-cols-1 gap-2.5 sm:grid-cols-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onExample(q)}
            className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 shadow-sm transition hover:border-indigo-200 hover:bg-indigo-50/50 hover:text-indigo-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatWindow({ messages, pending, onRetry, onExample }: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // 새 메시지/로딩 표시가 생기면 맨 아래로 스크롤 (auto-scroll on new message)
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: messages.length <= 1 ? 'auto' : 'smooth' });
  }, [messages.length, pending]);

  const isEmpty = messages.length === 0 && !pending;

  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
      <div className="mx-auto h-full w-full max-w-3xl px-4 sm:px-6">
        {isEmpty ? (
          <EmptyState onExample={onExample} disabled={pending} />
        ) : (
          <div className="space-y-6 py-6" role="log" aria-live="polite" aria-relevant="additions">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} onRetry={onRetry} retryDisabled={pending} />
            ))}
            {pending && <TypingBubble />}
          </div>
        )}
      </div>
    </div>
  );
}
