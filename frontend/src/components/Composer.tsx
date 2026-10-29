// 입력창 (composer): 자동 높이 조절 textarea + 전송 버튼
import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { countChars } from '../labels';
import { SendIcon } from './icons';

const MAX_TEXTAREA_HEIGHT = 200;

interface ComposerProps {
  /** 응답 대기 중이면 true (pending) */
  disabled: boolean;
  maxChars: number;
  /** 전송 성공 시 true 를 돌려주면 입력창을 비운다 */
  onSend: (text: string) => boolean;
}

export default function Composer({ disabled, maxChars, onSend }: ComposerProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // 한글 IME 조합 상태 추적 (composition events) — 일부 브라우저 보조 판정용
  const composingRef = useRef(false);
  const wasDisabledRef = useRef(disabled);

  const charCount = countChars(value);
  const trimmedEmpty = value.trim() === '';
  const overLimit = charCount > maxChars;
  const canSend = !disabled && !trimmedEmpty && !overLimit;

  // 내용에 맞춰 높이 자동 조절 (auto-grow)
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > MAX_TEXTAREA_HEIGHT ? 'auto' : 'hidden';
  }, [value]);

  // 응답이 끝나 다시 활성화되면 포커스 복귀 (마우스 환경에서만, 모바일 키보드 팝업 방지)
  useEffect(() => {
    if (wasDisabledRef.current && !disabled) {
      let finePointer = false;
      try {
        finePointer = window.matchMedia('(pointer: fine)').matches;
      } catch {
        finePointer = false;
      }
      if (finePointer) textareaRef.current?.focus({ preventScroll: true });
    }
    wasDisabledRef.current = disabled;
  }, [disabled]);

  const submit = () => {
    if (!canSend) return;
    if (onSend(value)) setValue('');
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    submit();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== 'Enter' || e.shiftKey) return;
    // 한글 IME 조합 중 Enter 는 글자 확정용이므로 전송하지 않음 (isComposing / keyCode 229)
    if (e.nativeEvent.isComposing || e.keyCode === 229 || composingRef.current) return;
    e.preventDefault();
    submit();
  };

  return (
    <div className="flex-none border-t border-slate-200 bg-white/90 backdrop-blur">
      <form onSubmit={handleSubmit} className="mx-auto w-full max-w-3xl px-4 pb-4 pt-3 sm:px-6">
        <div
          className={`flex items-end gap-2 rounded-2xl border bg-white p-2 shadow-sm transition focus-within:ring-2 ${
            overLimit
              ? 'border-rose-300 focus-within:ring-rose-100'
              : 'border-slate-300 focus-within:border-indigo-300 focus-within:ring-indigo-100'
          } ${disabled ? 'bg-slate-50' : ''}`}
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => {
              composingRef.current = true;
            }}
            onCompositionEnd={() => {
              composingRef.current = false;
            }}
            disabled={disabled}
            rows={1}
            placeholder="질문을 입력하세요 (Enter 전송, Shift+Enter 줄바꿈)"
            aria-label="질문 입력"
            className="max-h-[200px] min-h-[40px] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-6 text-slate-800 placeholder:text-slate-400 focus:outline-none disabled:cursor-not-allowed disabled:text-slate-400"
          />
          <button
            type="submit"
            disabled={!canSend}
            aria-label="전송"
            title="전송 (Enter)"
            className="flex h-10 w-10 flex-none items-center justify-center rounded-xl bg-indigo-600 text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
          >
            {disabled ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-500" />
            ) : (
              <SendIcon className="h-[18px] w-[18px]" />
            )}
          </button>
        </div>
        <div className="mt-1.5 flex items-center justify-between gap-3 px-1 text-[11px]">
          <p className={overLimit ? 'text-rose-500' : 'text-slate-400'}>
            {overLimit
              ? `최대 ${maxChars.toLocaleString('ko-KR')}자까지 입력할 수 있어요.`
              : 'AI 답변은 틀릴 수 있어요. 중요한 내용은 교재로 한 번 더 확인하세요.'}
          </p>
          <span className={`flex-none tabular-nums ${overLimit ? 'font-medium text-rose-500' : 'text-slate-400'}`}>
            {charCount.toLocaleString('ko-KR')} / {maxChars.toLocaleString('ko-KR')}
          </span>
        </div>
      </form>
    </div>
  );
}
