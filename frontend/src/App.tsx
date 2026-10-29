// 앱 루트 (app root): 서버 설정 로드, 대화 상태 관리, 채팅 요청
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ChatMessage, ConfigResponse, HealthResponse } from './types';
import type { DevSelection, EffectiveSettings, UiMessage, UserUiMessage } from './uiTypes';
import { newId } from './uiTypes';
import { ApiError, getConfig, getHealth, sendChat } from './api/client';
import { EMPTY_SELECTION, loadSelection, resolveEffective, saveSelection } from './settings';
import { GROUNDING_MODE_LABELS, labelOf } from './labels';
import ChatWindow from './components/ChatWindow';
import Composer from './components/Composer';
import DevSettings from './components/DevSettings';
import StatusBadge from './components/StatusBadge';
import { AlertCircleIcon, BookIcon, GearIcon, InfoIcon, NewChatIcon, RefreshIcon } from './components/icons';

const DEFAULT_MAX_CHARS = 2000;

type ConfigState =
  | { status: 'loading'; config: null }
  | { status: 'ready'; config: ConfigResponse }
  | { status: 'error'; config: null };

/** 성공한 교환만 history 로 변환 (role + content only, 실패/대기 메시지 제외) */
function toHistory(messages: UiMessage[]): ChatMessage[] {
  const history: ChatMessage[] = [];
  for (const m of messages) {
    if (m.kind === 'user' && m.state === 'sent') history.push({ role: 'user', content: m.content });
    else if (m.kind === 'assistant') history.push({ role: 'assistant', content: m.response.answer });
  }
  return history;
}

/* ---------------- 헤더 (header) ---------------- */

interface HeaderProps {
  configState: ConfigState;
  effective: EffectiveSettings | null;
  settingsOpen: boolean;
  canReset: boolean;
  onNewChat: () => void;
  onToggleSettings: () => void;
}

function HeaderBadges({ configState, effective }: Pick<HeaderProps, 'configState' | 'effective'>) {
  if (configState.status === 'loading') {
    return (
      <StatusBadge tone="gray" size="xs">
        설정 불러오는 중…
      </StatusBadge>
    );
  }
  if (configState.status === 'error' || !effective) return null;

  const config = configState.config;
  const gen = config.generator;
  const guardOff = effective.guard_profile === 'none' || config.guards.status !== 'enabled';

  return (
    <>
      {config.mock_mode ? (
        <StatusBadge tone="amber" size="xs" dot title="실제 LLM 대신 모의 응답(mock)을 사용합니다.">
          Mock 모드
        </StatusBadge>
      ) : (
        <StatusBadge
          tone="slate"
          size="xs"
          dot
          title={`생성 모델: ${gen.provider} · ${gen.model}${gen.detail ? `\n${gen.detail}` : ''}`}
        >
          {gen.model}
        </StatusBadge>
      )}
      <StatusBadge tone={effective.grounding_mode === 'kb' ? 'accent' : 'sky'} size="xs" dot>
        {labelOf(GROUNDING_MODE_LABELS, effective.grounding_mode)}
      </StatusBadge>
      {guardOff ? (
        <StatusBadge
          tone="gray"
          size="xs"
          dot
          title={
            effective.guard_profile === 'none'
              ? '가드 프로필: none'
              : `가드 상태: ${config.guards.status}${effective.guard_profile ? ` (프로필 ${effective.guard_profile})` : ''}`
          }
        >
          가드레일 미적용
        </StatusBadge>
      ) : (
        <StatusBadge tone="green" size="xs" dot>
          가드레일: {effective.guard_profile}
        </StatusBadge>
      )}
    </>
  );
}

function Header({ configState, effective, settingsOpen, canReset, onNewChat, onToggleSettings }: HeaderProps) {
  return (
    <header className="flex-none border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 flex-none items-center justify-center rounded-xl bg-indigo-600 text-white shadow-sm">
            <BookIcon className="h-[18px] w-[18px]" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold leading-tight text-slate-900 sm:text-lg">AI 튜터</h1>
            <p className="truncate text-xs text-slate-500">교육 자료 기반 학습 도우미</p>
          </div>
          <div className="flex flex-none items-center gap-1.5">
            <button
              type="button"
              onClick={onNewChat}
              disabled={!canReset}
              title="대화 내용을 지우고 새로 시작합니다"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <NewChatIcon className="h-4 w-4" />
              <span className="hidden min-[400px]:inline">새 대화</span>
              <span className="sr-only min-[400px]:hidden">새 대화</span>
            </button>
            <button
              type="button"
              onClick={onToggleSettings}
              aria-expanded={settingsOpen}
              aria-controls="dev-settings"
              aria-label="개발용 설정"
              title="개발용 설정"
              className={`rounded-lg border p-2 shadow-sm transition ${
                settingsOpen
                  ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                  : 'border-slate-200 bg-white text-slate-500 hover:bg-slate-50 hover:text-slate-700'
              }`}
            >
              <GearIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
        {configState.status !== 'error' && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <HeaderBadges configState={configState} effective={effective} />
          </div>
        )}
      </div>
    </header>
  );
}

/* ---------------- 앱 (app) ---------------- */

export default function App() {
  const [configState, setConfigState] = useState<ConfigState>({ status: 'loading', config: null });
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selection, setSelection] = useState<DevSelection>(() => loadSelection());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [pending, setPending] = useState(false);

  // 최신 메시지 목록을 동기적으로 읽기 위한 ref (avoid stale closure)
  const messagesRef = useRef<UiMessage[]>([]);
  // 이중 전송 방지 (double-submit guard)
  const pendingRef = useRef(false);
  // "새 대화" 시 진행 중 응답을 무시하기 위한 세대 번호 (conversation epoch)
  const epochRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const configAbortRef = useRef<AbortController | null>(null);

  const config = configState.status === 'ready' ? configState.config : null;
  const effective = useMemo(() => resolveEffective(config, selection), [config, selection]);
  const maxChars =
    config && Number.isFinite(config.limits?.max_message_chars) && config.limits.max_message_chars > 0
      ? config.limits.max_message_chars
      : DEFAULT_MAX_CHARS;

  const commitMessages = useCallback((updater: (prev: UiMessage[]) => UiMessage[]) => {
    const next = updater(messagesRef.current);
    messagesRef.current = next;
    setMessages(next);
  }, []);

  /* ----- 서버 설정 로드 (config) ----- */
  const loadConfig = useCallback(() => {
    configAbortRef.current?.abort();
    const controller = new AbortController();
    configAbortRef.current = controller;
    setConfigState({ status: 'loading', config: null });

    getConfig(controller.signal)
      .then((cfg) => {
        if (!controller.signal.aborted) setConfigState({ status: 'ready', config: cfg });
      })
      .catch(() => {
        if (!controller.signal.aborted) setConfigState({ status: 'error', config: null });
      });

    // 서버 버전 표시용 (optional) — 실패해도 무시
    getHealth(controller.signal)
      .then((h) => {
        if (!controller.signal.aborted) setHealth(h);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    loadConfig();
    return () => configAbortRef.current?.abort();
  }, [loadConfig]);

  /* ----- 개발용 설정 (dev settings) ----- */
  const updateSelection = useCallback((patch: Partial<DevSelection>) => {
    setSelection((prev) => {
      const next = { ...prev, ...patch };
      saveSelection(next);
      return next;
    });
  }, []);

  const resetSelection = useCallback(() => {
    setSelection(EMPTY_SELECTION);
    saveSelection(EMPTY_SELECTION);
  }, []);

  /* ----- 채팅 요청 (chat request) ----- */
  const runRequest = useCallback(
    async (history: ChatMessage[], userMessageId: string) => {
      pendingRef.current = true;
      setPending(true);
      const epoch = epochRef.current;
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await sendChat(
          {
            messages: history,
            answer_mode: effective?.answer_mode ?? null,
            grounding_mode: effective?.grounding_mode ?? null,
            guard_profile: effective?.guard_profile ?? null,
          },
          controller.signal,
        );
        if (epoch !== epochRef.current) return;
        commitMessages((prev) => [
          ...prev.map((m) => (m.id === userMessageId && m.kind === 'user' ? { ...m, state: 'sent' as const } : m)),
          { id: newId(), kind: 'assistant', response },
        ]);
      } catch (err) {
        if (epoch !== epochRef.current) return;
        const apiError =
          err instanceof ApiError
            ? err
            : new ApiError('알 수 없는 오류가 발생했습니다. 다시 시도해 주세요.', 0, 'unknown_error');
        commitMessages((prev) => [
          ...prev.map((m) => (m.id === userMessageId && m.kind === 'user' ? { ...m, state: 'failed' as const } : m)),
          {
            id: newId(),
            kind: 'error',
            userMessageId,
            message: apiError.message,
            status: apiError.status,
            code: apiError.code,
            traceId: apiError.traceId ?? null,
          },
        ]);
      } finally {
        if (epoch === epochRef.current) {
          pendingRef.current = false;
          abortRef.current = null;
          setPending(false);
        }
      }
    },
    [effective, commitMessages],
  );

  /** 새 질문 전송. 전송을 시작했으면 true. */
  const handleSend = useCallback(
    (raw: string): boolean => {
      const text = raw.trim();
      if (pendingRef.current || text === '' || Array.from(text).length > maxChars) return false;

      const history = [...toHistory(messagesRef.current), { role: 'user' as const, content: text }];
      const userMessage: UserUiMessage = { id: newId(), kind: 'user', content: text, state: 'pending' };
      commitMessages((prev) => [...prev, userMessage]);
      void runRequest(history, userMessage.id);
      return true;
    },
    [maxChars, commitMessages, runRequest],
  );

  /** 실패한 질문 다시 보내기: 실패 교환을 지우고 같은 질문을 현재 대화 끝에 재전송 */
  const handleRetry = useCallback(
    (errorId: string) => {
      if (pendingRef.current) return;
      const current = messagesRef.current;
      const errorMessage = current.find((m) => m.id === errorId);
      if (!errorMessage || errorMessage.kind !== 'error') return;
      const failedUser = current.find((m) => m.id === errorMessage.userMessageId);
      if (!failedUser || failedUser.kind !== 'user') return;

      const remaining = current.filter((m) => m.id !== errorMessage.id && m.id !== failedUser.id);
      const history = [...toHistory(remaining), { role: 'user' as const, content: failedUser.content }];
      const retryMessage: UserUiMessage = { id: newId(), kind: 'user', content: failedUser.content, state: 'pending' };
      commitMessages(() => [...remaining, retryMessage]);
      void runRequest(history, retryMessage.id);
    },
    [commitMessages, runRequest],
  );

  /** 새 대화: 전체 대화 삭제 (context reset) + 진행 중 요청 취소 */
  const handleNewChat = useCallback(() => {
    epochRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    pendingRef.current = false;
    setPending(false);
    commitMessages(() => []);
  }, [commitMessages]);

  const showGeneralNotice = effective?.grounding_mode === 'general';

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <Header
        configState={configState}
        effective={effective}
        settingsOpen={settingsOpen}
        canReset={messages.length > 0 || pending}
        onNewChat={handleNewChat}
        onToggleSettings={() => setSettingsOpen((v) => !v)}
      />

      {configState.status === 'error' && (
        <div className="flex-none border-b border-amber-200 bg-amber-50">
          <div className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 py-2 text-xs text-amber-800 sm:px-6">
            <AlertCircleIcon className="h-4 w-4 flex-none text-amber-600" />
            <span className="min-w-0 flex-1">서버 설정을 불러오지 못했습니다</span>
            <button
              type="button"
              onClick={loadConfig}
              className="inline-flex flex-none items-center gap-1 rounded-md border border-amber-300 bg-white px-2 py-1 font-medium text-amber-800 transition hover:bg-amber-100"
            >
              <RefreshIcon className="h-3 w-3" />
              다시 시도
            </button>
          </div>
        </div>
      )}

      {showGeneralNotice && (
        <div className="flex-none border-b border-sky-100 bg-sky-50/70">
          <div className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 py-2 text-xs text-sky-800 sm:px-6">
            <InfoIcon className="h-4 w-4 flex-none text-sky-600" />
            <span>교육 자료 미적용: 일반 지식 기반 답변</span>
          </div>
        </div>
      )}

      {settingsOpen && (
        <DevSettings
          config={config}
          health={health}
          selection={selection}
          effective={effective}
          onChange={updateSelection}
          onReset={resetSelection}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      <ChatWindow messages={messages} pending={pending} onRetry={handleRetry} onExample={(q) => handleSend(q)} />

      <Composer disabled={pending} maxChars={maxChars} onSend={handleSend} />
    </div>
  );
}
