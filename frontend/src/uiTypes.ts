// 화면 전용 타입 (UI-only types) — API 계약 타입은 types.ts 를 그대로 사용한다.
import type { AnswerMode, ChatResponse, GroundingMode } from './types';

/** 사용자 메시지 상태: 대기 중 / 성공 / 실패 (실패한 교환은 history 에서 제외) */
export type UserMessageState = 'pending' | 'sent' | 'failed';

export interface UserUiMessage {
  id: string;
  kind: 'user';
  content: string;
  state: UserMessageState;
}

export interface AssistantUiMessage {
  id: string;
  kind: 'assistant';
  response: ChatResponse;
}

export interface ErrorUiMessage {
  id: string;
  kind: 'error';
  /** 실패한 사용자 메시지 id (retry target) */
  userMessageId: string;
  message: string;
  status: number;
  code: string;
  traceId: string | null;
}

export type UiMessage = UserUiMessage | AssistantUiMessage | ErrorUiMessage;

/** 개발용 설정 선택값. null 이면 서버 기본값(config)을 따른다. */
export interface DevSelection {
  answer_mode: AnswerMode | null;
  grounding_mode: GroundingMode | null;
  guard_profile: string | null;
}

/** 실제로 요청에 실릴 값 (effective values) */
export interface EffectiveSettings {
  answer_mode: AnswerMode;
  grounding_mode: GroundingMode;
  guard_profile: string;
}

let idSeq = 0;

/** 메시지 id 생성 — URL hash 에 안전한 영숫자 (alphanumeric, used in #cite-... ids) */
export function newId(): string {
  idSeq += 1;
  return `m${Date.now().toString(36)}${idSeq.toString(36)}`;
}
