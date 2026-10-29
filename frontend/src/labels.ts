// 한국어 라벨과 표시 형식 도우미 (labels & formatters)
import type { AnswerMode, Decision, FinalAction, GroundingMode, ModuleStatus } from './types';

export const ANSWER_MODE_LABELS: Record<AnswerMode, string> = {
  direct: '바로 설명',
  hint: '힌트 중심',
  adaptive: '적응형',
};

/** 헤더 배지용 근거 모드 이름 */
export const GROUNDING_MODE_LABELS: Record<GroundingMode, string> = {
  general: '일반 지식 모드',
  kb: '교육 자료 근거 모드',
};

/** 설정 선택지용 근거 모드 이름 */
export const GROUNDING_OPTION_LABELS: Record<GroundingMode, string> = {
  general: '일반 지식 (general)',
  kb: '교육 자료 근거 (kb)',
};

export const MODULE_STATUS_LABELS: Record<ModuleStatus, string> = {
  enabled: '사용',
  disabled: '꺼짐',
  not_implemented: '미구현',
};

export const DECISION_LABELS: Record<Decision, string> = {
  allow: '통과',
  revise: '수정 요청',
  block: '차단',
};

export const FINAL_ACTION_LABELS: Record<FinalAction, string> = {
  allow: '통과',
  revise: '수정 후 통과',
  block: '차단',
  fallback: '안전 대체 답변',
};

/** 알 수 없는 값이 와도 깨지지 않도록 원문으로 대체 (fallback to raw value) */
export function labelOf(map: Record<string, string>, key: string): string {
  return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : key;
}

/** 지연 시간 표시: 1초 미만은 ms, 이상은 초 단위 */
export function formatLatency(ms: number): string {
  if (!Number.isFinite(ms)) return '-';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}초`;
}

export function formatNumber(value: number, digits: number): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '-';
}

/** 코드 포인트 기준 글자 수 (Python len() 과 동일하게 count code points) */
export function countChars(text: string): number {
  return Array.from(text).length;
}
