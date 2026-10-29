// 개발용 설정 저장/해석 (persist & resolve dev settings)
// localStorage 는 사생활 보호 모드 등에서 예외를 던질 수 있으므로 모두 try/catch 로 감싼다.
import type { AnswerMode, ConfigResponse, GroundingMode } from './types';
import type { DevSelection, EffectiveSettings } from './uiTypes';

const STORAGE_KEY = 'ai-tutor.dev-settings.v1';

export const EMPTY_SELECTION: DevSelection = {
  answer_mode: null,
  grounding_mode: null,
  guard_profile: null,
};

const ANSWER_MODES: readonly AnswerMode[] = ['direct', 'hint', 'adaptive'];
const GROUNDING_MODES: readonly GroundingMode[] = ['general', 'kb'];

function isAnswerMode(value: unknown): value is AnswerMode {
  return typeof value === 'string' && (ANSWER_MODES as readonly string[]).includes(value);
}

function isGroundingMode(value: unknown): value is GroundingMode {
  return typeof value === 'string' && (GROUNDING_MODES as readonly string[]).includes(value);
}

/** 저장된 선택값 읽기 (storage 접근 실패 시 빈 선택) */
export function loadSelection(): DevSelection {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY_SELECTION;
    const parsed = JSON.parse(raw) as Record<string, unknown> | null;
    if (typeof parsed !== 'object' || parsed === null) return EMPTY_SELECTION;
    return {
      answer_mode: isAnswerMode(parsed.answer_mode) ? parsed.answer_mode : null,
      grounding_mode: isGroundingMode(parsed.grounding_mode) ? parsed.grounding_mode : null,
      guard_profile:
        typeof parsed.guard_profile === 'string' && parsed.guard_profile !== '' ? parsed.guard_profile : null,
    };
  } catch {
    return EMPTY_SELECTION;
  }
}

/** 선택값 저장 (실패해도 앱 동작에는 영향 없음) */
export function saveSelection(selection: DevSelection): void {
  try {
    const isEmpty =
      selection.answer_mode === null && selection.grounding_mode === null && selection.guard_profile === null;
    if (isEmpty) window.localStorage.removeItem(STORAGE_KEY);
    else window.localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
  } catch {
    // 저장소 사용 불가 — 무시 (storage unavailable, ignore)
  }
}

/**
 * 실제 적용값 계산: 선택값이 서버가 허용하는 목록에 있으면 사용, 아니면 서버 기본값.
 * config 를 못 불러왔으면 null → 요청에 값을 싣지 않고 서버 기본값에 맡긴다.
 */
export function resolveEffective(config: ConfigResponse | null, selection: DevSelection): EffectiveSettings | null {
  if (!config) return null;
  const answerModes = config.answer_modes ?? [];
  const groundingModes = config.grounding_modes ?? [];
  const profiles = config.guards?.available_profiles ?? [];
  return {
    answer_mode:
      selection.answer_mode && answerModes.includes(selection.answer_mode) ? selection.answer_mode : config.answer_mode,
    grounding_mode:
      selection.grounding_mode && groundingModes.includes(selection.grounding_mode)
        ? selection.grounding_mode
        : config.grounding_mode,
    guard_profile:
      selection.guard_profile && profiles.includes(selection.guard_profile)
        ? selection.guard_profile
        : config.guards.profile,
  };
}
