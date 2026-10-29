// 백엔드 API 클라이언트 (fetch wrapper)
// - API 키는 절대 프론트엔드에 두지 않는다 (no secrets in frontend).
// - 오류는 항상 한국어 메시지를 가진 ApiError 로 변환한다.
import type {
  ApiErrorBody,
  ChatRequest,
  ChatResponse,
  ConfigResponse,
  HealthResponse,
} from '../types';

/** 백엔드 기본 주소. 비어 있으면 상대 경로(/api)로 Vite proxy 를 사용한다. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '');

export const CHAT_TIMEOUT_MS = 120_000;
export const CONFIG_TIMEOUT_MS = 10_000;

const MSG_TIMEOUT = '응답 시간이 초과되었습니다. 다시 시도해 주세요.';
const MSG_NETWORK = '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해 주세요.';
const MSG_ABORTED = '요청이 취소되었습니다.';
const MSG_INVALID = '서버 응답을 해석할 수 없습니다. 잠시 후 다시 시도해 주세요.';

/** API 오류 (status 0 = 네트워크 오류/타임아웃/취소처럼 HTTP 응답이 없는 경우) */
export class ApiError extends Error {
  status: number;
  code: string;
  traceId?: string | null;

  constructor(message: string, status: number, code: string, traceId?: string | null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.traceId = traceId ?? null;
  }
}

/** HTTP 상태 코드별 기본 한국어 메시지 (fallback when body is not ApiErrorBody) */
export function messageForStatus(status: number): string {
  switch (status) {
    case 400:
      return '요청 형식이 올바르지 않습니다. 입력 내용을 확인해 주세요.';
    case 413:
      return '메시지 또는 대화가 너무 깁니다. 내용을 줄이거나 새 대화를 시작해 주세요.';
    case 422:
      return '요청 내용을 처리할 수 없습니다. 입력과 설정을 확인해 주세요.';
    case 429:
      return '요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.';
    case 500:
      return '서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.';
    case 501:
      return '아직 지원하지 않는 기능입니다.';
    case 502:
      return 'AI 모델 서버와 통신하는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.';
    case 503:
      return '서비스를 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.';
    case 504:
      return MSG_TIMEOUT;
    default:
      return `요청을 처리하는 중 오류가 발생했습니다. (HTTP ${status})`;
  }
}

/** 오류 본문 타입 가드 (type guard for ApiErrorBody) */
function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== 'object' || value === null) return false;
  const err = (value as { error?: unknown }).error;
  if (typeof err !== 'object' || err === null) return false;
  const e = err as { code?: unknown; message?: unknown };
  return typeof e.message === 'string' && e.message.trim() !== '' && typeof e.code === 'string';
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

interface RequestOptions {
  method: 'GET' | 'POST';
  body?: unknown;
  timeoutMs: number;
  /** 호출 측 취소 신호 (caller-side cancellation, e.g. "새 대화") */
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;

  // 타임아웃 시 요청 중단 (AbortController timeout)
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, options.timeoutMs);

  const external = options.signal;
  const onExternalAbort = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener('abort', onExternalAbort, { once: true });
  }

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (options.body !== undefined) headers['Content-Type'] = 'application/json';

  try {
    let response: Response;
    let text: string;
    try {
      response = await fetch(`${API_BASE}${path}`, {
        method: options.method,
        headers,
        body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
      });
      // 본문 수신 중에도 타임아웃이 적용되도록 같은 try 안에서 읽는다 (read body under timeout)
      text = await response.text();
    } catch {
      if (timedOut) throw new ApiError(MSG_TIMEOUT, 0, 'timeout');
      if (external?.aborted) throw new ApiError(MSG_ABORTED, 0, 'aborted');
      throw new ApiError(MSG_NETWORK, 0, 'network_error');
    }

    if (!response.ok) {
      const parsed = safeJsonParse(text);
      if (isApiErrorBody(parsed)) {
        throw new ApiError(parsed.error.message, response.status, parsed.error.code, parsed.error.trace_id);
      }
      // Vite proxy 는 백엔드 연결 실패 시 빈 본문의 500 을 돌려준다 (proxy connection error)
      if (API_BASE === '' && response.status === 500 && text.trim() === '') {
        throw new ApiError(MSG_NETWORK, response.status, 'proxy_error');
      }
      throw new ApiError(messageForStatus(response.status), response.status, `http_${response.status}`);
    }

    const data = safeJsonParse(text);
    if (data === undefined || data === null || typeof data !== 'object') {
      throw new ApiError(MSG_INVALID, response.status, 'invalid_response');
    }
    return data as T;
  } finally {
    clearTimeout(timer);
    external?.removeEventListener('abort', onExternalAbort);
  }
}

/** 서버 상태 확인 (GET /api/health) */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health', { method: 'GET', timeoutMs: CONFIG_TIMEOUT_MS, signal });
}

/** 서버 설정 조회 (GET /api/config) */
export function getConfig(signal?: AbortSignal): Promise<ConfigResponse> {
  return request<ConfigResponse>('/api/config', { method: 'GET', timeoutMs: CONFIG_TIMEOUT_MS, signal });
}

/** 채팅 요청 (POST /api/chat) — 백엔드는 무상태(stateless)이므로 전체 대화를 보낸다 */
export async function sendChat(req: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  const data = await request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: req,
    timeoutMs: CHAT_TIMEOUT_MS,
    signal,
  });
  // 최소한의 응답 형태 검증 (minimal shape check) — 렌더링 중 예외 방지
  if (typeof data.answer !== 'string') {
    throw new ApiError(MSG_INVALID, 200, 'invalid_response', data.trace_id ?? null);
  }
  return {
    ...data,
    citations: Array.isArray(data.citations) ? data.citations : [],
  };
}
