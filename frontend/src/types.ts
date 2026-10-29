// 백엔드 API 계약 (API contract) — backend/app/schemas.py 와 1:1로 맞춘다.
// 필드를 바꿀 때는 양쪽을 함께 수정한다 (keep in sync with Pydantic models).

export type Role = 'user' | 'assistant';
export type AnswerMode = 'direct' | 'hint' | 'adaptive';
export type GroundingMode = 'general' | 'kb';
export type ModuleStatus = 'enabled' | 'disabled' | 'not_implemented';
export type Decision = 'allow' | 'block' | 'revise';
export type TurnStatus = 'answered' | 'revised' | 'blocked' | 'no_evidence' | 'fallback';
export type FinalAction = 'allow' | 'block' | 'revise' | 'fallback';

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  answer_mode?: AnswerMode | null;
  guard_profile?: string | null;
  grounding_mode?: GroundingMode | null;
}

/** 출처 카드 (source citation). index 는 답변 본문의 [n] 번호와 같다. */
export interface Citation {
  index: number;
  doc_id: string;
  doc_title: string;
  unit: string;
  section: string;
  chunk_id: string;
  text: string;
  score: number | null;
}

/** 가드 하나의 판정 결과 (single guard verdict). */
export interface GuardResult {
  name: string;
  stage: 'input' | 'output';
  decision: Decision;
  category: string;
  reason: string;
  score: number;
  latency_ms: number;
  /** 출력 가드: 0 = 최초 답변, 1 = 재생성 답변 (regeneration attempt) */
  attempt: number;
  details: Record<string, unknown>;
}

export interface GuardTrace {
  profile: string;
  input: GuardResult[];
  output: GuardResult[];
  regenerations: number;
  final_action: FinalAction;
}

export interface LLMCall {
  task: string;
  role: string;
  provider: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
  ok: boolean;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  llm_calls: number;
  estimated_cost_usd: number;
  calls: LLMCall[];
}

/** 이번 턴에 실제로 적용된 처리 정보 (only what actually ran). */
export interface TurnMeta {
  rewritten_query: string | null;
  rewrite_applied: boolean;
  question_type: string | null;
  learner_level: string | null;
  retrieved_count: number;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  grounding: GroundingMode;
  /** kb 모드에서 근거 청크를 찾았는지. general 모드에서는 null */
  grounded: boolean | null;
  status: TurnStatus;
  blocked: boolean;
  block_category: string | null;
  /** 가드 프로필이 none 이면 null (guardrails not applied) */
  guard_trace: GuardTrace | null;
  provider: string;
  model: string;
  answer_mode: AnswerMode;
  latency_ms: number;
  usage: Usage;
  meta: TurnMeta;
  trace_id: string;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    trace_id?: string | null;
  };
}

export interface ModuleInfo {
  status: ModuleStatus;
  detail: string | null;
}

export interface ProviderInfo {
  /** 실제로 호출되는 제공자 (MOCK_MODE 이면 "mock") */
  provider: string;
  model: string;
  /** 환경변수에 설정된 제공자/모델 (configured values) */
  configured_provider: string;
  configured_model: string;
  status: ModuleStatus;
  detail: string | null;
}

export interface GuardInfo {
  name: string;
  stage: 'input' | 'output';
  kind: string;
}

export interface ConfigResponse {
  mock_mode: boolean;
  generator: ProviderInfo;
  judge: ProviderInfo;
  answer_mode: AnswerMode;
  answer_modes: AnswerMode[];
  grounding_mode: GroundingMode;
  grounding_modes: GroundingMode[];
  kb: {
    status: ModuleStatus;
    detail: string | null;
    num_documents: number;
    num_chunks: number;
    units: string[];
    built_at: string | null;
    stale: boolean;
  };
  retrieval: {
    bm25: ModuleInfo;
    dense: ModuleInfo;
    hybrid: ModuleInfo;
    query_rewrite: ModuleInfo;
  };
  guards: {
    status: ModuleStatus;
    profile: string;
    available_profiles: string[];
    input: GuardInfo[];
    output: GuardInfo[];
  };
  limits: {
    max_message_chars: number;
    max_turns: number;
    max_request_bytes: number;
  };
}

export interface HealthResponse {
  status: 'ok';
  version: string;
  mock_mode: boolean;
}
