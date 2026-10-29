// 개발용 설정 패널 (dev settings) — 서버 config 가 알려주는 값만 표시한다
import type { ReactNode } from 'react';
import type { AnswerMode, ConfigResponse, GroundingMode, HealthResponse, ModuleInfo, ModuleStatus } from '../types';
import type { DevSelection, EffectiveSettings } from '../uiTypes';
import { ANSWER_MODE_LABELS, GROUNDING_OPTION_LABELS, MODULE_STATUS_LABELS, labelOf } from '../labels';
import StatusBadge, { type BadgeTone } from './StatusBadge';
import { GearIcon, XIcon } from './icons';

const MODULE_TONES: Record<ModuleStatus, BadgeTone> = {
  enabled: 'green',
  disabled: 'gray',
  not_implemented: 'amber',
};

function ModuleBadge({ status, detail }: { status: ModuleStatus; detail?: string | null }) {
  const tone = MODULE_TONES[status] ?? 'gray';
  return (
    <StatusBadge tone={tone} size="xs" dot title={detail ?? undefined}>
      {labelOf(MODULE_STATUS_LABELS, status)}
    </StatusBadge>
  );
}

interface SelectFieldProps {
  id: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  hint?: string;
}

function SelectField({ id, label, value, options, onChange, hint }: SelectFieldProps) {
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1 block text-xs font-medium text-slate-600">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {hint && <p className="mt-1 text-[11px] text-slate-400">{hint}</p>}
    </div>
  );
}

function InfoRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 py-1.5 sm:flex-row sm:items-start sm:gap-3">
      <dt className="w-28 flex-none text-[11px] font-medium text-slate-500">{label}</dt>
      <dd className="min-w-0 flex-1 text-xs text-slate-600 [overflow-wrap:anywhere]">{children}</dd>
    </div>
  );
}

interface DevSettingsProps {
  config: ConfigResponse | null;
  health: HealthResponse | null;
  selection: DevSelection;
  effective: EffectiveSettings | null;
  onChange: (patch: Partial<DevSelection>) => void;
  onReset: () => void;
  onClose: () => void;
}

export default function DevSettings({ config, health, selection, effective, onChange, onReset, onClose }: DevSettingsProps) {
  const hasOverride =
    selection.answer_mode !== null || selection.grounding_mode !== null || selection.guard_profile !== null;

  return (
    <section
      id="dev-settings"
      aria-label="개발용 설정"
      className="flex-none border-b border-slate-200 bg-slate-50/80"
    >
      <div className="mx-auto max-h-[55vh] w-full max-w-3xl overflow-y-auto px-4 py-4 sm:px-6">
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-800">
              <GearIcon className="h-4 w-4 text-slate-500" />
              개발용 설정
            </h2>
            <button
              type="button"
              onClick={onClose}
              aria-label="개발용 설정 닫기"
              className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
            >
              <XIcon className="h-4 w-4" />
            </button>
          </div>

          {!config || !effective ? (
            <p className="rounded-lg bg-slate-50 px-3 py-2.5 text-xs text-slate-500">
              서버 설정을 불러오지 못해 개발용 설정을 사용할 수 없습니다. 요청에는 서버 기본값이 적용됩니다.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <SelectField
                  id="dev-answer-mode"
                  label="응답 모드"
                  value={effective.answer_mode}
                  options={config.answer_modes.map((m) => ({
                    value: m,
                    label: `${labelOf(ANSWER_MODE_LABELS, m)}${m === config.answer_mode ? ' (기본)' : ''}`,
                  }))}
                  onChange={(v) => onChange({ answer_mode: v as AnswerMode })}
                />
                <SelectField
                  id="dev-grounding-mode"
                  label="근거 모드"
                  value={effective.grounding_mode}
                  options={config.grounding_modes.map((m) => ({
                    value: m,
                    label: `${labelOf(GROUNDING_OPTION_LABELS, m)}${m === config.grounding_mode ? ' (기본)' : ''}`,
                  }))}
                  onChange={(v) => onChange({ grounding_mode: v as GroundingMode })}
                  hint={
                    effective.grounding_mode === 'kb' && config.kb.status !== 'enabled'
                      ? `지식베이스 상태: ${labelOf(MODULE_STATUS_LABELS, config.kb.status)}`
                      : undefined
                  }
                />
                <SelectField
                  id="dev-guard-profile"
                  label="가드 프로필"
                  value={effective.guard_profile}
                  options={config.guards.available_profiles.map((p) => ({
                    value: p,
                    label: `${p === 'none' ? 'none (미적용)' : p}${p === config.guards.profile ? ' (기본)' : ''}`,
                  }))}
                  onChange={(v) => onChange({ guard_profile: v })}
                />
              </div>

              <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                <p className="text-[11px] text-slate-400">선택값은 이 브라우저에 저장되며 모든 질문에 함께 전송됩니다.</p>
                <button
                  type="button"
                  onClick={onReset}
                  disabled={!hasOverride}
                  className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  서버 기본값으로
                </button>
              </div>

              <dl className="mt-4 divide-y divide-slate-100 border-t border-slate-100 pt-2">
                <InfoRow label="지식베이스 (KB)">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <ModuleBadge status={config.kb.status} detail={config.kb.detail} />
                    <span>
                      문서 {config.kb.num_documents}개 · 청크 {config.kb.num_chunks}개
                    </span>
                    {config.kb.stale && (
                      <StatusBadge tone="amber" size="xs">
                        재빌드 필요
                      </StatusBadge>
                    )}
                  </div>
                  {config.kb.units.length > 0 && (
                    <p className="mt-1 text-slate-500">단원: {config.kb.units.join(', ')}</p>
                  )}
                  {config.kb.built_at && <p className="mt-0.5 text-slate-400">빌드: {config.kb.built_at}</p>}
                </InfoRow>

                <InfoRow label="검색 모듈">
                  <div className="flex flex-wrap gap-x-3 gap-y-1.5">
                    {(
                      [
                        ['bm25', config.retrieval.bm25],
                        ['dense', config.retrieval.dense],
                        ['hybrid', config.retrieval.hybrid],
                        ['query_rewrite', config.retrieval.query_rewrite],
                      ] as [string, ModuleInfo][]
                    ).map(([name, info]) => (
                      <span key={name} className="inline-flex items-center gap-1.5" title={info.detail ?? undefined}>
                        <span className="font-mono text-[11px] text-slate-600">{name}</span>
                        <ModuleBadge status={info.status} detail={info.detail} />
                      </span>
                    ))}
                  </div>
                </InfoRow>

                <InfoRow label="모델">
                  <div className="space-y-1">
                    {(
                      [
                        ['생성', config.generator],
                        ['판정', config.judge],
                      ] as const
                    ).map(([label, info]) => (
                      <div key={label} className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span className="text-slate-500">{label}</span>
                        <span className="font-mono text-[11px] text-slate-700">
                          {info.provider} · {info.model}
                        </span>
                        <ModuleBadge status={info.status} detail={info.detail} />
                      </div>
                    ))}
                  </div>
                </InfoRow>

                <InfoRow label="활성 가드">
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    <span className="text-slate-500">
                      서버 기본 프로필 <span className="font-mono text-slate-700">{config.guards.profile}</span>
                    </span>
                    <ModuleBadge status={config.guards.status} />
                  </div>
                  {config.guards.input.length === 0 && config.guards.output.length === 0 ? (
                    <p className="text-slate-400">적용된 가드가 없습니다.</p>
                  ) : (
                    <div className="space-y-1">
                      {(
                        [
                          ['입력', config.guards.input],
                          ['출력', config.guards.output],
                        ] as const
                      ).map(([label, guards]) => (
                        <div key={label} className="flex flex-wrap items-center gap-1.5">
                          <span className="w-7 flex-none text-slate-500">{label}</span>
                          {guards.length === 0 ? (
                            <span className="text-slate-400">없음</span>
                          ) : (
                            guards.map((g) => (
                              <span
                                key={`${g.stage}-${g.name}`}
                                title={g.kind}
                                className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-mono text-[11px] text-slate-600"
                              >
                                {g.name}
                              </span>
                            ))
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {effective.guard_profile !== config.guards.profile && (
                    <p className="mt-1 text-[11px] text-slate-400">
                      선택한 프로필({effective.guard_profile})의 실제 가드 구성은 답변의 &quot;판정 보기&quot;에서
                      확인할 수 있습니다.
                    </p>
                  )}
                </InfoRow>

                <InfoRow label="제한">
                  메시지 최대 {config.limits.max_message_chars.toLocaleString('ko-KR')}자 · 대화 최대{' '}
                  {config.limits.max_turns}턴
                </InfoRow>

                {health && (
                  <InfoRow label="서버">
                    v{health.version}
                    {health.mock_mode && <span className="ml-1.5 text-amber-600">(mock)</span>}
                  </InfoRow>
                )}
              </dl>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
