// 상태 배지 (status badge) — 헤더/메시지 공용
import type { ReactNode } from 'react';

export type BadgeTone = 'slate' | 'gray' | 'accent' | 'amber' | 'green' | 'red' | 'sky' | 'orange';

const TONE_CLASSES: Record<BadgeTone, string> = {
  slate: 'border-slate-200 bg-white text-slate-700',
  gray: 'border-slate-200 bg-slate-100 text-slate-500',
  accent: 'border-indigo-200 bg-indigo-50 text-indigo-700',
  amber: 'border-amber-200 bg-amber-50 text-amber-800',
  green: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  red: 'border-rose-200 bg-rose-50 text-rose-700',
  sky: 'border-sky-200 bg-sky-50 text-sky-700',
  orange: 'border-orange-200 bg-orange-50 text-orange-700',
};

const DOT_CLASSES: Record<BadgeTone, string> = {
  slate: 'bg-slate-400',
  gray: 'bg-slate-400',
  accent: 'bg-indigo-500',
  amber: 'bg-amber-500',
  green: 'bg-emerald-500',
  red: 'bg-rose-500',
  sky: 'bg-sky-500',
  orange: 'bg-orange-500',
};

interface StatusBadgeProps {
  tone?: BadgeTone;
  size?: 'sm' | 'xs';
  /** 왼쪽 색 점 표시 (status dot) */
  dot?: boolean;
  icon?: ReactNode;
  title?: string;
  className?: string;
  children: ReactNode;
}

export default function StatusBadge({
  tone = 'slate',
  size = 'sm',
  dot = false,
  icon,
  title,
  className = '',
  children,
}: StatusBadgeProps) {
  const sizeClass = size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-2 py-0.5 text-[11px]';
  return (
    <span
      title={title}
      className={`inline-flex max-w-full items-center gap-1.5 rounded-full border font-medium leading-none ${sizeClass} ${TONE_CLASSES[tone]} ${className}`}
    >
      {dot && <span className={`h-1.5 w-1.5 flex-none rounded-full ${DOT_CLASSES[tone]}`} aria-hidden="true" />}
      {icon && <span className="flex-none">{icon}</span>}
      <span className="truncate">{children}</span>
    </span>
  );
}
