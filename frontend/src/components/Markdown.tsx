// 답변 마크다운 렌더러 (react-markdown + GFM + highlight.js)
// 본문의 [n] 인용 표시를 출처 카드로 이동하는 링크로 바꾼다 (코드 블록/인라인 코드는 제외).
import { memo, useMemo, useRef, useState, type ComponentPropsWithoutRef, type MouseEvent } from 'react';
import ReactMarkdown, { type Components, type Options } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CheckIcon, CopyIcon } from './icons';

// 플러그인 배열은 모듈 상수로 고정해 불필요한 재처리를 막는다 (stable plugin arrays)
// rehype-highlight 기본값: highlight.js common 언어 세트, 언어 표기된 코드 블록만 하이라이트
const REMARK_PLUGINS: Options['remarkPlugins'] = [remarkGfm];
const REHYPE_PLUGINS: Options['rehypePlugins'] = [rehypeHighlight];

/** 출처 카드 DOM id (anchor id) */
export function citeAnchorId(messageId: string, index: number): string {
  return `cite-${messageId}-${index}`;
}

// 연속된 인용 묶음: [1] / [1][2] / [1, 2] (앞이 영문자·숫자·]·\·! 이면 제외 → arr[1], [a][1], ![1] 방지)
const CITE_GROUP_RE = /(?<![\]\\!\w])((?:\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\])+)(?!\()/g;
const SINGLE_GROUP_RE = /\[[^\]]*\]/g;
// 펜스 코드 블록 시작/끝 (fenced code, 목록·인용문 들여쓰기 허용)
const FENCE_OPEN_RE = /^\s*(?:>\s*)*(`{3,}|~{3,})/;
const FENCE_CLOSE_RE = /^\s*(?:>\s*)*(`{3,}|~{3,})\s*$/;

/** 일반 텍스트 구간의 [n] 을 링크 마크다운으로 치환 */
function linkCitations(text: string, messageId: string, valid: ReadonlySet<number>): string {
  return text.replace(CITE_GROUP_RE, (match: string, _group: string, offset: number, whole: string) => {
    // 줄 맨 앞의 "[1]: url" 은 참조 정의(reference definition)이므로 그대로 둔다
    const lineStart = whole.lastIndexOf('\n', offset - 1) + 1;
    const after = whole.slice(offset + match.length);
    if (after.startsWith(':') && whole.slice(lineStart, offset).trim() === '') return match;

    const groups = match.match(SINGLE_GROUP_RE) ?? [];
    return groups
      .map((group) => {
        const nums = group
          .slice(1, -1)
          .split(',')
          .map((s) => Number(s.trim()));
        if (nums.length === 0 || !nums.every((n) => Number.isInteger(n) && valid.has(n))) return group;
        return nums.map((n) => `[\\[${n}\\]](#${citeAnchorId(messageId, n)})`).join('');
      })
      .join('');
  });
}

/** 인라인 코드(`...`)를 건너뛰고 나머지 텍스트에만 변환 적용 (skip inline code spans) */
function transformOutsideInlineCode(text: string, fn: (plain: string) => string): string {
  let out = '';
  let plainStart = 0;
  let i = 0;
  while (i < text.length) {
    if (text[i] !== '`') {
      i += 1;
      continue;
    }
    let j = i;
    while (j < text.length && text[j] === '`') j += 1;
    const runLength = j - i;
    // 같은 길이의 닫는 backtick 묶음 찾기 (matching backtick run)
    let k = j;
    let closeAt = -1;
    while (k < text.length) {
      if (text[k] === '`') {
        let m = k;
        while (m < text.length && text[m] === '`') m += 1;
        if (m - k === runLength) {
          closeAt = k;
          break;
        }
        k = m;
      } else {
        k += 1;
      }
    }
    if (closeAt === -1) {
      i = j;
      continue;
    }
    out += fn(text.slice(plainStart, i)) + text.slice(i, closeAt + runLength);
    i = closeAt + runLength;
    plainStart = i;
  }
  return out + fn(text.slice(plainStart));
}

/** 펜스 코드 블록 바깥에서만 인용 링크 변환 (outside fenced code) */
export function preprocessCitations(markdown: string, messageId: string, valid: ReadonlySet<number>): string {
  if (valid.size === 0 || !markdown.includes('[')) return markdown;

  const transformPlain = (chunk: string) =>
    chunk
      // 빈 줄 단위(문단)로 나눠 인라인 코드 짝이 문단을 넘지 않게 한다
      .split(/(\n[ \t]*\n)/)
      .map((part) =>
        /^\n[ \t]*\n$/.test(part)
          ? part
          : transformOutsideInlineCode(part, (plain) => linkCitations(plain, messageId, valid)),
      )
      .join('');

  const out: string[] = [];
  let buffer: string[] = [];
  let fence: { char: string; length: number } | null = null;

  const flush = () => {
    if (buffer.length > 0) {
      out.push(transformPlain(buffer.join('\n')));
      buffer = [];
    }
  };

  for (const line of markdown.split('\n')) {
    if (fence) {
      out.push(line);
      const close = FENCE_CLOSE_RE.exec(line);
      if (close && close[1][0] === fence.char && close[1].length >= fence.length) fence = null;
      continue;
    }
    const open = FENCE_OPEN_RE.exec(line);
    if (open) {
      flush();
      fence = { char: open[1][0], length: open[1].length };
      out.push(line);
      continue;
    }
    buffer.push(line);
  }
  flush();
  return out.join('\n');
}

/** 코드 블록 + 복사 버튼 (copy to clipboard) */
function CodeBlock(props: ComponentPropsWithoutRef<'pre'>) {
  const preRef = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const text = preRef.current?.textContent ?? '';
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // 클립보드 권한 없음/비보안 컨텍스트 — 조용히 무시 (clipboard unavailable)
    }
  };

  return (
    <div className="group relative">
      <pre ref={preRef} {...props} />
      <button
        type="button"
        onClick={handleCopy}
        className="absolute right-2 top-2 inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white/90 px-2 py-1 text-[11px] font-medium text-slate-500 opacity-100 shadow-sm transition hover:text-slate-800 focus:opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
        aria-label="코드 복사"
      >
        {copied ? <CheckIcon className="h-3.5 w-3.5" /> : <CopyIcon className="h-3.5 w-3.5" />}
        {copied ? '복사됨' : '복사'}
      </button>
    </div>
  );
}

interface MarkdownProps {
  content: string;
  messageId: string;
  /** 본문 [n] 과 연결할 인용 번호 목록 */
  citationIndexes?: readonly number[];
  onCite?: (index: number) => void;
}

function MarkdownImpl({ content, messageId, citationIndexes, onCite }: MarkdownProps) {
  const indexKey = citationIndexes ? citationIndexes.join(',') : '';

  const processed = useMemo(() => {
    const valid = new Set<number>(indexKey ? indexKey.split(',').map(Number) : []);
    return preprocessCitations(content, messageId, valid);
  }, [content, messageId, indexKey]);

  const components = useMemo<Components>(() => {
    const citeHashPrefix = `#cite-${messageId}-`;
    return {
      a: ({ node: _node, href, children, ...rest }) => {
        if (href && href.startsWith(citeHashPrefix)) {
          const index = Number(href.slice(citeHashPrefix.length));
          const handleClick = (e: MouseEvent<HTMLAnchorElement>) => {
            // 기본 hash 이동 대신 해당 출처 카드로 스크롤 + 펼치기 (preventDefault + scrollIntoView)
            e.preventDefault();
            onCite?.(index);
          };
          return (
            <sup className="mx-0.5">
              <a
                href={href}
                onClick={handleClick}
                className="rounded px-0.5 text-[0.8em] font-semibold text-indigo-600 no-underline hover:bg-indigo-50 hover:text-indigo-800"
                aria-label={`출처 ${index}번 보기`}
              >
                {children}
              </a>
            </sup>
          );
        }
        // 외부 링크는 새 탭으로 (external links)
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
            {children}
          </a>
        );
      },
      pre: ({ node: _node, ...rest }) => <CodeBlock {...rest} />,
      table: ({ node: _node, ...rest }) => (
        <div className="table-wrap">
          <table {...rest} />
        </div>
      ),
    };
  }, [messageId, onCite]);

  return (
    <div className="tutor-prose">
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS} components={components}>
        {processed}
      </ReactMarkdown>
    </div>
  );
}

const Markdown = memo(MarkdownImpl);
export default Markdown;
