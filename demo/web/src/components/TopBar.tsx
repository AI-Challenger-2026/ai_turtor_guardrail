type Props = { onMenu: () => void; onPanel: () => void; }

export function TopBar({ onMenu, onPanel }: Props) {
  return (
    <header className="topbar">
      <button className="menu-button" onClick={onMenu} aria-label="메뉴 열기">
        ☰
      </button>
      <div className="top-actions">
        <button className="review-trigger" onClick={onPanel}>
          출처 · 가드
        </button>
      </div>
    </header>
  )
}
