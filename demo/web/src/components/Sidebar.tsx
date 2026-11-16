import { learningHistory } from "../mockData"
type Props = {
  open: boolean
  onClose: () => void
  onReset: () => void
  hasAnswer: boolean
}
export function Sidebar({ open, onClose, onReset, hasAnswer }: Props) {
  return (
    <aside className={`sidebar ${open ? "is-open" : ""}`}>
      <div>
        <div className="brand">
          <img src="assets/928fb.svg" alt="" />
          <span>tutor</span>
          <button
            className="mobile-close"
            onClick={onClose}
            aria-label="메뉴 닫기"
          >
            ×
          </button>
        </div>
        <button className="new-study" onClick={onReset}>
          <span>＋</span>
          <b>새 학습</b>
        </button>
        <div className="history">
          <p className="section-label">학습 기록</p>
          {learningHistory.map((item) => (
            <button
              className={`history-item ${
                hasAnswer && item.label === "파이썬 기초" ? "active" : ""
              }`}
              key={item.label}
            >
              <span>▢</span>
              <b>{item.label}</b>
            </button>
          ))}
          <button className="history-item">
            <span>◴</span>
            <b>전체 기록 보기</b>
          </button>
        </div>
      </div>
      <div className="profile">
        <span className="avatar">♟</span>
        <span className="profile-copy">
          <b>학습자 </b>
          <small>시연 계정</small>
        </span>
      </div>
    </aside>
  )
}
