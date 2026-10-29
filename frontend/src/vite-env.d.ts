/// <reference types="vite/client" />

// Vite 환경변수 타입 선언 (import.meta.env typing)
interface ImportMetaEnv {
  /** 백엔드 주소. 비워두면 상대 경로(/api)로 Vite proxy 를 사용한다. */
  readonly VITE_API_BASE_URL?: string;
  /** Vite 개발 서버 proxy 대상 (dev server only) */
  readonly VITE_PROXY_TARGET?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
