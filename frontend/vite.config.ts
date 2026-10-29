import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// 개발 서버 설정 (dev server config)
// /api 요청은 VITE_PROXY_TARGET(기본 http://localhost:8000)으로 프록시(proxy)한다.
export default defineConfig(({ mode }) => {
  // .env 파일 + 프로세스 환경변수 로드 (loadEnv, VITE_ prefix only)
  // envDir '.' = 현재 작업 디렉터리(frontend/) — @types/node 없이 process.cwd() 와 동일하게 동작
  const env = loadEnv(mode, '.', 'VITE_');
  const proxyTarget = (env.VITE_PROXY_TARGET || '').trim() || 'http://localhost:8000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: 4173,
    },
    build: {
      rollupOptions: {
        output: {
          // 벤더 청크 분리 (vendor chunk splitting): React / 마크다운·하이라이트
          manualChunks(id: string) {
            if (!id.includes('node_modules')) return undefined;
            if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return 'react';
            return 'markdown';
          },
        },
      },
    },
  };
});
