# 빌드 스테이지
FROM node:18-slim AS builder

WORKDIR /app

# 의존성 설치
COPY package*.json ./
RUN npm install

# 소스 복사 및 빌드
COPY . .
RUN npm run build

# 런타임 스테이지
FROM nginx:alpine

# 빌드된 정적 파일 복사
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx 설정 직접 생성
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
    \
    location /api/ { \
        proxy_pass https://construction-chatbot-api.onrender.com/; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
        proxy_set_header Host $host; \
        proxy_cache_bypass $http_upgrade; \
    } \
}' > /etc/nginx/conf.d/default.conf

# 포트 노출
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]