# ===== Static UI (Nginx) =====
FROM nginx:1.27-alpine AS ui

# Build-time config for API base url written into env.js
ARG API_BASE=http://localhost:8000

# Where your static files live (index.html, /js, /css, /public, etc.)
ARG UI_DIR=ui_static

WORKDIR /usr/share/nginx/html
COPY ${UI_DIR}/ ./

# Nginx config: serve static on 8080 and proxy /api/* to the API service
# Compose gives us DNS to reach "api" by name.
RUN rm -f /etc/nginx/conf.d/default.conf && \
    printf '%s\n' \
    "server {" \
    "  listen 8080;" \
    "  server_name _;" \
    "  root /usr/share/nginx/html;" \
    "  index index.html;" \
    "  sendfile on;" \
    "  gzip on;" \
    "  gzip_types text/plain application/javascript application/json text/css image/svg+xml;" \
    "" \
    "  # Static files" \
    "  location / {" \
    "    try_files \$uri \$uri/ /index.html;" \
    "  }" \
    "" \
    "  # Proxy API calls to the api service on port 8000" \
    "  location /api/ {" \
    "    proxy_http_version 1.1;" \
    "    proxy_set_header Host \$host;" \
    "    proxy_set_header X-Real-IP \$remote_addr;" \
    "    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;" \
    "    proxy_set_header X-Forwarded-Proto \$scheme;" \
    "    proxy_pass http://api:8000/api/;" \
    "  }" \
    "" \
    "  location ~* \.(js|css|png|jpg|jpeg|gif|svg|mp3|woff2?)$ {" \
    "    expires 7d;" \
    "    add_header Cache-Control \"public, max-age=604800, immutable\";" \
    "  }" \
    "}" \
    > /etc/nginx/conf.d/ui.conf

EXPOSE 8080
