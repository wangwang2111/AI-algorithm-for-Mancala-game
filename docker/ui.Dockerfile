# ===== UI builder (React + Vite) =====
FROM node:20-alpine AS ui-build

ARG API_BASE=http://localhost:8000/api
WORKDIR /app

COPY web/package*.json ./
RUN npm ci

COPY web/ ./
# Optional runtime env for apps that read window.API_BASE
RUN mkdir -p public && \
    printf "window.API_BASE=%s;\n" "$(printf '%s' "\"${API_BASE}\"")" > public/env.js

RUN npm run build


# ===== Static UI (Nginx) =====
FROM nginx:1.27-alpine

# Copy built assets
COPY --from=ui-build /app/dist /usr/share/nginx/html

# Replace default site
COPY docker/nginx.ui.conf /etc/nginx/conf.d/default.conf

# (Optional) healthcheck curl
RUN apk add --no-cache curl

EXPOSE 8080
STOPSIGNAL SIGQUIT
CMD ["nginx", "-g", "daemon off;"]
