FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG VITE_OIDC_URL=http://localhost:8080
ENV VITE_OIDC_URL=$VITE_OIDC_URL
RUN npm run build

FROM caddy:2.8-alpine
COPY --from=build /web/dist /srv
COPY Caddyfile /etc/caddy/Caddyfile
USER 1000
EXPOSE 8080
