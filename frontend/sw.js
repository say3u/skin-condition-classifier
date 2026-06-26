/*
 * Service worker for the Skin Classifier PWA.
 *
 * Caches the static frontend so the app shell loads instantly on repeat visits
 * and works offline. The /api/* calls obviously still need a network connection -
 * those are explicitly NOT cached.
 *
 * Bump CACHE_NAME when shipping breaking changes to static assets.
 */

const CACHE_NAME = "skin-classifier-v1";
const STATIC_ASSETS = [
  "/",
  "/style.css",
  "/app.js",
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png",
  "/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Don't cache or intercept API calls - they need fresh network responses
  if (url.pathname.startsWith("/api/")) return;

  // Only handle same-origin GETs
  if (event.request.method !== "GET" || url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response && response.status === 200 && response.type === "basic") {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => {
        // Offline fallback: serve the cached shell for navigation requests
        if (event.request.mode === "navigate") return caches.match("/");
      });
    })
  );
});
