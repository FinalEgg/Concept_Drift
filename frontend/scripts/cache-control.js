// Service Worker for Concept Drift Detection System
const CACHE_NAME = 'concept-drift-v1.0.0';

// 安装事件：预缓存核心资源
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll([
        '/',
        '/index.html',
        '/login.html',
        '/register.html',
        '/dashboard.html',
        '/scripts/cache-control.js',
        '/static/css/auth.css',
        '/static/css/welcome.css'
      ]);
    })
  );
});

// 激活事件：清理旧缓存
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(name => name !== CACHE_NAME)
          .map(name => caches.delete(name))
      );
    })
  );
});

// 请求拦截：优先使用缓存，同时更新缓存
self.addEventListener('fetch', event => {
  // 只缓存同源请求
  if (event.request.url.startsWith(self.location.origin)) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        // 返回缓存的响应，同时在背后更新缓存
        const fetchPromise = fetch(event.request).then(networkResponse => {
          // 检查是否为有效响应
          if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, responseToCache);
            });
          }
          return networkResponse;
        });
        
        return cachedResponse || fetchPromise;
      })
    );
  }
});