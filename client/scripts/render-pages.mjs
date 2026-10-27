import { createServer } from 'vite';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';

const routes = ['/', '/demo', '/research', '/contact'];

const server = await createServer({
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
});

try {
  const { default: App } = await server.ssrLoadModule('/src/App.jsx');
  for (const route of routes) {
    const html = renderToString(
      React.createElement(
        MemoryRouter,
        { initialEntries: [route] },
        React.createElement(App),
      ),
    );
    console.log(`${route}: rendered ${html.length} chars`);
  }
} finally {
  await server.close();
}
