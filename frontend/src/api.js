const UNREACHABLE = 'Backend not reachable. Is FastAPI running on port 8000?';

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error(UNREACHABLE);
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (data === null && response.status >= 500) throw new Error(UNREACHABLE);
    throw new Error(typeof data?.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
  }
  return data;
}

const json = (method, body) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body)
});

export const api = {
  search: (query) => request(`/api/assets/search?query=${encodeURIComponent(query)}`),
  details: (assetId) => request(`/api/assets/details?id=${encodeURIComponent(assetId)}`),
  inventory: () => request('/api/inventory'),
  download: (assetId) => request('/api/inventory', json('POST', { assetId })),
  remove: (itemId) => request(`/api/inventory/${itemId}`, { method: 'DELETE' }),
  setDimensions: (itemId, dimensions) => request(`/api/inventory/${itemId}/dimensions`, json('PUT', dimensions)),
  designModels: () => request('/api/design/models'),
  designGraph: () => request('/api/design/graph'),
  startDesignRun: (payload) => request('/api/design/runs', json('POST', payload)),
  cancelDesignRun: (runId) => request(`/api/design/runs/${runId}`, { method: 'DELETE' }),
  answerApproval: (runId, decision) => request(`/api/design/runs/${runId}/approval`, json('POST', decision)),
  designRunEventsUrl: (runId) => `/api/design/runs/${runId}/events`
};
