const API_BASE = 'http://127.0.0.1:8000';

async function fetchJson(url) {
  const response = await fetch(`${API_BASE}${url}`);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Request failed');
  }
  return response.json();
}

export function getDashboard() {
  return fetchJson('/api/dashboard');
}

export function getModels() {
  return fetchJson('/api/models');
}

export function searchEntities(query) {
  const safeQuery = encodeURIComponent(query.trim());
  return fetchJson(`/api/entities/search?q=${safeQuery}`);
}

export function getEntity(entityId) {
  return fetchJson(`/api/entities/${encodeURIComponent(entityId)}`);
}

export function getCandidates(entityId) {
  return fetchJson(`/api/entities/${encodeURIComponent(entityId)}/candidates`);
}

export function getMatches(entityId) {
  return fetchJson(`/api/entities/${encodeURIComponent(entityId)}/matches`);
}

export function getSubmission() {
  return fetchJson('/api/submission');
}
