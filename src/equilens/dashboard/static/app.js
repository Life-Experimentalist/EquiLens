// EquiLens Dashboard — app.js
// Plain JS, no modules. All functions are global for Alpine.js x-on/x-init use.

/* ── Fetch helpers ── */

/**
 * Fetch with a 10-second timeout and up to 2 retries on network error.
 * Throws on HTTP errors (error.status / error.statusText are set).
 */
async function apiFetch(url, options = {}) {
  const MAX_RETRIES = 2;
  const TIMEOUT_MS = 10000;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);

      if (!res.ok) {
        const err = new Error(`HTTP ${res.status} ${res.statusText}`);
        err.status = res.status;
        err.statusText = res.statusText;
        throw err;
      }

      return await res.json();
    } catch (err) {
      clearTimeout(timer);

      // Don't retry HTTP errors — only network/timeout errors
      if (err.status != null) throw err;

      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 1000));
      } else {
        throw err;
      }
    }
  }
}

async function apiGet(url) {
  return apiFetch(url);
}

async function apiPost(url, body) {
  return apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function apiDelete(url) {
  return apiFetch(url, { method: 'DELETE' });
}

/* ── SSE streaming ── */

/**
 * Stream job logs via Server-Sent Events.
 * Calls onLog(logObj) for each data event.
 * Calls onDone(finalStatus) when a "done" event arrives or the stream closes.
 * Returns the EventSource so the caller can close it manually.
 *
 * @param {string} jobId
 * @param {(log: {level: string, message: string, timestamp: string}) => void} onLog
 * @param {(status: string) => void} onDone
 * @returns {EventSource}
 */
function streamJobLogs(jobId, onLog, onDone) {
  const es = new EventSource(`/api/events/${jobId}`);

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.event === 'done') {
        es.close();
        onDone(data.status || 'completed');
        return;
      }

      onLog(data);
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  };

  es.onerror = () => {
    es.close();
    onDone('failed');
  };

  return es;
}

/* ── Display helpers ── */

/** Returns the CSS class string for a job status badge. */
function badgeClass(status) {
  return 'badge badge-' + (status || 'queued');
}

/** Returns the CSS class string for an online/offline status dot. */
function dotClass(ok) {
  return 'dot ' + (ok ? 'dot-green' : 'dot-red');
}

/** Format a byte count into a human-readable string. */
function fmtBytes(bytes) {
  if (bytes == null || isNaN(bytes)) return '—';
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  if (bytes >= 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return bytes + ' B';
}

/** Format an ISO timestamp into a readable local date+time string. */
function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return '—';
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format the duration between two ISO timestamps.
 * Returns "—" if either timestamp is missing.
 * Example output: "2m 14s", "45s", "1h 3m"
 */
function fmtDuration(startIso, endIso) {
  if (!startIso || !endIso) return '—';
  const start = new Date(startIso);
  const end = new Date(endIso);
  if (isNaN(start) || isNaN(end)) return '—';

  let secs = Math.max(0, Math.round((end - start) / 1000));
  const h = Math.floor(secs / 3600);
  secs -= h * 3600;
  const m = Math.floor(secs / 60);
  secs -= m * 60;

  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${secs}s`;
  return `${secs}s`;
}

/** Scroll an element to its bottom (useful for log panels). */
function scrollBottom(el) {
  if (el) el.scrollTop = el.scrollHeight;
}

/* ── Alpine.js store & components ── */

document.addEventListener('alpine:init', () => {
  // Global error store
  Alpine.store('globalStore', {
    errorMessage: null,
    errorVisible: false,

    showError(msg) {
      this.errorMessage = msg;
      this.errorVisible = true;
    },

    clearError() {
      this.errorMessage = null;
      this.errorVisible = false;
    },
  });

  // Reusable error-banner component (referenced via x-data="errorBanner")
  Alpine.data('errorBanner', () => ({
    get errorMessage() {
      return Alpine.store('globalStore').errorMessage;
    },
    get errorVisible() {
      return Alpine.store('globalStore').errorVisible;
    },
    clearError() {
      Alpine.store('globalStore').clearError();
    },
  }));
});
