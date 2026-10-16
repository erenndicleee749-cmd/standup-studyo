/**
 * AI Stand-Up Stüdyosu — Frontend App JS
 * Toast bildirimleri ve genel yardımcı fonksiyonlar
 */

// ── Toast Sistemi ──────────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icons = { success: '✅', error: '❌', warn: '⚠️', info: 'ℹ️' };
  toast.textContent = `${icons[type] || ''} ${message}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideIn 0.3s ease reverse';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── LocalStorage Proje Yöneticisi ──────────────────────────────────────────
const Project = {
  get() {
    try { return JSON.parse(localStorage.getItem('su_project') || '{}'); }
    catch { return {}; }
  },
  set(data) {
    localStorage.setItem('su_project', JSON.stringify(data));
  },
  update(patch) {
    const p = this.get();
    Object.assign(p, patch);
    this.set(p);
    return p;
  },
  clear() {
    localStorage.removeItem('su_project');
    localStorage.removeItem('su_characters');
  }
};

// ── Klavye Kısayolları ─────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  // Ctrl+Enter → "Devam" butonu
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    const nextBtn = document.querySelector('[onclick*="goto"], #btn-next-voice, .btn-primary[onclick*="go"]');
    if (nextBtn && !nextBtn.disabled) nextBtn.click();
  }
  // Escape → modal kapat
  if (e.key === 'Escape') {
    const modal = document.getElementById('char-modal');
    if (modal && modal.style.display !== 'none') modal.style.display = 'none';
  }
});

// ── Navbar: aktif adım göstergesi ─────────────────────────────────────────
(function markNavStep() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-steps a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
})();

// ── Yardımcı: metin uzunluğundan süre tahmini ─────────────────────────────
function estimateDuration(lines) {
  return lines.reduce((sum, l) => {
    if (!l.text) return sum + 2.5;
    const words = l.text.trim().split(/\s+/).length;
    return sum + Math.max(2, words * 0.4);
  }, 0);
}

// ── Yardımcı: sayı biçimlendirme ─────────────────────────────────────────
function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}dk ${s}sn` : `${s}sn`;
}
