// Progressive enhancement for /login (and inline 401 login pages at /):
// 1. Accessible Show/Hide toggle for the API key field (aria-pressed).
// 2. Clears server/AJAX error state as soon as the user edits the field.
// 3. Submits via JSON POST /login so a hash route (e.g. /#/models) survives
//    sign-in; falls back cleanly to native form POST when JS is unavailable.

const form = document.getElementById('login-form');
const input = document.getElementById('api-key');
const toggleBtn = document.getElementById('toggle-key');
const submitBtn = document.getElementById('login-submit');
const errorEl = document.getElementById('login-error');

function showError(message) {
  if (!errorEl) return;
  errorEl.textContent = message;
  errorEl.hidden = !message;
  if (input) {
    if (message) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }
}

if (toggleBtn && input) {
  toggleBtn.addEventListener('click', () => {
    const showing = input.type === 'text';
    const nextShow = !showing;
    input.type = nextShow ? 'text' : 'password';
    toggleBtn.setAttribute('aria-pressed', nextShow ? 'true' : 'false');
    toggleBtn.setAttribute('aria-label', nextShow ? 'Hide API key' : 'Show API key');
    toggleBtn.textContent = nextShow ? 'Hide' : 'Show';
  });
}

if (input) {
  input.addEventListener('input', () => {
    input.setCustomValidity('');
    if (errorEl && !errorEl.hidden) showError('');
  });
}

if (form && input && submitBtn) {
  form.addEventListener('submit', async (e) => {
    if (!form.reportValidity()) return;
    e.preventDefault();
    if (submitBtn.disabled) return;

    const apiKey = input.value;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Signing in…';
    showError('');

    try {
      const res = await fetch('/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ api_key: apiKey }),
      });
      if (res.ok) {
        const hash = window.location.hash || '';
        if (window.location.pathname === '/login') {
          window.location.replace(`/${hash}`);
        } else {
          window.location.reload();
        }
        return;
      }
      let message = 'Invalid API key.';
      try {
        const body = await res.json();
        message = body?.error?.message || body?.detail || message;
      } catch { /* keep fallback */ }
      showError(message);
      input.focus();
      input.select();
    } catch {
      showError('Could not reach the server. Check your connection and try again.');
      input.focus();
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Sign in';
    }
  });
}
