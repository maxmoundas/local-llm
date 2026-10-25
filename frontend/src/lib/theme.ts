// Theme preference: 'system' follows the OS; 'light' / 'dark' force one.
// The resolved theme is written to <html data-theme>, which global.css keys off.
// index.html applies the saved preference before first paint to avoid a flash.

export type ThemePref = 'system' | 'light' | 'dark'

const KEY = 'theme'

export function getThemePref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch { /* storage unavailable */ }
  return 'system'
}

export function setThemePref(pref: ThemePref): void {
  try { localStorage.setItem(KEY, pref) } catch { /* storage unavailable */ }
  applyTheme(pref)
}

export function applyTheme(pref: ThemePref = getThemePref()): void {
  const root = document.documentElement
  if (pref === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', pref)
}
