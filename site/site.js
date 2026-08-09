(() => {
  const STORAGE_KEY = "dependence-is-not-advantage-theme";
  const THEMES = Object.freeze({ light: "light", dark: "dark" });
  const themeMedia = window.matchMedia("(prefers-color-scheme: dark)");

  const isTheme = (value) => value === THEMES.light || value === THEMES.dark;

  const readStoredTheme = () => {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  };

  const preferredTheme = () => {
    const storedTheme = readStoredTheme();
    return isTheme(storedTheme)
      ? storedTheme
      : themeMedia.matches
        ? THEMES.dark
        : THEMES.light;
  };

  let currentTheme = preferredTheme();

  const nextTheme = (theme) =>
    theme === THEMES.dark ? THEMES.light : THEMES.dark;

  const themeName = (theme) => (theme === THEMES.dark ? "Dark" : "Light");

  const themeControls = () => ({
    toggle: document.getElementById("theme-toggle"),
    label: document.querySelector("[data-theme-label]"),
    status: document.getElementById("theme-status"),
  });

  const updateThemeControls = (theme, announce) => {
    const { toggle, label, status } = themeControls();
    const next = nextTheme(theme);

    toggle?.setAttribute("aria-label", `Switch to ${next} mode`);
    toggle?.setAttribute("aria-pressed", String(theme === THEMES.dark));
    if (label) {
      label.textContent = themeName(next);
    }
    if (announce && status) {
      status.textContent = `${themeName(theme)} mode enabled.`;
    }
  };

  const applyTheme = (theme, announce = false) => {
    document.documentElement.dataset.theme = theme;
    updateThemeControls(theme, announce);
  };

  const rememberTheme = (theme) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Private browsing may disable storage; the current choice still applies.
    }
  };

  const initializeThemeToggle = () => {
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) {
      return;
    }

    toggle.addEventListener("click", () => {
      currentTheme = nextTheme(currentTheme);
      const theme = currentTheme;
      applyTheme(theme, true);
      rememberTheme(theme);
    });
  };

  const copyWithFallback = (text) => {
    const fallback = document.createElement("textarea");
    fallback.value = text;
    fallback.setAttribute("readonly", "");
    fallback.className = "copy-fallback";
    document.body.appendChild(fallback);
    fallback.select();

    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      fallback.remove();
    }
  };

  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      return copyWithFallback(text);
    }
  };

  const initializeCitationCopy = () => {
    const copyButton = document.getElementById("copy-bibtex");
    const bibtex = document.getElementById("bibtex-content");
    const copyStatus = document.getElementById("copy-status");

    if (!copyButton || !bibtex) {
      return;
    }

    copyButton.addEventListener("click", async () => {
      const copied = await copyText(`${bibtex.textContent.trim()}\n`);
      if (copyStatus) {
        copyStatus.textContent = copied
          ? "BibTeX copied."
          : "Copy was unavailable; select the BibTeX entry above.";
      }
    });
  };

  applyTheme(currentTheme);

  document.addEventListener(
    "DOMContentLoaded",
    () => {
      applyTheme(currentTheme);
      initializeThemeToggle();
      initializeCitationCopy();
    },
    { once: true },
  );
})();
