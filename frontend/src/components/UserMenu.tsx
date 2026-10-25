/* The sidebar's foot: the single user ("You", no accounts) and a small menu above it.
   In:  the theme choice and its setter, whether the Memories page is open. Out: a menu
        with Memories (#/memories) and the theme as a radio group. Keyboard: Enter, Space
        or the arrows open it, arrows / Home / End move, Esc closes and returns focus to
        the button, Tab or a click outside closes it. */

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Icon } from "../icons";

export type Theme = "system" | "light" | "dark";

const THEMES: { value: Theme; label: string; icon: string }[] = [
  { value: "system", label: "Match the system", icon: "auto" },
  { value: "light", label: "Light", icon: "sun" },
  { value: "dark", label: "Dark", icon: "moon" },
];

interface Props {
  theme: Theme;
  onTheme: (t: Theme) => void;
  memoriesActive: boolean;
}

export function UserMenu({ theme, onTheme, memoriesActive }: Props) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  // Which item to focus once the menu has rendered: first, last, or none (mouse open).
  const focusOnOpen = useRef<"first" | "last" | null>(null);

  const items = () => Array.from(menu.current?.querySelectorAll<HTMLElement>("[role^='menuitem']") ?? []);

  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) button.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return;
    const target = focusOnOpen.current;
    focusOnOpen.current = null;
    const all = items();
    if (target === "first") all[0]?.focus();
    else if (target === "last") all[all.length - 1]?.focus();
    else menu.current?.focus();
    const onDown = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const onButtonKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    // Enter and Space arrive as a click (detail 0), handled there.
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      // The menu opens upwards, so ArrowUp lands on the item nearest the button.
      focusOnOpen.current = e.key === "ArrowUp" ? "last" : "first";
      setOpen(true);
    }
  };

  const onMenuKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const all = items();
    const at = all.indexOf(document.activeElement as HTMLElement);
    const move = (i: number) => {
      e.preventDefault();
      all[(i + all.length) % all.length]?.focus();
    };
    if (e.key === "Escape") {
      e.preventDefault();
      // Not also the app's Esc, which would close the sidebar drawer on a narrow screen.
      e.stopPropagation();
      close(true);
    } else if (e.key === "ArrowDown") move(at + 1);
    else if (e.key === "ArrowUp") move(at < 0 ? all.length - 1 : at - 1);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(all.length - 1);
    else if (e.key === "Tab") setOpen(false);
  };

  return (
    <div className="user-menu" ref={wrap}>
      {open && (
        <div className="menu" role="menu" id="user-menu" aria-label="You" ref={menu} tabIndex={-1} onKeyDown={onMenuKey}>
          <a
            role="menuitem"
            tabIndex={-1}
            className={`menu-item${memoriesActive ? " is-on" : ""}`}
            href="#/memories"
            aria-current={memoriesActive ? "page" : undefined}
            onClick={() => close(false)}
          >
            <Icon name="memory" size={16} />
            <span className="menu-label">Memories</span>
          </a>
          <div className="menu-sep" role="separator" />
          <div className="menu-head" id="user-menu-theme">Theme</div>
          <div role="group" aria-labelledby="user-menu-theme">
            {THEMES.map((t) => (
              <button
                key={t.value}
                role="menuitemradio"
                tabIndex={-1}
                aria-checked={theme === t.value}
                className="menu-item"
                onClick={() => {
                  onTheme(t.value);
                  close(true);
                }}
              >
                <Icon name={t.icon} size={16} />
                <span className="menu-label">{t.label}</span>
                {theme === t.value && <Icon name="check" size={16} />}
              </button>
            ))}
          </div>
        </div>
      )}
      <button
        ref={button}
        className={`user-btn${open ? " is-open" : ""}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? "user-menu" : undefined}
        onClick={(e) => {
          if (!open && e.detail === 0) focusOnOpen.current = "first"; // from the keyboard
          setOpen(!open);
        }}
        onKeyDown={onButtonKey}
      >
        <span className="avatar" aria-hidden="true">Y</span>
        <span className="user-name">You</span>
        <Icon name="chevron" size={16} />
      </button>
    </div>
  );
}
