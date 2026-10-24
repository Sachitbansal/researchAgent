/* Inline SVG icons, hand-drawn on a 20px grid so the page needs no icon library.
   In:  a name and an optional size. Out: an aria-hidden <svg> using currentColor. */

const PATHS: Record<string, string> = {
  plus: "M10 4v12M4 10h12",
  trash: "M5 6h10M8 6V4.5h4V6M6.5 6l.6 9.5h5.8l.6-9.5",
  books: "M4 4.5h3v11H4zM8.5 4.5h3v11h-3zM13 5.2l2.6-.7 2.4 10.6-2.6.7z",
  panel: "M3.5 4.5h13v11h-13zM12 4.5v11",
  menu: "M4 6h12M4 10h12M4 14h12",
  sun: "M10 6.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zM10 2.5v1.6M10 15.9v1.6M2.5 10h1.6M15.9 10h1.6M4.7 4.7l1.1 1.1M14.2 14.2l1.1 1.1M4.7 15.3l1.1-1.1M14.2 5.8l1.1-1.1",
  moon: "M15.5 12.5A6 6 0 0 1 7.5 4.5a6 6 0 1 0 8 8z",
  auto: "M10 3.5a6.5 6.5 0 1 0 0 13zM10 3.5a6.5 6.5 0 1 1 0 13",
  send: "M10 15.5v-11M5.5 9 10 4.5 14.5 9",
  close: "M5 5l10 10M15 5 5 15",
  chat: "M4 5h12v8H9l-3.5 2.5V13H4z",
  external: "M8 5H5v10h10v-3M11 4h5v5M16 4l-7 7",
};

export function Icon({ name, size = 18 }: { name: keyof typeof PATHS | string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor"
         strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name] ?? ""} />
    </svg>
  );
}
