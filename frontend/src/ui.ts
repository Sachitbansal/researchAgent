/* Class strings shared by more than one component, so a pattern is written once.
   In:  nothing. Out: Tailwind class lists (the tokens are in styles.css @theme). Anything
        more involved than a class list lives in styles.css @layer components instead. */

// The Corpus and Memories pages: a scrolling page whose children sit in one centred column.
export const PAGE = "flex-1 overflow-y-auto px-6 pt-8 pb-12 *:mx-auto *:max-w-page max-md:px-4 max-md:pt-6";
export const PAGE_TITLE = "mb-2 text-22 font-semibold";
export const PAGE_LEDE = "mb-8 leading-1.6 text-soft";
// The small count after a heading: "Findings 12".
export const COUNT = "text-13 font-normal text-faint";
// A quiet line of status text with the paragraph spacing it had before preflight.
export const QUIET = "my-[1em] text-soft";
export const QUIET_SMALL = "my-[1em] text-13 text-soft";
