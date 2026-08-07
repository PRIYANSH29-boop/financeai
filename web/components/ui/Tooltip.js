"use client";

/*
  Tooltip — one definition surface for the whole product.

  WP3 will hang the glossary off this, so it has to work everywhere the glossary does:

    * hover AND focus AND tap. The v1 tooltip opened on mouseenter only, which means a
      phone could never read a single definition. Here the trigger is a real <button>:
      click toggles, focus opens, Escape closes.
    * described-by, not label. The trigger keeps its own accessible name (the term), and
      the bubble is announced as its description — an aria-label would have replaced the
      term with its definition in the accessibility tree.
    * `align` pins the bubble to an edge near the viewport border so it cannot render
      off-screen on a narrow phone.

  Two shapes:
    <Tooltip text="…">Sharpe</Tooltip>   term with a dotted underline
    <Tooltip text="…" />                 the bare "i" dot, for labels that read fine alone
*/

import { useId, useState } from "react";

export default function Tooltip({
  text,
  children,
  label = "What this means",  // accessible name when there is no visible term
  placement = "top",   // "top" | "bottom"
  align = "center",    // "center" | "start" | "end"
  className = "",
}) {
  const [open, setOpen] = useState(false);
  const id = useId();

  const bubbleClasses = [
    "ra-tooltip-bubble",
    placement === "bottom" ? "ra-tooltip-below" : "",
    align === "start" ? "ra-tooltip-start" : "",
    align === "end" ? "ra-tooltip-end" : "",
  ].filter(Boolean).join(" ");

  return (
    <span
      className={`ra-tooltip ${className}`.trim()}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="ra-tooltip-trigger"
        // The dot is decorative, so a bare tooltip would otherwise be a nameless button.
        aria-label={children ? undefined : label}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
      >
        {children && <span className="ra-tooltip-term">{children}</span>}
        <span className="ra-tooltip-dot" aria-hidden="true">i</span>
      </button>
      {open && (
        <span id={id} role="tooltip" className={bubbleClasses}>
          {text}
        </span>
      )}
    </span>
  );
}
