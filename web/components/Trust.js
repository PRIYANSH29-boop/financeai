"use client";

/*
  The trust surface — #32 WP3.

  Two components, one job: make the honest parts of this product readable instead of
  merely present.

  <Term> turns a piece of jargon into a definition on hover, focus or tap, pulled from the
  single glossary. It renders the word itself as the trigger, so the sentence still reads
  normally and the definition is opt-in.

  <TrustPanel> replaces the stacked red caveat boxes. Those said the right things in the
  most alarming available style, and a wall of warnings reads as boilerplate — it gets
  skipped, which is the opposite of disclosing. The same sentences are now calm one-line
  chips inside a single "How honest is this?" expander, each expanding to plain English
  about why it matters.

  Nothing here composes or edits a disclosure. Every chip's visible line is the exporter's
  string, byte for byte, and lib/glossary.test.js asserts that against the shipped bundle.
*/

import { useState } from "react";
import { Tooltip, Expander } from "./ui";
import { define } from "../lib/glossary";
import { trustChips } from "../lib/disclosure";

/**
 * A glossary term. `k` overrides the lookup key when the visible word differs from the
 * entry — `<Term k="max drawdown">worst fall</Term>`.
 *
 * An unknown term degrades to plain text rather than an empty tooltip, so a typo costs a
 * definition and never a broken control.
 */
export function Term({ children, k }) {
  const key = k ?? (typeof children === "string" ? children : "");
  const def = define(key);
  if (!def) return <>{children}</>;
  return <Tooltip text={def}>{children}</Tooltip>;
}

/** One caveat: its exact sentence, and — if we have written one — why it matters. */
function TrustChip({ chip }) {
  const [open, setOpen] = useState(false);
  const expandable = Boolean(chip.detail);

  // A line with nothing to expand is text, not a broken control. Rendering it as a
  // disabled <button> announced it to screen readers as an unavailable widget — worse
  // than plain prose for something whose whole job is to be read.
  const inner = (
    <>
      <span className="trust-chip-dot" aria-hidden="true" />
      <span className="trust-chip-text">{chip.text}</span>
      {expandable && (
        <span className="trust-chip-caret" aria-hidden="true">{open ? "−" : "+"}</span>
      )}
    </>
  );

  return (
    <li className={`trust-chip${open ? " open" : ""}`}>
      {expandable ? (
        <button
          type="button"
          className="trust-chip-line"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {inner}
        </button>
      ) : (
        <p className="trust-chip-line trust-chip-static">{inner}</p>
      )}
      {open && chip.detail && <p className="trust-chip-detail">{chip.detail}</p>}
    </li>
  );
}

/**
 * The single disclosure home.
 *
 * `children` is the panel's own evidence — charts, tiles, the measured-beta notes — which
 * sits above the chips. The disclosure hierarchy the instruction asks for is enforced by
 * placement: at most one inline caveat lives in a panel's body, and everything else lives
 * in here.
 */
export function TrustPanel({ index, statsBundle, defaultOpen = false, children, title }) {
  const chips = trustChips(index, statsBundle ?? index);

  return (
    <Expander
      summary={title ?? "How honest is this?"}
      hint={`${chips.length} thing${chips.length === 1 ? "" : "s"} to know`}
      defaultOpen={defaultOpen}
      className="trust-expander"
    >
      {children}
      <ul className="trust-chips">
        {chips.map((c) => <TrustChip key={c.id} chip={c} />)}
      </ul>
      <p className="trust-foot">
        These are the limits of what the numbers above can tell you. Tap any line for what
        it means in practice.
      </p>
    </Expander>
  );
}

/**
 * The one-line version, for a panel that is not the disclosure home — Explore's table
 * footer, the Basket scorecard. It shows the single most situational caveat inline and
 * points at the full set rather than restacking it.
 */
export function TrustFooter({ index, statsBundle }) {
  const chips = trustChips(index, statsBundle ?? index);
  if (!chips.length) return null;
  const [lead, ...rest] = chips;

  return (
    <p className="trust-footer">
      <span className="trust-chip-dot" aria-hidden="true" />
      {lead.text}
      {rest.length > 0 && (
        <span className="faint">
          {" "}· and {rest.length} more limit{rest.length === 1 ? "" : "s"} under{" "}
          <strong>How honest is this?</strong>
        </span>
      )}
    </p>
  );
}
