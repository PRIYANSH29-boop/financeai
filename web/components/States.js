"use client";

/*
  Loading, empty and error states — #32 WP4.

  v1 had almost none of these. Changing the risk slider swapped one pie for another with no
  sign that anything was in flight; a search with no matches produced an empty table with no
  explanation; a failed bundle fetch printed one red sentence and left the stale pie on
  screen underneath it, which is the worst of the three — a number that is no longer the
  one you asked for, presented exactly like one that is.

  The rule these follow: an empty or failed state must say WHAT happened, WHY, and WHAT TO
  DO. "No results" on its own is a dead end, and a spinner that hides real numbers already
  on screen is a downgrade.
*/

/** A shimmer block. Sized by the caller so a skeleton matches the shape it stands in for. */
export function Skeleton({ w = "100%", h = 14, radius = "var(--radius-sm)", className = "" }) {
  return (
    <span
      className={`skeleton ${className}`.trim()}
      style={{ width: w, height: h, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}

/**
 * The pie's loading shape: a ring and a few legend rows.
 *
 * Deliberately the same geometry as the real donut, so the layout does not jump when the
 * data lands — a skeleton that resizes on arrival is just a slower flicker.
 */
export function DonutSkeleton() {
  return (
    <div className="donut-skeleton" role="status" aria-live="polite">
      <span className="sr-only">Loading the pie for that risk level…</span>
      <span className="skeleton skeleton-ring" aria-hidden="true" />
      <ul className="pie-legend" aria-hidden="true">
        {Array.from({ length: 8 }, (_, i) => (
          <li key={i} style={{ pointerEvents: "none" }}>
            <Skeleton w="10px" h={10} radius="2px" />
            <Skeleton w="3rem" h={11} />
            <Skeleton w="2.2rem" h={11} className="skeleton-push" />
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Nothing to show, and it is not an error.
 *
 * `action` is the way out. An empty state without one tells the reader they are stuck.
 */
export function EmptyState({ title, hint, action }) {
  return (
    <div className="state-block" role="status">
      <p className="state-title">{title}</p>
      {hint && <p className="state-hint">{hint}</p>}
      {action}
    </div>
  );
}

/**
 * Something failed.
 *
 * Red is correct here — this is the warning half of the reserved semantic, not decoration.
 * `detail` carries the real reason rather than a generic apology, because the reader may be
 * the one person who can act on it (offline, blocked, stale deploy).
 */
export function ErrorState({ title, detail, onRetry }) {
  return (
    <div className="state-block state-error" role="alert">
      <p className="state-title loss">{title}</p>
      {detail && <p className="state-hint">{detail}</p>}
      {onRetry && (
        <button type="button" className="chip" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
