/*
  Expander — the disclosure primitive.

  Built on <details>/<summary> rather than a useState toggle, so it opens without
  JavaScript, is findable by the browser's in-page search, and is announced correctly by
  screen readers for free. WP3's "How honest is this?" and every trust-chip sit on this.

  `hint` is the optional right-aligned summary line — "3 things to know", "measured, not
  promised" — that lets a collapsed expander still say what it is hiding. A disclosure
  the reader cannot see the shape of is a disclosure that does not count.
*/

export default function Expander({
  summary,
  hint,
  defaultOpen = false,
  children,
  className = "",
  ...rest
}) {
  return (
    <details className={`ra-expander ${className}`.trim()} open={defaultOpen} {...rest}>
      <summary>
        <span>{summary}</span>
        {hint && <span className="ra-expander-hint">{hint}</span>}
        <svg
          className="ra-expander-caret" width="12" height="12" viewBox="0 0 12 12"
          aria-hidden="true" focusable="false" style={hint ? undefined : { marginLeft: "auto" }}
        >
          <path d="M2 4.5 L6 8.5 L10 4.5" fill="none" stroke="currentColor"
                strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </summary>
      <div className="ra-expander-body">{children}</div>
    </details>
  );
}
