/*
  The RankAlpha wordmark — Frontend v3, WP1.

  Inline SVG rather than an asset: it inherits `currentColor`, scales without a second
  file, and adds nothing to the static export's network graph. The mark is a three-rung
  ladder whose top rung is the accent — "rank", said without a caption.

  No hooks, so it renders in a server component as happily as a client one.
*/

export default function Wordmark({ size = 20, showText = true, className = "", ...rest }) {
  const box = 24;
  return (
    <span className={`ra-wordmark ${className}`.trim()} {...rest}>
      <svg
        className="ra-wordmark-mark"
        width={size} height={size} viewBox={`0 0 ${box} ${box}`}
        focusable="false"
        // Beside the wordtype the mark is decoration and must not be announced twice;
        // on its own it IS the name.
        {...(showText
          ? { "aria-hidden": "true" }
          : { role: "img", "aria-label": "RankAlpha" })}
      >
        <rect className="ra-wordmark-rung" x="2" y="15" width="5" height="7" rx="1.5" />
        <rect className="ra-wordmark-rung" x="9.5" y="9" width="5" height="13" rx="1.5" />
        <rect className="ra-wordmark-rung-top" x="17" y="2" width="5" height="20" rx="1.5" />
      </svg>
      {showText && (
        <span>
          Rank<span className="ra-wordmark-alpha">Alpha</span>
        </span>
      )}
    </span>
  );
}
