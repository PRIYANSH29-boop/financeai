/*
  Num — the numeric track of the type scale, as a component.

  "Monospace for all numbers" is a rule that only holds if it is easy to obey, so every
  figure in v3 goes through here. Tone is semantic and mirrors the token rules exactly:

    "default"  plain ink
    "pos"      the accent — a gain
    "loss"     red — a LOSS, and nothing else. Not "a big change", not "a warning icon".
    "neutral"  bright neutral ink — a delta that is INFORMATION: a weight drift in
               percentage points is not a loss, and colouring it red would say it was.

  This component never formats. Formatting lives in lib/format.js and stays there, so
  there remains exactly one place where precision and sign conventions are decided.
*/

export default function Num({ children, tone = "default", size, className = "", ...rest }) {
  const classes = [
    "ra-num",
    tone === "pos" ? "ra-pos" : "",
    tone === "loss" ? "ra-loss" : "",
    tone === "neutral" ? "ra-neutral" : "",
    tone === "muted" ? "ra-muted" : "",
    className,
  ].filter(Boolean).join(" ");

  const style = size ? { fontSize: `var(--ra-text-${size})`, ...rest.style } : rest.style;
  return <span className={classes} {...rest} style={style}>{children}</span>;
}
