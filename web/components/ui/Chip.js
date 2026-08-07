/*
  Chip — a small INTERACTIVE control: a beta preset, a filter, a basket pick.

  Deliberately separate from Badge, which is the same silhouette but inert. A reader
  should never have to click a pill to find out whether it is clickable.

  Variants are semantic, not decorative:
    "default"  neutral surface
    "loss"     losses and warnings only — never "a third colour"
  Selection is expressed with aria-pressed so assistive tech sees the state that the
  accent tint is showing sighted readers.
*/

export default function Chip({
  children,
  selected = false,
  variant = "default",
  size = "md",
  onDismiss,
  dismissLabel,
  className = "",
  type = "button",
  ...rest
}) {
  const classes = [
    "ra-chip",
    size === "sm" ? "ra-chip-sm" : "",
    variant === "loss" ? "ra-chip-loss" : "",
    className,
  ].filter(Boolean).join(" ");

  return (
    <button type={type} className={classes} aria-pressed={selected} {...rest}>
      {children}
      {onDismiss && (
        // A nested <button> is invalid HTML, so the dismiss affordance is a span with an
        // explicit role — it stays keyboard-reachable without breaking the outer control.
        <span
          role="button"
          tabIndex={0}
          className="ra-chip-dismiss"
          aria-label={dismissLabel || "Remove"}
          onClick={(e) => { e.stopPropagation(); onDismiss(e); }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              e.stopPropagation();
              onDismiss(e);
            }
          }}
        >
          ×
        </span>
      )}
    </button>
  );
}
