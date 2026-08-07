/*
  Badge — an INERT label. "Educational simulation", a sector tag, a measured beta.

  The `num` variant drops the uppercase/tracking treatment and switches to tabular
  monospace, because a number is not a label: it belongs to the type scale's numeric
  track wherever it appears.
*/

export default function Badge({
  children,
  variant = "default",   // "default" | "accent" | "loss"
  num = false,
  className = "",
  ...rest
}) {
  const classes = [
    "ra-badge",
    variant === "accent" ? "ra-badge-accent" : "",
    variant === "loss" ? "ra-badge-loss" : "",
    num ? "ra-badge-num" : "",
    className,
  ].filter(Boolean).join(" ");

  return <span className={classes} {...rest}>{children}</span>;
}
