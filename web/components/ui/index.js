/*
  The RankAlpha UI kit — Frontend v3, WP1.

  Import the stylesheets ONCE, here, so a consumer cannot get a component without its
  styles (or import the CSS twice from two packages). Next dedupes the CSS module graph,
  and the static export inlines it into the single build stylesheet.

  Usage:
      import { Chip, Badge, Tooltip, Expander, Num, Wordmark } from "../components/ui";
      <div className="ra-root ra-page"> … </div>

  Nothing here reads, computes or formats a number — the bundle remains the single
  source of truth, and this layer only decides how a figure looks once it arrives.
*/

import "../../styles/tokens.css";
import "../../styles/ui.css";

export { default as Wordmark } from "./Wordmark";
export { default as Chip } from "./Chip";
export { default as Badge } from "./Badge";
export { default as Tooltip } from "./Tooltip";
export { default as Expander } from "./Expander";
export { default as Num } from "./Num";
export * from "./chartTheme";
