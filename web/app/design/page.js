/*
  /design — the v3 design system, rendering itself.

  A reference surface for reviewers and for WP2/WP3/WP4, not a product page: it is not
  linked from the app, carries no product data, and is marked noindex. It ships with the
  static export so the system can be reviewed as pixels rather than as a diff.
*/

import Gallery from "../../components/ui/Gallery";

export const metadata = {
  title: "RankAlpha — design system (v3)",
  description: "Tokens and components for the RankAlpha frontend. Internal reference.",
  robots: { index: false, follow: false },
};

export default function DesignPage() {
  return <Gallery />;
}
