import "./globals.css";

export const metadata = {
  title: "RankAlpha — build a pie",
  description:
    "Educational simulation of a risk-targeted, self-explaining equity portfolio. " +
    "Not investment advice.",
};

export default function RootLayout({ children }) {
  // `ra-root` scopes the v3 token layer and the UI kit (Chip, Badge, Tooltip, Expander).
  // It sits on <body> so the kit is usable from any tab, not only the pie page.
  return (
    <html lang="en">
      <body className="ra-root">{children}</body>
    </html>
  );
}
