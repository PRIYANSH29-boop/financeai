import "./globals.css";

export const metadata = {
  title: "RankAlpha — build a pie",
  description:
    "Educational simulation of a risk-targeted, self-explaining equity portfolio. " +
    "Not investment advice.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
