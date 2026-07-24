/*
  Build-time data load.

  With `output: 'export'` this server component runs once at build, so reading the bundle
  off disk here costs the user nothing and guarantees the first paint shows real numbers
  rather than a spinner. Only the *other* betas are fetched client-side on demand.

  If the bundle is missing the build fails loudly, on purpose: a silently empty UI would be
  worse than no UI, and the fix is one command.
*/

import { readFile } from "node:fs/promises";
import path from "node:path";
import App from "../components/App";

async function readBundle(rel, { required = true } = {}) {
  const file = path.join(process.cwd(), "public", "bundle", rel);
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch (e) {
    if (!required) return null;   // explore.json is optional (wide universe may be absent)
    throw new Error(
      `Web bundle missing or unreadable: ${file}\n` +
      `Generate it first:  make web-bundle   (or python scripts/export_web_bundle.py)\n` +
      `Original error: ${e.message}`
    );
  }
}

export default async function Page() {
  const index = await readBundle("index.json");

  // Default to the middle preset ("Balanced") when present, so the landing state is a
  // deliberate choice from the bundle rather than whichever file sorts first.
  const preset = index.presets?.[1] || index.presets?.[0];
  const initialPie = await readBundle(`beta/${preset.key}.json`);

  // #23 — Explore + Basket data, embedded at build time like the pie.
  const stocks = await readBundle("stocks.json");
  const explore = await readBundle("explore.json", { required: false });

  return <App index={index} initialPie={initialPie} stocks={stocks} explore={explore} />;
}
