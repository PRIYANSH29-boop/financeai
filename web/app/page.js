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
import PieApp from "../components/PieApp";

async function readBundle(rel) {
  const file = path.join(process.cwd(), "public", "bundle", rel);
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch (e) {
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

  return <PieApp index={index} initialPie={initialPie} />;
}
