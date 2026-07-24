"use client";

/*
  Hand-rolled SVG charts.

  No charting library: the bundle is small, the shapes are simple, and every extra
  dependency is another thing that can pull in a runtime fetch or bloat a static export.
  These take the bundle's {d, v} series verbatim — they never compute a statistic, so
  there is no path by which a chart can show a number the exporter did not produce.
*/

const PAD = { t: 10, r: 8, b: 22, l: 44 };

function scales(series, width, height, { minAtZero = false } = {}) {
  const xs = series.map((_, i) => i);
  const ys = series.map((p) => p.v);
  let lo = Math.min(...ys);
  let hi = Math.max(...ys);
  if (minAtZero) hi = Math.max(hi, 0);
  if (lo === hi) { lo -= 0.01; hi += 0.01; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;
  const w = width - PAD.l - PAD.r;
  const h = height - PAD.t - PAD.b;
  const x = (i) => PAD.l + (xs.length <= 1 ? w / 2 : (i / (xs.length - 1)) * w);
  const y = (v) => PAD.t + h - ((v - lo) / (hi - lo)) * h;
  return { x, y, lo, hi, w, h };
}

const path = (series, x, y) =>
  series.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(p.v).toFixed(2)}`).join(" ");

function yearTicks(series) {
  const seen = new Set();
  const out = [];
  series.forEach((p, i) => {
    const yr = p.d.slice(0, 4);
    if (!seen.has(yr)) { seen.add(yr); out.push({ i, label: yr }); }
  });
  return out.length > 8 ? out.filter((_, k) => k % 2 === 0) : out;
}

/** Growth of capital: the pie against its benchmark. */
export function GrowthChart({ series, benchmark, capital, currency, height = 230 }) {
  const width = 640;
  if (!series?.length) return null;
  const all = benchmark?.length ? [...series, ...benchmark] : series;
  const { x, y, lo, hi } = scales(all, width, height);
  const fmt = (v) =>
    new Intl.NumberFormat("en-GB", {
      style: "currency", currency: currency || "GBP", maximumFractionDigits: 0,
      notation: "compact",
    }).format(v * capital);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
         role="img" aria-label="Growth of your capital versus the benchmark">
      {[lo, (lo + hi) / 2, hi].map((v, i) => (
        <g key={i}>
          <line x1={PAD.l} x2={width - PAD.r} y1={y(v)} y2={y(v)}
                stroke="var(--border)" strokeWidth="1" />
          <text x={PAD.l - 7} y={y(v) + 3.5} textAnchor="end"
                fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
            {fmt(v)}
          </text>
        </g>
      ))}
      {yearTicks(series).map((t) => (
        <text key={t.label} x={x(t.i)} y={height - 6} textAnchor="middle"
              fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
          {t.label}
        </text>
      ))}
      {benchmark?.length > 0 && (
        <path d={path(benchmark, x, y)} fill="none" stroke="var(--text-faint)"
              strokeWidth="1.4" strokeDasharray="3 3" />
      )}
      <path d={path(series, x, y)} fill="none" stroke="var(--accent)" strokeWidth="2.2" />
    </svg>
  );
}

/** Drawdown: always ≤ 0, filled red — this is a loss, which is what red is for. */
export function DrawdownChart({ series, height = 150 }) {
  const width = 640;
  if (!series?.length) return null;
  const { x, y } = scales(series, width, height, { minAtZero: true });
  const area =
    `${path(series, x, y)} L${x(series.length - 1).toFixed(2)},${y(0).toFixed(2)} ` +
    `L${x(0).toFixed(2)},${y(0).toFixed(2)} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
         role="img" aria-label="Drawdown from the previous peak">
      <line x1={PAD.l} x2={width - PAD.r} y1={y(0)} y2={y(0)}
            stroke="var(--border-strong)" strokeWidth="1" />
      <text x={PAD.l - 7} y={y(0) + 3.5} textAnchor="end" fontSize="10"
            fill="var(--text-faint)" fontFamily="var(--mono)">0%</text>
      {(() => {
        const worst = Math.min(...series.map((p) => p.v));
        return (
          <text x={PAD.l - 7} y={y(worst) + 3.5} textAnchor="end" fontSize="10"
                fill="var(--loss)" fontFamily="var(--mono)">
            {(worst * 100).toFixed(0)}%
          </text>
        );
      })()}
      {yearTicks(series).map((t) => (
        <text key={t.label} x={x(t.i)} y={height - 6} textAnchor="middle"
              fontSize="10" fill="var(--text-faint)" fontFamily="var(--mono)">
          {t.label}
        </text>
      ))}
      <path d={area} fill="var(--loss)" fillOpacity="0.13" />
      <path d={path(series, x, y)} fill="none" stroke="var(--loss)" strokeWidth="1.7" />
    </svg>
  );
}
