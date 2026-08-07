"use client";

/*
  Count a number up to its real value once, on first paint.

  This is presentation, not arithmetic. Two rules keep it that way:

    * it always LANDS on the value it was given — the final frame is set from the prop, not
      from the last interpolation step, so a rounding error can never leave a figure on
      screen that differs from the bundle's;
    * it runs once, on mount. Re-animating every time the slider moves would turn a
      readout into a slot machine, and the instruction is "counting up on load".

  prefers-reduced-motion skips straight to the value. Nothing about the layout or the final
  number changes either way.
*/

import { useEffect, useRef, useState } from "react";

export function useCountUp(value, { duration = 900, delay = 0 } = {}) {
  const finite = Number.isFinite(value);
  // Seeded with the REAL value, never 0. This hook runs inside a static export whose whole
  // premise is that first paint carries real numbers: seeding at 0 put "£0" in the
  // prerendered HTML, so a reader with slow or disabled JavaScript saw a pie worth
  // nothing. The animation is a post-hydration flourish and starts below.
  const [shown, setShown] = useState(value);
  const doneRef = useRef(false);
  const rafRef = useRef(0);

  useEffect(() => {
    // Already counted, or there is nothing countable — track the prop directly from here.
    if (doneRef.current || !finite) { setShown(value); return undefined; }

    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { doneRef.current = true; setShown(value); return undefined; }

    let t0 = 0;
    const tick = (now) => {
      if (!t0) t0 = now;
      const p = Math.min(1, (now - t0 - delay) / duration);
      if (p < 0) { setShown(0); rafRef.current = requestAnimationFrame(tick); return; }
      // ease-out quint: fast off the mark, settles rather than stops
      setShown(value * (1 - (1 - p) ** 5));
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else { doneRef.current = true; setShown(value); }   // land on the real figure
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, finite, duration, delay]);

  return shown;
}
