"use client";

/*
  Top-level shell — Phase 23. Holds the tab state and the shared header/footer, and swaps in
  one of the three views: the pie builder, Explore, or Basket. Kept as client-side tabs (not
  routes) so the static export stays a single first-paint with the bundle already embedded.
*/

import { useState } from "react";
import PieApp from "./PieApp";
import ExploreView from "./ExploreView";
import BasketView from "./BasketView";
import { HowItWorks } from "./Panels";

const TABS = [
  { id: "pie", label: "Build a pie" },
  { id: "explore", label: "Explore the market" },
  { id: "basket", label: "Build your basket" },
];

export default function App({ index, initialPie, stocks, explore }) {
  const [tab, setTab] = useState("pie");
  const [showHow, setShowHow] = useState(false);

  return (
    <>
      <header className="site-header">
        <div className="inner">
          <span className="brand">Rank<span>Alpha</span></span>
          <span className="sim-badge">Educational simulation</span>
          <nav className="tabs" role="tablist" aria-label="Sections">
            {TABS.map((t) => (
              <button key={t.id} role="tab" aria-selected={tab === t.id}
                      className={`tab${tab === t.id ? " active" : ""}`}
                      onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </nav>
          <button className="link-btn" style={{ marginLeft: "auto" }}
                  onClick={() => setShowHow(true)}>
            How it works →
          </button>
        </div>
      </header>

      {tab === "pie" && <PieApp index={index} initialPie={initialPie} />}
      {tab === "explore" && (
        <ExploreView explore={explore} onPickBasket={() => setTab("basket")} />
      )}
      {tab === "basket" && <BasketView stocks={stocks} index={index} explore={explore} />}

      <footer className="site-footer">
        <div className="inner">
          <span>Educational simulation. No real money. Not investment advice.</span>
          <a href="https://github.com/PRIYANSH29-boop/financeai" target="_blank"
             rel="noreferrer">GitHub repo</a>
          <span style={{ marginLeft: "auto" }}>Built by Priyansh Patel</span>
        </div>
      </footer>

      {showHow && <HowItWorks steps={index.how_it_works} onClose={() => setShowHow(false)} />}
    </>
  );
}
