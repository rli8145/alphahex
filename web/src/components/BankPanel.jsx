import React from "react";
import { RESOURCE_META, RESOURCE_ORDER } from "../format.js";

// Standard Catan supply is 19 of each resource. The engine doesn't enforce a
// finite resource bank, so we show supply as 19 minus what's in players' hands
// (clamped at 0). The development-card deck count is exact from the state.
const BANK_TOTAL = 19;

export default function BankPanel({ state }) {
  const held = Object.fromEntries(RESOURCE_ORDER.map((r) => [r, 0]));
  for (const p of state.players) {
    for (const r of RESOURCE_ORDER) held[r] += p.resources[r] || 0;
  }
  const devLeft = state.dev_card_deck?.length ?? 0;

  return (
    <div className="panel bank-panel">
      <h2>Bank</h2>
      <div className="bank-res">
        {RESOURCE_ORDER.map((r) => (
          <div className="bank-cell" key={r} title={RESOURCE_META[r].label}>
            <span className="bank-emoji">{RESOURCE_META[r].emoji}</span>
            <span className="bank-count">{Math.max(0, BANK_TOTAL - held[r])}</span>
          </div>
        ))}
      </div>
      <div className="bank-dev">
        <span>🃏 Dev card deck</span>
        <strong>{devLeft}</strong>
      </div>
    </div>
  );
}
