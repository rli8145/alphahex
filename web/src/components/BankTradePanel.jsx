import React, { useState } from "react";
import { RESOURCE_META, RESOURCE_ORDER } from "../format.js";

// Bank/maritime trade. The engine enumerates one action per (give, receive)
// pair at the player's best ratio for that give resource; the player picks a
// give and a receive and we apply the matching action.
export default function BankTradePanel({ actions, onAction }) {
  const ratios = {};
  for (const a of actions) ratios[a.payload.give] = a.payload.give_count;
  const gives = RESOURCE_ORDER.filter((r) => r in ratios);

  const [give, setGive] = useState(gives[0]);
  const [receive, setReceive] = useState(null);

  const effGive = gives.includes(give) ? give : gives[0];
  const receives = RESOURCE_ORDER.filter((r) => r !== effGive);
  const effReceive = receives.includes(receive) ? receive : receives[0];
  const ratio = ratios[effGive];
  const trade = actions.find(
    (a) => a.payload.give === effGive && a.payload.receive === effReceive
  );

  return (
    <div className="trade-panel">
      <div className="action-group-head">
        <span>Bank trade</span>
      </div>

      <div className="trade-grid">
        <span className="trade-label">You give</span>
        <div className="res-chips">
          {gives.map((r) => (
            <button
              key={r}
              className={`res-chip${r === effGive ? " sel" : ""}`}
              onClick={() => setGive(r)}
              title={`${ratios[r]}:1 ${RESOURCE_META[r].label}`}
            >
              <span className="res-chip-emoji">{RESOURCE_META[r].emoji}</span>
              <span className="res-chip-ratio">{ratios[r]}:1</span>
            </button>
          ))}
        </div>

        <span className="trade-label">You get</span>
        <div className="res-chips">
          {receives.map((r) => (
            <button
              key={r}
              className={`res-chip${r === effReceive ? " sel" : ""}`}
              onClick={() => setReceive(r)}
              title={RESOURCE_META[r].label}
            >
              <span className="res-chip-emoji">{RESOURCE_META[r].emoji}</span>
            </button>
          ))}
        </div>
      </div>

      <button className="btn-primary trade-go" disabled={!trade} onClick={() => trade && onAction(trade)}>
        <span className="trade-go-side">
          {ratio} {RESOURCE_META[effGive].emoji}
        </span>
        <span className="trade-arrow">→</span>
        <span className="trade-go-side">1 {RESOURCE_META[effReceive].emoji}</span>
      </button>
    </div>
  );
}
