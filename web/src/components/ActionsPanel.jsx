import React, { useState } from "react";
import { BUILD_COST, RESOURCE_META } from "../format.js";
import BankTradePanel from "./BankTradePanel.jsx";

function Cost({ cost }) {
  if (!cost) return null;
  return (
    <span className="action-cost">
      {cost.map((r, i) => (
        <span key={i}>{RESOURCE_META[r].emoji}</span>
      ))}
    </span>
  );
}

function ActionRow({ label, cost, active, disabled, onClick }) {
  return (
    <button className={`action-row${active ? " active" : ""}`} disabled={disabled} onClick={onClick}>
      <span className="action-label">{label}</span>
      <Cost cost={cost} />
    </button>
  );
}

// Standing action menu: build actions, buy a dev card, bank trade, end turn.
// Each row is greyed out unless that action is currently legal. Build actions
// arm a board placement mode (App handles the click). Playing dev cards is done
// from the player panel (click a card in your hand).
export default function ActionsPanel({ byType, onAction, actionMode, onToggleMode, isHumanTurn }) {
  const [tradeOpen, setTradeOpen] = useState(false);
  const can = (t) => isHumanTurn && !!byType[t];

  const arm = (mode) => {
    setTradeOpen(false);
    onToggleMode(mode);
  };
  const toggleTrade = () => {
    if (actionMode) onToggleMode(null);
    setTradeOpen((o) => !o);
  };

  const hint =
    actionMode === "BUILD_ROAD"
      ? "Click a highlighted edge to build a road."
      : actionMode === "BUILD_SETTLEMENT"
      ? "Click a highlighted spot to build a settlement."
      : actionMode === "BUILD_CITY"
      ? "Click one of your settlements to upgrade it."
      : actionMode === "KNIGHT"
      ? "Click a hex to move the robber & steal."
      : actionMode === "ROAD_BUILDING"
      ? "Click up to 2 highlighted edges (free roads)."
      : null;

  return (
    <div className="panel actions-panel">
      <h2>Actions</h2>

      <ActionRow label="Road" cost={BUILD_COST.ROAD} active={actionMode === "BUILD_ROAD"} disabled={!can("BUILD_ROAD")} onClick={() => arm("BUILD_ROAD")} />
      <ActionRow label="Settlement" cost={BUILD_COST.SETTLEMENT} active={actionMode === "BUILD_SETTLEMENT"} disabled={!can("BUILD_SETTLEMENT")} onClick={() => arm("BUILD_SETTLEMENT")} />
      <ActionRow label="City" cost={BUILD_COST.CITY} active={actionMode === "BUILD_CITY"} disabled={!can("BUILD_CITY")} onClick={() => arm("BUILD_CITY")} />
      <ActionRow label="Dev card" cost={BUILD_COST.DEV} disabled={!can("BUY_DEV_CARD")} onClick={() => onAction(byType.BUY_DEV_CARD[0])} />

      <ActionRow label="Bank trade" active={tradeOpen} disabled={!can("MARITIME_TRADE")} onClick={toggleTrade} />
      {tradeOpen && can("MARITIME_TRADE") && (
        <div className="inset">
          <BankTradePanel actions={byType.MARITIME_TRADE} onAction={onAction} />
        </div>
      )}

      {hint && <p className="actions-hint">{hint}</p>}

      <button className="btn-secondary end-turn-btn" disabled={!can("END_TURN")} onClick={() => onAction(byType.END_TURN[0])}>
        End turn
      </button>
    </div>
  );
}
