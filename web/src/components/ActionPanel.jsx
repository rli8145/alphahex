import React, { useState } from "react";
import { actionLabel, BOT_ID, HUMAN_ID, PLAYER_NAMES } from "../format.js";
import DiscardPanel from "./DiscardPanel.jsx";
import Dice from "./Dice.jsx";

// Order and friendly headers for the non-spatial action groups.
const GROUPS = [
  { type: "ROLL_DICE", title: null },
  { type: "STEAL_RESOURCE", title: "Steal from" },
  { type: "BUY_DEV_CARD", title: null },
  { type: "PLAY_KNIGHT", title: "Play Knight" },
  { type: "PLAY_MONOPOLY", title: "Play Monopoly" },
  { type: "PLAY_YEAR_OF_PLENTY", title: "Play Year of Plenty" },
  { type: "PLAY_ROAD_BUILDING", title: "Play Road Building" },
  { type: "MARITIME_TRADE", title: "Maritime trade" },
  { type: "END_TURN", title: null },
];

function Group({ title, actions, onAction }) {
  const collapsible = actions.length > 6;
  const [open, setOpen] = useState(false);
  const shown = collapsible && !open ? [] : actions;

  return (
    <div className="action-group">
      {title && (
        <div className="action-group-head">
          <span>{title}</span>
          {collapsible && (
            <button className="link-btn" onClick={() => setOpen((o) => !o)}>
              {open ? "hide" : `show ${actions.length}`}
            </button>
          )}
        </div>
      )}
      <div className="action-buttons">
        {shown.map((action, i) => (
          <button
            key={i}
            className={action.action_type === "END_TURN" ? "btn-secondary" : action.action_type === "ROLL_DICE" ? "btn-primary" : "btn"}
            onClick={() => onAction(action)}
          >
            {actionLabel(action)}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ActionPanel({ state, legalActions, onAction, busy, winner }) {
  const isHumanTurn = state.current_player === HUMAN_ID && winner == null;
  const phase = state.phase;

  const byType = {};
  for (const a of legalActions) {
    (byType[a.action_type] ??= []).push(a);
  }
  const rollAction = isHumanTurn ? byType.ROLL_DICE?.[0] : null;

  return (
    <div className="panel action-panel">
      <div className="dice-tray">
        <Dice
          total={state.dice_roll}
          salt={state.turn_number}
          onClick={rollAction ? () => onAction(rollAction) : undefined}
          hint={rollAction ? "click to roll" : undefined}
          disabled={busy}
        />
      </div>

      {winner != null ? (
        <p className="muted">The game is over. Start a new game to play again.</p>
      ) : !isHumanTurn ? (
        <p className="thinking">
          <span className="thinking-dot" />
          {PLAYER_NAMES[BOT_ID]} is plotting...
        </p>
      ) : phase === "DISCARD" ? (
        <DiscardPanel state={state} onSubmit={onAction} />
      ) : (
        GROUPS.map((g) =>
          g.type !== "ROLL_DICE" && byType[g.type] ? (
            <Group key={g.type} title={g.title} actions={byType[g.type]} onAction={onAction} />
          ) : null
        )
      )}
    </div>
  );
}
