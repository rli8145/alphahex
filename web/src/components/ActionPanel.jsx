import React from "react";
import { actionLabel, BOT_ID, HUMAN_ID, PLAYER_NAMES } from "../format.js";
import DiscardPanel from "./DiscardPanel.jsx";
import Dice from "./Dice.jsx";

// The dice panel: the always-visible dice (clickable to roll on your turn) plus
// the prompt for whatever forced sub-phase is active. The standing build / dev /
// trade menu lives in ActionsPanel.
export default function ActionPanel({ state, legalActions, onAction, busy, winner }) {
  const isHumanTurn = state.current_player === HUMAN_ID && winner == null;
  const phase = state.phase;

  const byType = {};
  for (const a of legalActions) {
    (byType[a.action_type] ??= []).push(a);
  }
  const rollAction = isHumanTurn ? byType.ROLL_DICE?.[0] : null;
  const steals = byType.STEAL_RESOURCE ?? [];

  let prompt = null;
  if (winner != null) {
    prompt = <p className="muted">The game is over. Start a new game to play again.</p>;
  } else if (!isHumanTurn) {
    prompt = (
      <p className="thinking">
        <span className="thinking-dot" />
        {PLAYER_NAMES[BOT_ID]} is plotting...
      </p>
    );
  } else if (phase === "DISCARD") {
    prompt = <DiscardPanel state={state} onSubmit={onAction} />;
  } else if (steals.length > 0) {
    prompt = (
      <div className="action-group">
        <div className="action-group-head">
          <span>Steal from</span>
        </div>
        <div className="action-buttons">
          {steals.map((a, i) => (
            <button key={i} className="btn" onClick={() => onAction(a)}>
              {actionLabel(a)}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="panel action-panel">
      <div className="dice-tray">
        <Dice
          total={state.dice_roll}
          salt={state.turn_number}
          onClick={rollAction ? () => onAction(rollAction) : undefined}
          hint="click to roll"
          disabled={busy}
        />
      </div>
      {prompt}
    </div>
  );
}
