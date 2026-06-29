import React, { useState } from "react";
import {
  DEV_CARD_META,
  DEV_CARD_ORDER,
  HUMAN_ID,
  PLAYER_COLORS,
  PLAYER_NAMES,
  RESOURCE_META,
  RESOURCE_ORDER,
  victoryPointCards,
  visibleVp,
} from "../format.js";

const PLAY_TYPE = {
  KNIGHT: "PLAY_KNIGHT",
  MONOPOLY: "PLAY_MONOPOLY",
  YEAR_OF_PLENTY: "PLAY_YEAR_OF_PLENTY",
  ROAD_BUILDING: "PLAY_ROAD_BUILDING",
};
const multisetKey = (arr) => [...arr].sort().join(",");

function totalResources(resources) {
  return Object.values(resources).reduce((a, b) => a + b, 0);
}
function totalDev(player) {
  const sum = (obj) => Object.values(obj).reduce((a, b) => a + b, 0);
  return sum(player.dev_cards) + sum(player.new_dev_cards);
}

export default function PlayerPanel({ state, playerId, targetVp, byType = {}, onAction, onToggleMode, actionMode }) {
  const player = state.players[playerId];
  const isHuman = playerId === HUMAN_ID;
  const isTurn = state.current_player === playerId;
  // You see your own hidden Victory Point cards; the opponent's stay secret.
  const vp = visibleVp(state, playerId) + (isHuman ? victoryPointCards(player) : 0);

  const [picker, setPicker] = useState(null); // "MONOPOLY" | "YOP"
  const [yopPick, setYopPick] = useState([]);

  // The opponent panel shows a compact stat line (no hidden resource chips).
  if (!isHuman) {
    return (
      <div className={`panel player-panel ${isTurn ? "active-turn" : ""}`}>
        <div className="player-head">
          <span className="player-chip" style={{ background: PLAYER_COLORS[playerId] }} />
          <h2>{PLAYER_NAMES[playerId]}</h2>
          {isTurn && <span className="turn-tag">to move</span>}
          <span className="vp-badge">
            {vp}/{targetVp} VP
          </span>
        </div>

        <div className="badges">
          {state.longest_road_owner === playerId && <span className="badge">Longest Road</span>}
          {state.largest_army_owner === playerId && <span className="badge">Largest Army ({player.played_knights}⚔)</span>}
        </div>

        <div className="stockpile">
          <span>🃏 {totalDev(player)} dev</span>
          <span>🃏 {totalResources(player.resources)} res</span>
          <span>🏠 {5 - player.settlements_remaining}</span>
          <span>🏛️ {4 - player.cities_remaining}</span>
          <span>🛣️ {15 - player.roads_remaining}</span>
        </div>
      </div>
    );
  }

  const playable = (c) => !!PLAY_TYPE[c] && (byType[PLAY_TYPE[c]]?.length ?? 0) > 0;
  const cardActive = (c) =>
    (c === "KNIGHT" && actionMode === "KNIGHT") ||
    (c === "ROAD_BUILDING" && actionMode === "ROAD_BUILDING") ||
    (c === "MONOPOLY" && picker === "MONOPOLY") ||
    (c === "YEAR_OF_PLENTY" && picker === "YOP");

  const clickCard = (c) => {
    if (!playable(c)) return;
    if (c === "KNIGHT") { setPicker(null); onToggleMode("KNIGHT"); }
    else if (c === "ROAD_BUILDING") { setPicker(null); onToggleMode("ROAD_BUILDING"); }
    else if (c === "MONOPOLY") { if (actionMode) onToggleMode(null); setYopPick([]); setPicker((p) => (p === "MONOPOLY" ? null : "MONOPOLY")); }
    else if (c === "YEAR_OF_PLENTY") { if (actionMode) onToggleMode(null); setYopPick([]); setPicker((p) => (p === "YOP" ? null : "YOP")); }
  };

  const playMonopoly = (res) => {
    const a = (byType.PLAY_MONOPOLY ?? []).find((x) => x.payload.resource === res);
    if (a) { onAction(a); setPicker(null); }
  };
  const addYop = (res) => setYopPick((cur) => (cur.length >= 2 ? [res] : [...cur, res]));
  const yopAction =
    yopPick.length === 2
      ? (byType.PLAY_YEAR_OF_PLENTY ?? []).find((x) => multisetKey(x.payload.resources) === multisetKey(yopPick))
      : null;
  const takeYop = () => { if (yopAction) { onAction(yopAction); setPicker(null); setYopPick([]); } };

  const held = DEV_CARD_ORDER.filter((c) => (player.dev_cards[c] || 0) + (player.new_dev_cards[c] || 0) > 0);

  return (
    <div className={`panel player-panel ${isTurn ? "active-turn" : ""}`}>
      <div className="player-head">
        <span className="player-chip" style={{ background: PLAYER_COLORS[playerId] }} />
        <h2>{PLAYER_NAMES[playerId]}</h2>
        {isTurn && <span className="turn-tag">to move</span>}
        <span className="vp-badge">
          {vp}/{targetVp} VP
        </span>
      </div>

      <div className="badges">
        {state.longest_road_owner === playerId && <span className="badge">Longest Road</span>}
        {state.largest_army_owner === playerId && <span className="badge">Largest Army ({player.played_knights}⚔)</span>}
      </div>

      <div className="resources">
        {RESOURCE_ORDER.map((res) => (
          <div className="resource" key={res} title={RESOURCE_META[res].label}>
            <span className="resource-emoji">{RESOURCE_META[res].emoji}</span>
            <span className="resource-count">{player.resources[res] ?? 0}</span>
          </div>
        ))}
      </div>

      <div className="stockpile">
        <span>🏠 {5 - player.settlements_remaining}/5</span>
        <span>🏛️ {4 - player.cities_remaining}/4</span>
        <span>🛣️ {15 - player.roads_remaining}/15</span>
      </div>

      <div className="dev-hand">
        <span className="dev-hand-label">Dev cards</span>
        {held.length === 0 ? (
          <span className="dev-hand-empty">none</span>
        ) : (
          <div className="dev-hand-cards">
            {held.map((c) => {
              const count = (player.dev_cards[c] || 0) + (player.new_dev_cards[c] || 0);
              const canPlay = playable(c);
              const Tag = canPlay ? "button" : "span";
              return (
                <Tag
                  key={c}
                  className={`dev-card${canPlay ? " playable" : ""}${cardActive(c) ? " active" : ""}`}
                  title={canPlay ? `Play ${DEV_CARD_META[c].label}` : DEV_CARD_META[c].label}
                  onClick={canPlay ? () => clickCard(c) : undefined}
                >
                  <span className="dev-card-emoji">{DEV_CARD_META[c].emoji}</span>
                  <span className="dev-card-count">{count}</span>
                </Tag>
              );
            })}
          </div>
        )}
      </div>

      {picker === "MONOPOLY" && (byType.PLAY_MONOPOLY?.length ?? 0) > 0 && (
        <div className="inset">
          <p className="dev-hint">Monopoly — take all of one resource</p>
          <div className="res-chips">
            {RESOURCE_ORDER.map((r) => (
              <button key={r} className="res-chip" onClick={() => playMonopoly(r)} title={RESOURCE_META[r].label}>
                <span className="res-chip-emoji">{RESOURCE_META[r].emoji}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {picker === "YOP" && (byType.PLAY_YEAR_OF_PLENTY?.length ?? 0) > 0 && (
        <div className="inset">
          <p className="dev-hint">
            Year of Plenty — pick 2{" "}
            {yopPick.length > 0 && <strong>{yopPick.map((r) => RESOURCE_META[r].emoji).join(" ")}</strong>}
          </p>
          <div className="res-chips">
            {RESOURCE_ORDER.map((r) => (
              <button key={r} className="res-chip" onClick={() => addYop(r)} title={RESOURCE_META[r].label}>
                <span className="res-chip-emoji">{RESOURCE_META[r].emoji}</span>
              </button>
            ))}
          </div>
          <div className="confirm-actions">
            <button className="btn-primary" disabled={!yopAction} onClick={takeYop}>Take</button>
            {yopPick.length > 0 && <button className="btn-secondary" onClick={() => setYopPick([])}>Clear</button>}
          </div>
        </div>
      )}
    </div>
  );
}
