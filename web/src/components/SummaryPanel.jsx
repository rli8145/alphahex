import React from "react";
import {
  BOT_ID,
  HUMAN_ID,
  PLAYER_COLORS,
  PLAYER_NAMES,
  longestRoadLength,
  productionPips,
  victoryPointCards,
  visibleVp,
} from "../format.js";

// Compact at-a-glance summary: victory points, expected production, and
// longest-road pressure for both players.
export default function SummaryPanel({ state }) {
  const rows = [HUMAN_ID, BOT_ID].map((pid) => {
    const player = state.players[pid];
    const hiddenVp = pid === HUMAN_ID ? victoryPointCards(player) : 0;
    return {
      pid,
      vp: visibleVp(state, pid) + hiddenVp,
      hiddenOpponent: pid !== HUMAN_ID,
      pips: productionPips(state, pid),
      road: longestRoadLength(state, pid),
      knights: player.played_knights ?? 0,
      hasLongestRoad: state.longest_road_owner === pid,
      hasLargestArmy: state.largest_army_owner === pid,
    };
  });

  const owner = state.longest_road_owner;
  let roadNote;
  if (owner == null) {
    const leader = rows[0].road >= rows[1].road ? rows[0] : rows[1];
    roadNote =
      leader.road > 0
        ? `Longest Road unclaimed (needs 5+). Best: ${PLAYER_NAMES[leader.pid]} at ${leader.road}.`
        : "Longest Road unclaimed (needs 5+).";
  } else {
    const ownerRow = rows.find((row) => row.pid === owner);
    const rival = rows.find((row) => row.pid !== owner);
    const needed = ownerRow.road + 1 - rival.road;
    roadNote =
      needed > 0
        ? `${PLAYER_NAMES[owner]} holds Longest Road at ${ownerRow.road}. ${PLAYER_NAMES[rival.pid]} needs ${needed} more road${needed === 1 ? "" : "s"}.`
        : `${PLAYER_NAMES[owner]} holds Longest Road at ${ownerRow.road} - ${PLAYER_NAMES[rival.pid]} can take it!`;
  }

  return (
    <div className="panel summary-panel">
      <h2>Summary</h2>
      <table className="summary-table">
        <thead>
          <tr>
            <th></th>
            <th title="Victory points (opponent hidden dev cards not shown)">VP</th>
            <th title="Expected production pips per roll (robber excluded)">Prod</th>
            <th title="Longest single road chain">Road</th>
            <th title="Knights played">Army</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.pid}>
              <td className="summary-name">
                <span className="summary-dot" style={{ background: PLAYER_COLORS[row.pid] }} />
                {PLAYER_NAMES[row.pid]}
              </td>
              <td>
                {row.vp}
                {row.hiddenOpponent ? <span className="summary-hidden" title="May hold hidden VP cards">+?</span> : null}
              </td>
              <td>{row.pips}</td>
              <td>
                {row.road}
                {row.hasLongestRoad ? <span className="summary-badge" title="Holds Longest Road (+2 VP)">LR</span> : null}
              </td>
              <td>
                {row.knights}
                {row.hasLargestArmy ? <span className="summary-badge" title="Holds Largest Army (+2 VP)">LA</span> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="summary-road-note">{roadNote}</p>
    </div>
  );
}
