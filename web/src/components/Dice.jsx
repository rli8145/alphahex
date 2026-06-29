import React, { useEffect, useState } from "react";

// The engine reports only the dice total, so we reconstruct a plausible pair of
// faces that sum to it (varied by `salt` so the same total isn't always shown
// the same way). Purely cosmetic — the total is what the game acts on.
function facePair(total, salt) {
  const opts = [];
  for (let d = Math.max(1, total - 6); d <= Math.min(6, total - 1); d++) opts.push(d);
  const d1 = opts[Math.abs(salt) % opts.length];
  return [d1, total - d1];
}

// Pip positions on a 3x3 grid for each face value.
const PIPS = {
  1: [4],
  2: [0, 8],
  3: [0, 4, 8],
  4: [0, 2, 6, 8],
  5: [0, 2, 4, 6, 8],
  6: [0, 2, 3, 5, 6, 8],
};

function Die({ value, rolling }) {
  const on = new Set(PIPS[value] ?? []);
  return (
    <div className={`die${rolling ? " rolling" : ""}`} aria-label={`die showing ${value}`}>
      {Array.from({ length: 9 }, (_, i) => (
        <span key={i} className={on.has(i) ? "pip on" : "pip"} />
      ))}
    </div>
  );
}

const rand6 = () => 1 + Math.floor(Math.random() * 6);
const ROLL_TICK_MS = 70;
const ROLL_TICKS = 11;

// Always-visible dice. With no real roll yet they default to 6/6. When a roll
// comes in they tumble and settle on the engine's total. Pass `onClick` to make
// them the clickable roll control.
export default function Dice({ total, salt = 0, onClick, hint, disabled = false }) {
  const isReal = total != null;
  const display = isReal ? total : 12; // default faces: 6 + 6
  const [a, b] = facePair(display, salt);
  const [faces, setFaces] = useState([a, b]);
  const [rolling, setRolling] = useState(false);

  // On each new real roll (total/salt change) tumble through random faces, then
  // settle on the reconstructed pair that sums to the engine's total.
  useEffect(() => {
    if (!isReal) {
      setFaces([a, b]);
      setRolling(false);
      return;
    }
    setRolling(true);
    let tick = 0;
    const id = setInterval(() => {
      tick += 1;
      if (tick >= ROLL_TICKS) {
        clearInterval(id);
        setFaces([a, b]);
        setRolling(false);
      } else {
        setFaces([rand6(), rand6()]);
      }
    }, ROLL_TICK_MS);
    return () => clearInterval(id);
  }, [total, salt, a, b, isReal]);

  const inner = (
    <>
      <Die value={faces[0]} rolling={rolling} />
      <Die value={faces[1]} rolling={rolling} />
      {isReal && (
        <span className={`dice-total${total === 7 ? " seven" : ""}`}>{rolling ? "·" : total}</span>
      )}
    </>
  );

  // With a hint, render as the roll control — always visible, greyed (disabled)
  // when there's no roll to make.
  if (hint != null) {
    return (
      <button
        type="button"
        className={`dice dice-roll${rolling ? " is-rolling" : ""}`}
        onClick={onClick}
        disabled={disabled || !onClick}
        aria-label="Roll the dice"
      >
        {inner}
        <span className="dice-roll-hint">{hint}</span>
      </button>
    );
  }
  return <div className={`dice${rolling ? " is-rolling" : ""}`}>{inner}</div>;
}
