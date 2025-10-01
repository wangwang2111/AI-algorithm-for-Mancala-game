// Pit.jsx
import React, { useEffect, useRef } from "react";

export default function Pit({ count, side, pitIndex, active, disabled, onClick }) {
  const faceRef = useRef(null);

  useEffect(() => {
    const face = faceRef.current;
    if (!face) return;
    let scatter = face.querySelector('.pit__scatter');
    if (!scatter) {
      scatter = document.createElement('div');
      scatter.className = 'pit__scatter';
      // tag with an id so controllers can find it (player side + pit index)
      scatter.dataset.pitId = `${side}:${pitIndex}`;
      face.appendChild(scatter);
    }
  }, [side, pitIndex]);

  const cls = [
    "pit", active ? "pit--active" : "", disabled ? "pit--disabled" : ""
  ].join(" ").trim();

  return (
    <button className={cls} disabled={disabled} onClick={onClick}>
      <div className="pit__face" ref={faceRef}>
        <span className="pit__count">{count}</span>
        {/* .pit__scatter is created imperatively; no React-rendered seeds */}
        <span className="pit__index">{pitIndex}</span>
      </div>
    </button>
  );
}
