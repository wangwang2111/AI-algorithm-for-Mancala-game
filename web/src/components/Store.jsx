// Store.jsx
import React, { useEffect, useRef } from "react";

export default function Store({ value = 0, title = "Store", id }) {
  const faceRef = useRef(null);

  useEffect(() => {
    // ensure a scatter layer exists once (imperative; persists across renders)
    const face = faceRef.current;
    if (!face) return;
    let scatter = face.querySelector('.store__scatter');
    if (!scatter) {
      scatter = document.createElement('div');
      scatter.className = 'store__scatter';
      face.appendChild(scatter);
    }
    // Tag for controller lookup
    scatter.dataset.storeId = id ?? title;
  }, [id, title]);

  return (
    <div className="store store__face" ref={faceRef} aria-label={`${title} with ${value} stones`}>
      <div className="store__value">{value}</div>
      <div className="store__title">{title}</div>
      {/* scatter layer is created imperatively and persists */}
    </div>
  );
}
