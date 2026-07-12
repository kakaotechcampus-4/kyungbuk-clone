import React from 'react';

function HouseholdGrid({ households, matchedIds }) {
  return (
    <>
      <div className="section-title">Households</div>
      <div className="households-grid">
        {households.map(h => {
          const surplus = h.gen - h.demand;
          const isSeller = surplus > 0;
          const isMatched = matchedIds.includes(h.id);
          return (
            <div
              key={h.id}
              className={`household-card ${isSeller ? 'seller' : 'buyer'} ${isMatched ? 'matched' : ''}`}
            >
              <div className={`role-badge ${isSeller ? 'badge-sell' : 'badge-buy'}`}>
                {isSeller ? 'SELL' : 'BUY'}
              </div>
              <div className="h-icon">{h.icon}</div>
              <div className="h-id">House {h.id}</div>
              <div className="h-gen">⚡ Gen: {h.gen} kWh</div>
              <div className="h-demand">📊 Use: {h.demand} kWh</div>
              <div className={`h-surplus ${surplus >= 0 ? 'pos' : 'neg'}`}>
                {surplus >= 0 ? '+' : ''}{surplus.toFixed(1)} kWh
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

export default HouseholdGrid;