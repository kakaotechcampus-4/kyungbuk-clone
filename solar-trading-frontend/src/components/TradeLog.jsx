import React from 'react';

function TradeLog({ result }) {
  if (!result) {
    return (
      <div className="log-card">
        <div className="section-title">Trade Log</div>
        <div className="empty-log">No trades yet. Run the trading engine to begin.</div>
      </div>
    );
  }

  const totalTraded = result.trades.reduce((s, t) => s + t.amount, 0).toFixed(2);
  const totalRevenue = result.trades.reduce((s, t) => s + t.total, 0).toFixed(2);

  return (
    <div className="log-card">
      <div className="section-title">Trade Log</div>
      {result.trades.map((t, i) => (
        <div key={i} className="trade-item">
          <div className="trade-flow">
            <span className="t-seller">House {t.seller}</span>
            <span className="t-arrow"> → </span>
            <span className="t-buyer">House {t.buyer}</span>
          </div>
          <div className="trade-right">
            <div className="t-amount">{t.amount} kWh</div>
            <div className="t-price">₹{t.price}/kWh</div>
            <div className="t-total">Total: ₹{t.total}</div>
          </div>
        </div>
      ))}
      <div className="log-summary">
        <div className="summary-row">
          <span className="s-key">Total Traded</span>
          <span className="s-val">{totalTraded} kWh</span>
        </div>
        <div className="summary-row">
          <span className="s-key">Total Revenue</span>
          <span className="s-val amber">₹{totalRevenue}</span>
        </div>
        <div className="summary-row">
          <span className="s-key">Trades Matched</span>
          <span className="s-val">{result.trades.length} trades</span>
        </div>
      </div>
    </div>
  );
}

export default TradeLog;