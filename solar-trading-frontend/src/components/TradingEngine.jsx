import React from 'react';

const STEPS = ['Weather', 'SDR', 'Correction', 'Price', 'Match'];

function TradingEngine({ result, running, step, onRun }) {
  return (
    <div className="engine-card">
      <div className="section-title">Trading Engine</div>
      <div className="steps">
        {STEPS.map((s, i) => (
          <div
            key={s}
            className={`step ${
              result && step > i ? 'done' :
              step === i + 1 ? 'active' : ''
            }`}
          >
            {i + 1}. {s}
          </div>
        ))}
      </div>
      <div className="formula-box">
        {`SDR   = Supply / Demand\nWF    = 1 + (ΔIrr% × 0.3)\nPrice = ₹8 / SDR × WF`}
      </div>
      <div className="summary-rows">
        <div className="summary-row">
          <span className="s-key">Total Supply</span>
          <span className="s-val">{result ? result.totalSupply + ' kWh' : '--'}</span>
        </div>
        <div className="summary-row">
          <span className="s-key">Total Demand</span>
          <span className="s-val">{result ? result.totalDemand + ' kWh' : '--'}</span>
        </div>
        <div className="summary-row">
          <span className="s-key">Base Price</span>
          <span className="s-val">₹8.00 / kWh</span>
        </div>
        <div className="summary-row">
          <span className="s-key">Irradiance Drop</span>
          <span className="s-val">{result ? result.irrDropRate + '%' : '75%'}</span>
        </div>
      </div>
      <button className="run-btn" onClick={onRun} disabled={running}>
        {running ? '⏳ Processing...' : result ? '🔄 Run Again' : '⚡ Run Trading Engine'}
      </button>
    </div>
  );
}

export default TradingEngine;