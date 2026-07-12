import React from 'react';

function PriceCards({ result }) {
  return (
    <div className="price-grid">
      <div className="price-card highlight">
        <div className="card-label">Current P2P Price</div>
        <div className="card-value amber">
          {result ? `₹${result.price}` : '--'}
        </div>
        <div className="card-unit">₹ / kWh</div>
        {result && (
          <div className={`price-change ${result.price > 8 ? 'up' : 'down'}`}>
            {result.price > 8 ? '▲' : '▼'} ₹{Math.abs(result.price - 8).toFixed(2)} vs base ₹8.00
          </div>
        )}
      </div>

      <div className="price-card">
        <div className="card-label">SDR (Supply / Demand)</div>
        <div className="card-value white">{result ? result.SDR : '--'}</div>
        <div className="card-unit">ratio</div>
        <div className="sdr-bar">
          <div
            className="sdr-fill"
            style={{
              width: result ? Math.min(100, result.SDR * 50) + '%' : '0%',
              background: result && result.SDR >= 1 ? '#10b981' : '#ef4444'
            }}
          />
        </div>
      </div>

      <div className="price-card">
        <div className="card-label">Weather Correction Factor</div>
        <div className="card-value white">{result ? result.weatherFactor : '--'}</div>
        <div className="card-unit">multiplier</div>
        {result && (
          <div className="weather-note">
            Irradiance drops {result.irrDropRate}% → price +{((result.weatherFactor - 1) * 100).toFixed(0)}%
          </div>
        )}
      </div>
    </div>
  );
}

export default PriceCards;