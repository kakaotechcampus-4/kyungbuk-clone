import React from 'react';

function WeatherBar({ irrNow, irrForecast }) {
  return (
    <div className="weather-bar">
      <div className="weather-item">
        <div className="weather-label">Irradiance (Now)</div>
        <div className="weather-value">{irrNow} W/m²</div>
      </div>
      <div className="divider" />
      <div className="weather-item">
        <div className="weather-label">Irradiance (2hr Forecast)</div>
        <div className="weather-value" style={{ color: '#ef4444' }}>{irrForecast} W/m²</div>
      </div>
      <div className="divider" />
      <div className="weather-item">
        <div className="weather-label">Temperature</div>
        <div className="weather-value">32°C</div>
      </div>
      <div className="divider" />
      <div className="weather-item">
        <div className="weather-label">Condition</div>
        <div className="weather-value">⛅ Partly Cloudy</div>
      </div>
      <div className="forecast-badge">
        ⚠️ Cloud cover expected in 2hr → Price rising
      </div>
    </div>
  );
}

export default WeatherBar;