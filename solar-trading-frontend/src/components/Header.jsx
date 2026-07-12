import React, { useState, useEffect } from 'react';

function Header() {
  const [time, setTime] = useState('');
  const [date, setDate] = useState('');

  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setTime(
        now.getHours().toString().padStart(2, '0') + ':' +
        now.getMinutes().toString().padStart(2, '0')
      );
      setDate(
        now.toLocaleDateString('en-IN', {
          weekday: 'short',
          year: 'numeric',
          month: 'short',
          day: 'numeric'
        })
      );
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="header">
      <div className="logo">
        <div className="logo-icon">☀️</div>
        <div>
          <div className="logo-text">SolarTrade</div>
          <div className="logo-sub">P2P Energy Trading Platform · India</div>
        </div>
      </div>
      <div className="header-right">
        <div className="status-badge">
          <span className="status-dot" />
          Live Simulation
        </div>
        <div className="status-badge">📅 {date}</div>
        <div className="status-badge">🕐 {time}</div>
      </div>
    </div>
  );
}

export default Header;