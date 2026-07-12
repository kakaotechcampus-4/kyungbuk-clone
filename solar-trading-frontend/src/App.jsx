import React, { useState } from 'react';
import Header from './components/Header';
import WeatherBar from './components/WeatherBar';
import PriceCards from './components/PriceCards';
import HouseholdGrid from './components/HouseholdGrid';
import TradingEngine from './components/TradingEngine';
import TradeLog from './components/TradeLog';
import { runTradingEngine } from './utils/tradingEngine';
import './App.css';

const HOUSEHOLDS = [
  { id: 'A', icon: '🏠', gen: 5.0, demand: 2.0 },
  { id: 'B', icon: '🏡', gen: 1.0, demand: 4.0 },
  { id: 'C', icon: '🏘️', gen: 4.0, demand: 2.0 },
  { id: 'D', icon: '🏠', gen: 0.5, demand: 3.5 },
  { id: 'E', icon: '🏡', gen: 6.0, demand: 2.5 },
];

const IRR_NOW = 800;
const IRR_FORECAST = 200;

function App() {
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);

  async function handleRun() {
    setRunning(true);
    setResult(null);
    setStep(0);
    for (let i = 1; i <= 5; i++) {
      await new Promise(r => setTimeout(r, 400));
      setStep(i);
    }
    const res = runTradingEngine(HOUSEHOLDS, IRR_NOW, IRR_FORECAST);
    setResult(res);
    setRunning(false);
  }

  const matchedIds = result
    ? result.trades.flatMap(t => [t.seller, t.buyer])
    : [];

  return (
    <div className="app">
      <Header />
      <div className="main">
        <WeatherBar irrNow={IRR_NOW} irrForecast={IRR_FORECAST} />
        <PriceCards result={result} />
        <HouseholdGrid households={HOUSEHOLDS} matchedIds={matchedIds} />
        <div className="bottom-grid">
          <TradingEngine result={result} running={running} step={step} onRun={handleRun} />
          <TradeLog result={result} />
        </div>
      </div>
    </div>
  );
}

export default App;