const BASE_PRICE = 8.0;
const MAX_PRICE = 15.0;
const MIN_PRICE = 3.0;

export function calculateSDR(totalSupply, totalDemand) {
  if (totalDemand === 0) return 99;
  return totalSupply / totalDemand;
}

export function calculateWeatherFactor(irrNow, irrForecast) {
  if (irrNow === 0) return 1;
  const dropRate = (irrNow - irrForecast) / irrNow;
  return 1 + dropRate * 0.3;
}

export function calculatePrice(SDR, weatherFactor) {
  const sdrPrice = BASE_PRICE / SDR;
  const finalPrice = sdrPrice * weatherFactor;
  return Math.min(MAX_PRICE, Math.max(MIN_PRICE, finalPrice));
}

export function matchTrades(households, price) {
  // 판매자: 잉여 많은 순으로 정렬 (Double Auction 방식)
  const sellers = households
    .filter(h => h.gen - h.demand > 0)
    .map(h => ({ ...h, avail: parseFloat((h.gen - h.demand).toFixed(2)) }))
    .sort((a, b) => b.avail - a.avail);  // 잉여 많은 순

  // 구매자: 부족량 많은 순으로 정렬
  const buyers = households
    .filter(h => h.gen - h.demand < 0)
    .map(h => ({ ...h, need: parseFloat(Math.abs(h.gen - h.demand).toFixed(2)) }))
    .sort((a, b) => b.need - a.need);  // 부족량 많은 순

  const trades = [];

  for (const buyer of buyers) {
    for (const seller of sellers) {
      if (seller.avail <= 0 || buyer.need <= 0) continue;

      const amount = Math.min(seller.avail, buyer.need);

      trades.push({
        seller: seller.id,
        buyer: buyer.id,
        amount: parseFloat(amount.toFixed(2)),
        price: parseFloat(price.toFixed(2)),
        total: parseFloat((amount * price).toFixed(2))
      });

      seller.avail = parseFloat((seller.avail - amount).toFixed(2));
      buyer.need = parseFloat((buyer.need - amount).toFixed(2));
    }
  }

  return trades;
}

export function runTradingEngine(households, irrNow, irrForecast) {
  const sellers = households.filter(h => h.gen - h.demand > 0);
  const buyers  = households.filter(h => h.gen - h.demand < 0);

  const totalSupply = sellers.reduce((s, h) => s + (h.gen - h.demand), 0);
  const totalDemand = Math.abs(buyers.reduce((s, h) => s + (h.gen - h.demand), 0));

  const SDR = calculateSDR(totalSupply, totalDemand);
  const weatherFactor = calculateWeatherFactor(irrNow, irrForecast);
  const price = calculatePrice(SDR, weatherFactor);
  const trades = matchTrades(households, price);

  return {
    totalSupply: parseFloat(totalSupply.toFixed(2)),
    totalDemand: parseFloat(totalDemand.toFixed(2)),
    SDR: parseFloat(SDR.toFixed(3)),
    weatherFactor: parseFloat(weatherFactor.toFixed(3)),
    price: parseFloat(price.toFixed(2)),
    trades,
    irrDropRate: parseFloat(((irrNow - irrForecast) / irrNow * 100).toFixed(1))
  };
}