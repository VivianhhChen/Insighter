const stocks = [
    { symbol: 'SPY', change: -0.17 },
    { symbol: 'NASDAQ', change: -0.25 },
    { symbol: '^FTSE', change: 0.41 },
    { symbol: 'HSBA.L', change: -0.63 },
    { symbol: 'TSLA', change: 1.23 },
    { symbol: 'AAPL', change: 1.50 },
    { symbol: 'GOOGL', change: 2.20 },
    { symbol: 'AMZN', change: -0.80 },
    { symbol: 'AMD', change: 0.35 },
    { symbol: 'QQQ', change: -0.23 },
    { symbol: 'HANG SENG', change: 0.23 },
    { symbol: 'GRUDE OIL', change: -1.39 },
    { symbol: 'GOLD FUTURES', change: -0.18 }
];

function updateStocks() {
    const stockContainer = document.querySelector('.stock-scroll-container .scrolling-container');
    stockContainer.innerHTML = '';

    stocks.forEach(stock => {
        const stockItem = document.createElement('div');
        stockItem.classList.add('stock-item');
        
        const symbolSpan = document.createElement('span');
        symbolSpan.textContent = stock.symbol;
        symbolSpan.classList.add('stock-item-symbol');
        stockItem.appendChild(symbolSpan);
        
        const colorClass = stock.change >= 0 ? 'green' : 'red';
        stockItem.classList.add(colorClass);

        const changeSpan = document.createElement('span');
        changeSpan.textContent = ` ${stock.change > 0 ? '+' : ''}${stock.change}%`;
        stockItem.appendChild(changeSpan);
        
        stockContainer.appendChild(stockItem);
    });
}


function init() {
    updateStocks();
    setInterval(updateStocks, 30000); // Refresh
}

window.onload = init;

