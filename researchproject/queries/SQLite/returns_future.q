SELECT ticker, SUM(returns) FROM tickerdata
WHERE
    date >= ? AND
    date < ?
