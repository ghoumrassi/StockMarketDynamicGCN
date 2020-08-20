SELECT ticker, SUM(returns) FROM tickerdata
WHERE
    date = :date
GROUP BY ticker