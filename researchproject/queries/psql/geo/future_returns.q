SELECT ticker, SUM(returns) FROM tickerdata

INNER JOIN nasdaq100 AS nd
ON ticker = nd."Symbol"

WHERE
    date = :date
GROUP BY ticker