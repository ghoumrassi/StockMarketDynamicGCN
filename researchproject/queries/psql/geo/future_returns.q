SELECT ticker, SUM(returns) FROM tickerdata

INNER JOIN nasdaq100 AS nd
ON ticker = nd."Symbol"

WHERE
    date > :date
AND
    date <= :futuredate
GROUP BY ticker