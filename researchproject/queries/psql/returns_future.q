SELECT ticker, SUM(returns) FROM tickerdata

WHERE
    date >= :startdate AND
    date < :enddate
GROUP BY ticker