SELECT DISTINCT ticker FROM tickerdata

INNER JOIN nasdaq100 AS nd
ON ticker = nd."Symbol"

WHERE
    date > :startdate AND
    date < :enddate
ORDER BY ticker asc