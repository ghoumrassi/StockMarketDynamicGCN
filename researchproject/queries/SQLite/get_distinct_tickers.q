SELECT DISTINCT ticker FROM tickerdata
WHERE
    date > ? AND
    date < ?
ORDER BY ticker asc