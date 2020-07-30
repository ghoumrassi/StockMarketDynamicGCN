SELECT DISTINCT date FROM tickerdata
WHERE
    date >= ? AND
    date < ?
ORDER BY date asc