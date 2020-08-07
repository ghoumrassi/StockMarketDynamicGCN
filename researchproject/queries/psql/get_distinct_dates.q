SELECT DISTINCT date FROM tickerdata
WHERE
    date >= :startdate AND
    date < :enddate
ORDER BY date asc