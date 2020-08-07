SELECT DISTINCT ticker FROM tickerdata
WHERE
    date > :startdate AND
    date < :enddate
ORDER BY ticker asc