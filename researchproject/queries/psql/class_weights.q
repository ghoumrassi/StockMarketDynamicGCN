SELECT cls."class", count(cls."class") FROM (
	SELECT
	       (CASE
	           WHEN returns > :threshold THEN 2
	           WHEN returns < -:threshold THEN 0
	           ELSE 1
	           END
	        )
	           AS "class"
	FROM tickerdata
	INNER JOIN nasdaq100 AS nd
        ON
	ticker = nd."Symbol"
    WHERE date > :startdate AND date <= :enddate
) cls
GROUP BY cls."class"
ORDER BY cls."class"