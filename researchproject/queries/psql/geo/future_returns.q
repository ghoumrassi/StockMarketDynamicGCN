SELECT tickerdata."ticker", SUM(returns) FROM tickerdata

-- INNER JOIN nasdaq100 AS nd
-- ON ticker = nd."Symbol"

-- Restricts to Top companies
INNER JOIN top_by_volume as top
ON
	tickerdata."ticker" = top."ticker"

WHERE
    date > :date
AND
    date <= :futuredate
AND "30-day" IS NOT NULL

GROUP BY tickerdata."ticker"
