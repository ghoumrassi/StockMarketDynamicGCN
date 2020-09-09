SELECT ticker, returns{additional_columns} FROM tickerdata

-- INNER JOIN nasdaq100 AS nd
-- ON ticker = nd."Symbol"

-- Restricts to Top companies
INNER JOIN top_by_volume as top
ON
	tickerdata."ticker" = top."ticker"

WHERE
    date = :date

