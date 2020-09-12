SELECT ticker_x, ticker_y, "jointOwnership" FROM sec_jointownership

-- Restricts to NASDAQ100 companies
INNER JOIN top_by_volume as top
ON
	sec_jointownership."ticker_x" = top."ticker"
INNER JOIN top_by_volume as top2
ON
	sec_jointownership."ticker_y" = top2."ticker"

WHERE "jointOwnership" != 0
AND "relStart" < :date
AND "relEnd" > :prevdate
AND ticker_x < ticker_y