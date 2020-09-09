SELECT ticker_x, ticker_y, "jointOwnership" FROM sec_jointownership

-- Restricts to Top companies
INNER JOIN top_by_volume as top
ON
	subsa."ticker" = top."ticker"
INNER JOIN top_by_volume as top2
ON
	subsb."ticker" = top2."ticker"

WHERE "jointOwnership" != 0
AND "relStart" <= :date
AND "relEnd" > :prevdate