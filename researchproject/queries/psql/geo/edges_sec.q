SELECT ticker_x, ticker_y, "jointOwnership" FROM sec_jointownership

-- Restricts to NASDAQ100 companies
INNER JOIN nasdaq100 AS nd
ON
	sec_jointownership."ticker_x" = nd."Symbol"
INNER JOIN nasdaq100 AS nd2
ON
	sec_jointownership."ticker_y" = nd2."Symbol"

WHERE "jointOwnership" != 0
AND "relStart" <= :date
AND "relEnd" > :prevdate