SELECT "a", "b", returns FROM returns_correlations

-- Restricts to NASDAQ100 companies
INNER JOIN nasdaq100 AS nd
ON
	returns_correlations."a" = nd."Symbol"
INNER JOIN nasdaq100 AS nd2
ON
	returns_correlations."b" = nd2."Symbol"

WHERE "date" = :date