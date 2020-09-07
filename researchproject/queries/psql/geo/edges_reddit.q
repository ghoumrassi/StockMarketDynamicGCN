SELECT "a", "b", CAST(SUM(count) as int) FROM reddit_mentions

-- Restricts to NASDAQ100 companies
INNER JOIN nasdaq100 AS nd
ON
	reddit_mentions."a" = nd."Symbol"
INNER JOIN nasdaq100 AS nd2
ON
	reddit_mentions."b" = nd2."Symbol"

WHERE "date" <= :date
AND "date" > :prevdate

GROUP BY "a", "b"