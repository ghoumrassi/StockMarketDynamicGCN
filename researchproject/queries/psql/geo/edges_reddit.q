SELECT "a", "b", CAST(SUM(count) as int) FROM reddit_mentions

-- Restricts to Top companies
INNER JOIN top_by_volume as top
ON
	subsa."ticker" = top."ticker"
INNER JOIN top_by_volume as top2
ON
	subsb."ticker" = top2."ticker"

WHERE "date" <= :date
AND "date" > :prevdate

GROUP BY "a", "b"