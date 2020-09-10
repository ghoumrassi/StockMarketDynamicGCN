SELECT "a", "b", CAST(SUM(count) as int) FROM reddit_mentions

-- Restricts to Top companies
INNER JOIN top_by_volume as top
ON
	reddit_mentions."a" = top."ticker"
INNER JOIN top_by_volume as top2
ON
	reddit_mentions."b" = top2."ticker"

WHERE "date" <= :date
AND "date" > :prevdate

GROUP BY "a", "b"