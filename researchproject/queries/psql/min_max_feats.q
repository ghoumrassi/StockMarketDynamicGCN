select MIN("{feature}"), MAX("{feature}") FROM tickerdata

-- Restricts to Top companies
INNER JOIN top_by_volume as top
ON
	tickerdata."ticker" = top."ticker"

WHERE date >= :startdate AND date < :enddate
