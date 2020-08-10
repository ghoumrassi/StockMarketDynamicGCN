SELECT DISTINCT cikmapper."cik_str" FROM tickerdata

INNER JOIN cikmapper
ON tickerdata."ticker" = cikmapper."ticker"