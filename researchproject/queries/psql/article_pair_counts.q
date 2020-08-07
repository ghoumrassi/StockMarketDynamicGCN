-- noinspection SqlNoDataSourceInspectionForFile

-- Returns the count of each (date, ticker,ticker) combination

SELECT
    summaries."Date" AS "date",
    subsa."ticker" AS "a",
    subsb."ticker" AS "b",
    COUNT(subsa."ticker") AS "count"
FROM summaries

LEFT JOIN companymapper AS mappera
ON summaries."Company A" = mappera."index"

LEFT JOIN companymapper AS mapperb
ON summaries."Company B" = mapperb."index"

LEFT JOIN subsidiaries AS subsa
ON mappera."company" = subsa."altLabel"

LEFT JOIN subsidiaries AS subsb
ON mapperb."company" = subsb."altLabel"

WHERE
    subsa."exchangeLabel" = 'NASDAQ'
        AND
    subsb."exchangeLabel" = 'NASDAQ'
        AND
    -- Added without testing, if query breaks - this is #1 suspect.
    subsa."ticker" != subsb."ticker"
        AND
    summaries."Date" >= :startdate
        AND
    summaries."Date" < :enddate

GROUP BY summaries."Date", subsa."ticker", subsb."ticker"