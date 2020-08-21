SELECT ticker, returns{additional_columns} FROM tickerdata

INNER JOIN nasdaq100 AS nd
ON ticker = nd."Symbol"

WHERE
    date = :date

