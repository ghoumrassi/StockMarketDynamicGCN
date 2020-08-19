SELECT ticker, returns{additional_columns} FROM tickerdata
WHERE
    date = :date

