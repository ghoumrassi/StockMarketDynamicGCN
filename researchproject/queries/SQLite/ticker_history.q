SELECT date, ticker, returns{additional_columns} FROM tickerdata
WHERE
    date >= ? AND
    date < ?
