select "relStart", "relEnd", ticker_x, ticker_y, "jointOwnership" from sec_jointownership
where "jointOwnership" != 0
and "relStart" <= :enddate
and "relEnd" >= :startdate