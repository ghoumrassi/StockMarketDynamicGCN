import sqlite3

from src import SQLITE_DB
from src.errors import error_handling


def create_connection(db_file):
    """ Creates a connection to the db. """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Exception as e:
        error_handling(e)
    return conn


def execute_query(conn, query):
    """ Execute a query on SQLite DB. """
    try:
        c = conn.cursor()
        c.execute(query)
    except Exception as e:
        error_handling(e)


if __name__ == '__main__':
    connection = create_connection(SQLITE_DB)
    connection.close()
