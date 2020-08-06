import sqlite3

from src import SQLITE_DB


def create_connection(db_file, timeout=30):
    """ Creates a connection to the db. """
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=timeout)
        print(sqlite3.version)
    except Exception as e:
        pass
    return conn


def execute_query(conn, query):
    """ Execute a query on SQLite DB. """
    try:
        c = conn.cursor()
        c.execute(query)
    except Exception as e:
        pass


if __name__ == '__main__':
    connection = create_connection(SQLITE_DB)
    connection.close()
