import sqlite3
import psycopg2
import json
import sqlalchemy

from src import SQLITE_DB, PG_CREDENTIALS


def create_connection(db_file, timeout=30):
    """ Creates a connection to the db. """
    conn = None
    try:
        conn = sqlite3.connect(db_file, timeout=timeout)
        print(sqlite3.version)
    except Exception as e:
        print(e)
    return conn


def create_connection_psql(credentials):
    """ Creates a connection to the db. """
    with open(credentials, 'r') as f:
        j = json.load(f)

    db_string = f"postgres://{j['user']}:{j['password']}@{j['hostname']}:{j['port']}/{j['database']}"
    db = sqlalchemy.create_engine(db_string)

    return db


def execute_query(conn, query):
    """ Execute a query on sqlite DB. """
    try:
        c = conn.cursor()
        c.execute(query)
    except Exception as e:
        pass


if __name__ == '__main__':
    connection = create_connection_psql()
    result_set = connection.execute("SELECT version()")
    print(next(result_set))

