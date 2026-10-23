import pytest

from app.kevdb import connect_kev
from app.sqlguard import GuardedSQL, SqlRejected


@pytest.fixture(scope="module")
def g():
    return GuardedSQL(connect_kev())


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM kev",
    "SELECT vendor, COUNT(*) n FROM kev GROUP BY vendor ORDER BY n DESC LIMIT 3;",
    "SELECT strftime('%Y', date_added) y, COUNT(*) FROM kev GROUP BY y",
    "WITH t AS (SELECT vendor, COUNT(*) n FROM kev GROUP BY vendor) SELECT * FROM t WHERE n > 100",
    "SELECT CAST(SUM(ransomware_use='Known') AS REAL) / COUNT(*) FROM kev",
])
def test_reads_are_allowed(g, sql):
    assert g.run(sql).rows


@pytest.mark.parametrize("sql", [
    "DROP TABLE kev", "DELETE FROM kev", "UPDATE kev SET vendor='x'", "INSERT INTO kev(cve_id) VALUES('x')",
    "SELECT 1; DROP TABLE kev", "PRAGMA table_info(kev)", "ATTACH DATABASE 'x.db' AS x",
    "SELECT * FROM sqlite_master", "SELECT load_extension('x')", "SELECT randomblob(10)",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r",
    "SELECT * FROM nope", "", "SELEC * FROM kev", "CREATE TABLE x(a)", "VACUUM",
])
def test_everything_else_is_refused(g, sql):
    with pytest.raises(SqlRejected):
        g.run(sql)


@pytest.mark.parametrize("sql", [
    "WITH x AS (SELECT 1) DELETE FROM kev",
    "WITH x AS (SELECT 1) UPDATE kev SET vendor='pwned'",
    "WITH x AS (SELECT 1) INSERT INTO kev(cve_id) VALUES('CVE-0')",
])
def test_write_hidden_behind_with_is_stopped_by_the_authorizer(g, sql):
    # passes the cheap "starts with SELECT/WITH" check; only the engine-level authorizer stops it
    with pytest.raises(SqlRejected):
        g.run(sql)
    assert g.run("SELECT COUNT(*) FROM kev WHERE vendor='pwned'").rows == [(0,)]


def test_query_only_blocks_writes_even_without_the_authorizer():
    con = connect_kev()
    GuardedSQL(con)
    con.set_authorizer(None)
    with pytest.raises(Exception, match="readonly"):
        con.execute("DELETE FROM kev")


def test_rows_are_capped_and_flagged(g):
    r = g.run("SELECT cve_id FROM kev")
    assert r.truncated and len(r.rows) == 50 and "truncated" in r.to_text()


def test_overlong_query_is_refused(g):
    with pytest.raises(SqlRejected, match="longer"):
        g.run("SELECT " + "1," * 2000 + "1")


def test_result_text_clips_long_cells(g):
    assert "…" in g.run("SELECT description FROM kev LIMIT 1").to_text(max_cell=10)
