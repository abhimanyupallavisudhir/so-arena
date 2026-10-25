# Probe the single-process sandbox (run_python) with and without Landlock.
import sys, os
import oversight_arena.domains._exec as ex
os.environ["OA_SECRET_ENV"] = "parent-secret-123"
PROBE = r'''
import os, sys
def t(name, f):
    try: print(name, "->", repr(f())[:120])
    except BaseException as e: print(name, "-> BLOCKED", type(e).__name__, str(e)[:80])
t("environ", lambda: dict(os.environ))
t("open secret", lambda: open("/tmp/fixrev1/secret.txt").read())
t("parent environ", lambda: open(f"/proc/{os.getppid()}/environ","rb").read()[:200])
t("stat secret", lambda: os.stat("/tmp/fixrev1/secret.txt").st_size)
def rl():
    import readline
    readline.clear_history(); readline.read_history_file("/tmp/fixrev1/secret.txt")
    return [readline.get_history_item(i) for i in range(1, readline.get_current_history_length()+1)]
t("readline read secret", rl)
def rl2():
    import readline
    readline.clear_history(); readline.read_history_file(f"/proc/{os.getppid()}/environ")
    return [readline.get_history_item(i) for i in range(1, readline.get_current_history_length()+1)]
t("readline parent environ", rl2)
def sq():
    import sqlite3
    c = sqlite3.connect(":memory:"); c.execute("VACUUM INTO '/tmp/fixrev1/escaped.db'"); return os.path.exists("/tmp/fixrev1/escaped.db")
t("sqlite VACUUM INTO outside", sq)
def sq2():
    import sqlite3
    c = sqlite3.connect(":memory:"); c.execute("ATTACH '/tmp/fixrev1/victim.db' AS v"); return c.execute("select * from v.t").fetchall()
t("sqlite ATTACH read outside", sq2)
def sock():
    import socket; s = socket.socket(); return s
t("socket", sock)
def mm():
    import _posixsubprocess
t("import _posixsubprocess", mm)
t("os.fork", lambda: os.fork())
'''
import sqlite3
c = sqlite3.connect("/tmp/fixrev1/victim.db"); c.execute("create table if not exists t(x)"); c.execute("delete from t"); c.execute("insert into t values ('PRIVATE_DB_ROW')"); c.commit(); c.close()
if os.path.exists("/tmp/fixrev1/escaped.db"): os.remove("/tmp/fixrev1/escaped.db")
mode = sys.argv[1]
if mode == "nolandlock":
    ex._LIB_SRC = ex._LIB_SRC.replace("def landlock_abi():", "def landlock_abi():\n    return 0\n\ndef _orig_abi():")
elif mode == "none":
    ex._LIB_SRC = ex._LIB_SRC.replace("def landlock_abi():", "def landlock_abi():\n    return 0\n\ndef _orig_abi():").replace('spec = _SECCOMP.get(os.uname().machine)', 'spec = None')
r = ex.run_python(PROBE, timeout=20)
print("MODE", mode, "ok", r.ok); print(r.stdout); print(r.stderr[-500:])
