"""Confinement for model-written code, and a trusted/untrusted bridge for grading it.

Stdlib-only and self-contained: the parent process imports this module, and generated scripts
embed its source verbatim (see ``_exec``), so it must not import the package.

A *confined* process applies these layers to itself before any untrusted code runs:

1. rlimits: memory, CPU time, file size, no core dumps, and no new processes or threads
   (``RLIMIT_NPROC = 0``, enforced by the kernel for non-root users);
2. Landlock (Linux >= 5.13, when enabled): a kernel-enforced filesystem allow-list. Reads only
   from the Python installation, system directories and explicitly allowed paths; writes only in
   the sandbox directory. No /proc, home directory, data caches or harness files;
3. seccomp (Linux x86-64 and arm64): no sockets, no signals to other processes, no ptrace or
   cross-process memory access, no exec, no namespaces, io_uring, BPF or keyrings;
4. a PEP 578 audit hook: the same policy at the Python level, with readable errors. It is the only
   filesystem layer where Landlock is unavailable (e.g. macOS), where it is best-effort.

Plus a parent-death signal and an empty environment with HOME and TMPDIR in the sandbox.

The *bridge* (:class:`Untrusted`, :func:`serve`) runs untrusted code in a confined server process
and trusted code (tests, checks) in the harness process. Values cross as tagged JSON and are
rebuilt as plain Python objects, so untrusted code cannot reach the harness's memory, output or
exit status, and cannot smuggle objects with a forged ``__eq__`` into a comparison.
"""

import json
import os
import sys

if __name__ == "__main__":  # only in the sandbox's own processes, never in a host that imports this
    try:  # honest results such as 10**5000 (HumanEval/83, /139) exceed Python's 4300-digit int<->str guard
        sys.set_int_max_str_digits(600_000)  # bounded, but far above any honest grading result
    except AttributeError:
        pass

SYSTEM_READ = (
    "/usr", "/lib", "/lib32", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/localtime", "/etc/timezone",
    "/etc/mime.types", "/etc/os-release", "/dev/urandom", "/dev/random", "/dev/zero",
)
DEV_RW = ("/dev/null",)


def python_roots():
    """Directories holding the running Python installation and its packages."""
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
             os.path.dirname(os.path.realpath(sys.executable))}
    try:
        import sysconfig

        for k in ("stdlib", "platstdlib", "purelib", "platlib"):
            roots.add(sysconfig.get_path(k))
    except Exception:
        pass
    return sorted(os.path.realpath(r) for r in roots if r)


def default_read_roots(extra=()):
    return list(python_roots()) + list(SYSTEM_READ) + [os.path.realpath(p) for p in extra]


# ============================================================================ confinement
def _libc():
    import ctypes

    return ctypes, ctypes.CDLL(None, use_errno=True)


def landlock_abi():
    """Landlock ABI version supported by the kernel (0 = unavailable)."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        ctypes, libc = _libc()
        v = libc.syscall(ctypes.c_long(444), None, ctypes.c_long(0), ctypes.c_long(1))
        return v if v > 0 else 0
    except Exception:
        return 0


def _landlock(read_roots, write_roots):
    import stat
    import struct

    ctypes, libc = _libc()
    L = ctypes.c_long
    abi = landlock_abi()
    if abi < 1:
        return False
    handled = (1 << 13) - 1  # ABI 1: every filesystem right
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE
    attr = ctypes.create_string_buffer(struct.pack("=Q", handled), 8)
    rs = libc.syscall(L(444), attr, L(8), L(0))
    if rs < 0:
        return False
    read, file_rights = 1 | 4 | 8, 1 | 2 | 4 | (1 << 14)  # execute, read file, read dir / rights valid on files

    def allow(path, access):
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            acc = access & handled
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                acc &= file_rights
            if acc:
                rule = ctypes.create_string_buffer(struct.pack("=Qi", acc, fd), 12)
                libc.syscall(L(445), L(rs), L(1), rule, L(0))
        finally:
            os.close(fd)

    try:
        for p in read_roots:
            allow(p, read)
        for p in DEV_RW:
            allow(p, 2 | 4)
        for p in write_roots:
            allow(p, handled)
        libc.prctl(L(38), L(1), L(0), L(0), L(0))  # PR_SET_NO_NEW_PRIVS
        return libc.syscall(L(446), L(rs), L(0)) == 0
    finally:
        os.close(rs)


# (audit arch, syscall numbers); signals are allowed only to the process itself
_SECCOMP = {
    "x86_64": (0xC000003E, {
        "deny": [41, 53, 101, 310, 311, 312, 424, 434, 438, 425, 426, 427, 321, 323, 298, 248, 249, 250, 272, 308],
        "exec": [59, 322],  # execve, execveat
        "self": [62, 200, 234, 129, 297],  # kill tkill tgkill rt_sigqueueinfo rt_tgsigqueueinfo
    }),
    "aarch64": (0xC00000B7, {
        "deny": [198, 199, 117, 270, 271, 272, 424, 434, 438, 425, 426, 427, 280, 282, 241, 217, 218, 219, 97, 268],
        "exec": [221, 281],
        "self": [129, 130, 131, 138, 240],
    }),
}


def _seccomp(allow_exec=False):
    import struct

    spec = _SECCOMP.get(os.uname().machine)
    if spec is None:
        return False
    arch, nr = spec
    ctypes, libc = _libc()
    L = ctypes.c_long
    LD, JEQ, JSET, RET = 0x20, 0x15, 0x45, 0x06
    ALLOW, EPERM, KILL = 0x7FFF0000, 0x00050001, 0x80000000
    prog = []

    def op(code, jt, jf, k):
        prog.append(struct.pack("=HBBI", code, jt, jf, k))

    op(LD, 0, 0, 4)  # seccomp_data.arch
    op(JEQ, 1, 0, arch)
    op(RET, 0, 0, KILL)  # foreign ABI (e.g. int 0x80): refuse outright
    op(LD, 0, 0, 0)  # seccomp_data.nr
    if arch == 0xC000003E:
        op(JSET, 0, 1, 0x40000000)  # x32 ABI
        op(RET, 0, 0, EPERM)
    for n in nr["deny"] + ([] if allow_exec else nr["exec"]):
        op(JEQ, 0, 1, n)
        op(RET, 0, 0, EPERM)
    pid = os.getpid()
    for n in nr["self"]:
        op(JEQ, 0, 4, n)
        op(LD, 0, 0, 16)  # low word of args[0]
        op(JEQ, 1, 0, pid)
        op(RET, 0, 0, EPERM)
        op(RET, 0, 0, ALLOW)
    op(RET, 0, 0, ALLOW)
    code = b"".join(prog)
    buf = ctypes.create_string_buffer(code, len(code))

    class Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fp = Fprog(len(prog), ctypes.addressof(buf))
    libc.prctl(L(38), L(1), L(0), L(0), L(0))
    return libc.prctl(L(22), L(2), ctypes.byref(fp), L(0), L(0)) == 0  # PR_SET_SECCOMP, FILTER


def _rlimits(mem_mb, cpu_s, no_fork):
    import resource

    def cap(res, v):
        _, hard = resource.getrlimit(res)
        v = v if hard == resource.RLIM_INFINITY else min(v, hard)
        resource.setrlimit(res, (v, v))

    if mem_mb:
        cap(resource.RLIMIT_AS, int(mem_mb) << 20)
    if cpu_s:
        cap(resource.RLIMIT_CPU, int(cpu_s))
    cap(resource.RLIMIT_FSIZE, 16 << 20)
    cap(resource.RLIMIT_CORE, 0)
    if no_fork:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def confine(read_roots, write_roots, mem_mb=None, cpu_s=None, parent=None, no_fork=True, audit=True,
            require_kernel=False, allow_exec=False):
    """Confine the current process (irreversibly). Returns the layers that are active.

    ``allow_exec`` keeps ``execve`` available (for :func:`exec_confined`); Landlock, seccomp and
    rlimits survive it, so the new program stays confined.

    ``require_kernel``: on Linux, refuse to continue (exit) if the kernel filesystem sandbox
    (Landlock) did not activate, rather than falling back silently to the Python audit hook. This
    is the safe default for running untrusted code; set ``OA_ALLOW_NO_KERNEL_SANDBOX=1`` to override
    (only inside a container that already isolates the filesystem)."""
    layers = {}
    if parent is not None and sys.platform.startswith("linux"):
        ctypes, libc = _libc()
        libc.prctl(ctypes.c_long(1), ctypes.c_long(9), ctypes.c_long(0), ctypes.c_long(0), ctypes.c_long(0))
        if os.getppid() != parent:  # the parent died before the death signal was armed
            os._exit(1)
    read_roots = [os.path.realpath(p) for p in read_roots]
    write_roots = [os.path.realpath(p) for p in write_roots]
    layers["landlock"] = _landlock(read_roots, write_roots) if sys.platform.startswith("linux") else False
    layers["seccomp"] = _seccomp(allow_exec) if sys.platform.startswith("linux") else False
    if require_kernel and sys.platform.startswith("linux") and not layers["landlock"] \
            and os.environ.get("OA_ALLOW_NO_KERNEL_SANDBOX") != "1":
        # fail closed: the audit hook alone is not enough to grade untrusted code (e.g. it cannot see
        # readline reading /proc, and an old kernel or restricted container may lack Landlock)
        sys.stderr.write("OA-SANDBOX-FAILOPEN: kernel filesystem sandbox (Landlock) unavailable; refusing to "
                         "run untrusted code. Run inside a container and set OA_ALLOW_NO_KERNEL_SANDBOX=1 to override, "
                         "or plug in your own executor with domains._exec.set_executor.\n")
        sys.stderr.flush()
        os._exit(3)
    _rlimits(mem_mb, cpu_s, no_fork)
    layers["rlimits"] = True
    if audit:
        install_audit_hook(read_roots, write_roots)
    layers["audit"] = audit
    for m in ("ctypes", "_ctypes", "ctypes._endian"):
        sys.modules.pop(m, None)
    return layers


def install_audit_hook(read_roots, write_roots):
    """Python-level policy (PEP 578). Only C builtins captured here are used at check time, so
    monkeypatching modules cannot change a decision."""
    import posix

    _lstat, _readlink, _getcwd, _fspath = posix.lstat, posix.readlink, posix.getcwd, posix.fspath
    _type, _isinstance, _str, _bytes, _int = type, isinstance, str, bytes, int
    _str_copy, _decode = str.__str__, bytes.decode
    _OSError, _PermissionError, _RuntimeError, _ImportError, _TypeError = (
        OSError, PermissionError, RuntimeError, ImportError, TypeError)
    IFMT, IFLNK, IFDIR = 0o170000, 0o120000, 0o040000
    WFLAGS = posix.O_WRONLY | posix.O_RDWR | posix.O_CREAT | posix.O_TRUNC | posix.O_APPEND
    O_CREAT, O_WR = posix.O_CREAT, posix.O_WRONLY | posix.O_RDWR  # O_TMPFILE = create + write on a directory

    def as_str(x):
        t = _type(x)
        if t is _str:
            return x
        if t is _bytes:
            return _decode(x, "utf-8", "surrogateescape")
        y = _fspath(x)
        if _isinstance(y, _str):
            return _str_copy(y)
        if _isinstance(y, _bytes):
            return _decode(y, "utf-8", "surrogateescape")
        raise _TypeError("bad path")

    def resolve(p):
        if not p.startswith("/"):
            p = _getcwd() + "/" + p
        todo = p.split("/")
        todo.reverse()
        out = []
        hops = 0
        while todo:
            name = todo.pop()
            if name == "" or name == ".":
                continue
            if name == "..":
                if out:
                    out.pop()
                continue
            cur = "/" + "/".join(out + [name])
            try:
                st = _lstat(cur)
            except _OSError:
                out.append(name)
                continue
            if (st.st_mode & IFMT) == IFLNK:
                hops += 1
                if hops > 40:
                    raise _PermissionError("sandbox: symlink loop")
                target = _readlink(cur)
                if target.startswith("/"):
                    out = []
                more = target.split("/")
                more.reverse()
                todo.extend(more)
                continue
            out.append(name)
        return "/" + "/".join(out)

    def roots(ps):
        return tuple(resolve(p).rstrip("/") + "/" for p in ps)

    W = roots(write_roots)
    RW = roots(read_roots) + W + ("/dev/null/",)

    def check(path, write):
        p = resolve(as_str(path)) + "/"
        if not p.startswith(W if write else RW):
            raise _PermissionError(f"sandbox: access to {p[:-1]!r} is not allowed")
        return p[:-1]

    DENY = frozenset({
        "os.system", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile", "os.kill",
        "os.killpg", "signal.pthread_kill", "subprocess.Popen", "os.symlink", "os.link", "resource.setrlimit",
        "resource.prlimit", "gc.get_objects", "gc.get_referrers", "gc.get_referents", "code.__new__",
        "sys._current_frames", "webbrowser.open", "sqlite3.enable_load_extension",
        "sqlite3.load_extension", "pty.spawn",
    })
    DENY_PREFIX = ("ctypes.", "socket.", "urllib.", "http.", "ftplib.", "smtplib.", "poplib.", "imaplib.",
                   "nntplib.", "telnetlib.", "_winapi.", "winreg.", "msvcrt.")
    BLOCK_IMPORT = frozenset({"ctypes", "_ctypes", "_posixsubprocess", "_multiprocessing", "cffi", "_cffi_backend",
                              "_posixshmem", "_dbm", "_gdbm", "readline", "rlcompleter"})
    READ1 = frozenset({"os.listdir", "os.scandir", "os.listxattr", "os.getxattr"})
    WRITE1 = frozenset({"os.chmod", "os.chown", "os.mkdir", "os.remove", "os.rmdir", "os.truncate", "os.utime",
                        "os.setxattr", "os.removexattr", "shutil.rmtree", "os.mkfifo", "os.mknod", "os.chflags",
                        "os.lchflags", "tempfile.mkstemp", "tempfile.mkdtemp", "shutil.chown", "os.lchmod"})
    WRITE2 = frozenset({"os.rename", "os.replace", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
                        "shutil.copytree", "shutil.move"})

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            if path is None or _type(path) is _int:
                return
            # O_TMPFILE opens a *directory* to create an unnamed file in it (create + write intent);
            # a read-only directory open is the dir_fd-relative-escape vector we block
            tmpfile = _type(flags) is _int and (flags & O_CREAT) and (flags & O_WR)
            w = tmpfile or (_type(flags) is _int and flags & WFLAGS) or (
                _type(mode) is _str and ("w" in mode or "a" in mode or "x" in mode or "+" in mode))
            p = check(path, bool(w))
            try:
                st = _lstat(p)
            except _OSError:
                return
            if (st.st_mode & IFMT) == IFDIR and not tmpfile:  # O_TMPFILE into an already-writable dir is allowed
                raise _PermissionError("sandbox: opening directories is not allowed")
            return
        if event == "import":
            name = args[0]
            if _type(name) is _str and (name in BLOCK_IMPORT or name.partition(".")[0] in BLOCK_IMPORT):
                raise _ImportError(f"sandbox: import of {name!r} is not allowed")
            return
        if event in DENY or event.startswith(DENY_PREFIX):
            raise _RuntimeError(f"sandbox: {event} is not allowed")
        if event in READ1:
            if args and args[0] is not None and _type(args[0]) is not _int:
                check(args[0], False)
        elif event in WRITE1:
            if args and args[0] is not None and _type(args[0]) is not _int:
                check(args[0], True)
        elif event in WRITE2:
            check(args[0], True)
            check(args[1], True)
        elif event == "sqlite3.connect" and args:
            db = args[0]
            if _type(db) is _str and db != ":memory:" and not db.startswith("file::memory:"):
                check(db[5:].partition("?")[0] if db.startswith("file:") else db, True)

    sys.addaudithook(hook)
    _guard_sqlite()


def _guard_sqlite() -> None:
    """Set an authorizer on every new sqlite connection that denies ATTACH and DETACH (action 24
    also covers ``VACUUM INTO``), so a connection to an allowed database cannot reach another file.
    Best-effort defence for when the kernel sandbox is unavailable (with Landlock the file open is
    blocked anyway); no effect if sqlite3 is not importable."""
    try:
        import sqlite3
        import sqlite3.dbapi2 as _dbapi
    except Exception:
        return
    _deny = frozenset({24, 25})  # SQLITE_ATTACH, SQLITE_DETACH
    _set = sqlite3.Connection.set_authorizer
    _orig = sqlite3.connect

    def authorizer(action, a1, a2, dbname, source):
        return 1 if action in _deny else 0  # SQLITE_DENY / SQLITE_OK

    def connect(*args, **kwargs):
        con = _orig(*args, **kwargs)
        try:
            _set(con, authorizer)
        except Exception:
            pass
        return con

    sqlite3.connect = connect  # type: ignore[assignment]
    _dbapi.connect = connect  # type: ignore[assignment]


def exec_confined(cfg):
    """Confine this process, then replace it with the external program ``cfg["argv"]``. It keeps
    the confinement (filesystem, no network or ptrace, limits) and so does every process it starts."""
    confine(cfg["read"], cfg["write"], cfg.get("mem_mb"), cfg.get("cpu_s"), parent=cfg.get("parent"),
            no_fork=False, audit=False, require_kernel=cfg.get("require_kernel", True), allow_exec=True)
    os.execve(cfg["argv"][0], cfg["argv"], cfg["env"])


def run_confined(cfg, code):
    """Entry point of a single-process sandboxed run: confine, then execute ``code``."""
    confine(cfg["read"], cfg["write"], cfg.get("mem_mb"), cfg.get("cpu_s"),
            require_kernel=cfg.get("require_kernel", False))
    exec(compile(code, "<main>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})


# ============================================================================ values across the boundary
_DEPTH = 200


def encode(x, depth=0):
    """Tagged JSON for plain data; anything else becomes an opaque ``repr``."""
    if depth > _DEPTH:
        raise ValueError("value nested too deeply")
    t = type(x)
    if x is None or t is bool or t is str:
        return x
    if t is int:
        # very large ints would trip Python's decimal int<->str guard in json; hex is unaffected
        return x if x.bit_length() <= 4000 else {"$i": format(x, "x")}
    if t is float:
        return x if x - x == 0 else {"$f": repr(x)}  # nan / inf are not JSON
    d = depth + 1
    if t is list:
        return [encode(v, d) for v in x]
    if t is tuple:
        return {"$t": [encode(v, d) for v in x]}
    if t is dict:
        return {"$d": [[encode(k, d), encode(v, d)] for k, v in x.items()]}
    if t is set:
        return {"$s": [encode(v, d) for v in x]}
    if t is frozenset:
        return {"$fs": [encode(v, d) for v in x]}
    if t is bytes:
        return {"$b": x.hex()}
    if t is bytearray:
        return {"$ba": x.hex()}
    if t is complex:
        return {"$c": [encode(x.real, d), encode(x.imag, d)]}
    import numbers

    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, str):
        return str.__str__(x)
    if isinstance(x, (bytes, bytearray)):
        return encode(bytes(x), depth)
    if isinstance(x, list):
        return encode(list(x), depth)
    if isinstance(x, tuple):
        return encode(tuple(x), depth)
    if isinstance(x, dict):
        return encode(dict(x), depth)
    if isinstance(x, frozenset):
        return encode(frozenset(x), depth)
    if isinstance(x, set):
        return encode(set(x), depth)
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        return encode(float(x), depth)
    if isinstance(x, numbers.Complex):
        return encode(complex(x), depth)
    if hasattr(x, "tolist") and callable(x.tolist):  # numpy arrays and scalars
        return encode(x.tolist(), d)
    try:
        text = repr(x)
    except Exception:
        text = f"<{t.__name__}>"
    return {"$o": text[:300]}


class Opaque:
    """A value that could not cross the boundary (only its repr did). Equal only to itself."""

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


def decode(x, depth=0):
    """Inverse of :func:`encode`; builds only plain built-in values (strict: rejects anything else)."""
    if depth > _DEPTH:
        raise ValueError("value nested too deeply")
    t = type(x)
    if x is None or t is bool or t is str or t is int or t is float:
        return x
    d = depth + 1
    if t is list:
        return [decode(v, d) for v in x]
    if t is dict and len(x) == 1:
        (tag, v), = x.items()
        if tag == "$i" and type(v) is str:
            return int(v, 16)
        if tag == "$f" and type(v) is str and v in ("nan", "inf", "-inf"):
            return float(v)
        if tag == "$b" and type(v) is str:
            return bytes.fromhex(v)
        if tag == "$ba" and type(v) is str:
            return bytearray.fromhex(v)
        if tag == "$o" and type(v) is str:
            return Opaque(v)
        if type(v) is list:
            if tag == "$t":
                return tuple(decode(e, d) for e in v)
            if tag == "$s":
                return {decode(e, d) for e in v}
            if tag == "$fs":
                return frozenset(decode(e, d) for e in v)
            if tag == "$d" and all(type(e) is list and len(e) == 2 for e in v):
                return {decode(k, d): decode(val, d) for k, val in v}
            if tag == "$c" and len(v) == 2:
                re_, im = decode(v[0], d), decode(v[1], d)
                if type(re_) is float and type(im) is float:
                    return complex(re_, im)
    raise ValueError("malformed value from untrusted process")


# ============================================================================ framing
class _Lines:
    """Newline-framed reader on a raw fd, with an optional deadline and a size cap."""

    def __init__(self, fd, limit=64 << 20):
        self.fd, self.buf, self.scan, self.limit = fd, bytearray(), 0, limit

    def readline(self, deadline=None):
        import select
        import time

        while True:
            i = self.buf.find(b"\n", self.scan)
            if i >= 0:
                line = bytes(self.buf[:i])
                del self.buf[: i + 1]
                self.scan = 0
                return line
            self.scan = len(self.buf)
            wait = None
            if deadline is not None:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    raise TimeoutError("untrusted code timed out")
            if not select.select([self.fd], [], [], wait)[0]:
                raise TimeoutError("untrusted code timed out")
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                return None
            self.buf += chunk
            if len(self.buf) > self.limit:
                raise ValueError("message from untrusted process too large")


def _write_all(fd, data, deadline=None):
    import select
    import time

    view = memoryview(data)
    while view:
        if deadline is not None:
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([], [fd], [], wait)[1]:
                raise TimeoutError("untrusted code timed out")
        try:
            n = os.write(fd, view[: 1 << 16])
        except BlockingIOError:
            continue
        view = view[n:]


# ============================================================================ server (untrusted side)
def _listing(mod):
    names = getattr(mod, "__all__", None)
    if not (isinstance(names, (list, tuple)) and all(type(n) is str for n in names)):
        names = [n for n in vars(mod) if not n.startswith("_")]
    out = []
    for n in names:
        try:
            v = getattr(mod, n)
            out.append([n, "f"] if callable(v) else [n, "v", encode(v)])
        except Exception:
            continue
    return out


def serve(cfg):
    """Confined server: loads untrusted code and answers requests (fd 0 in, fd 1 out)."""
    import importlib
    import types

    rfd, wfd = os.dup(0), os.dup(1)
    nul = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(nul, fd)  # untrusted prints and input() never touch the protocol
    os.close(nul)
    confine(cfg["read"], [os.getcwd()], cfg.get("mem_mb"), cfg.get("cpu_s"), parent=cfg.get("parent"),
            require_kernel=cfg.get("require_kernel", False))
    sys.path[:0] = list(cfg.get("path") or [])
    lines = _Lines(rfd)

    def module(name):
        m = sys.modules.get(name)
        if m is None:
            raise ImportError(f"module {name!r} is not loaded")
        return m

    def handle(req):
        op = req["op"]
        if op == "load":
            mod = types.ModuleType(req["module"])
            mod.__file__ = f"<{req['module']}>"
            sys.modules[req["module"]] = mod
            exec(compile(req["code"], mod.__file__, "exec"), mod.__dict__)
            return None
        if op == "import":
            return _listing(importlib.import_module(req["module"]))
        if op == "lookup":
            v = getattr(module(req["module"]), req["name"])
            return ["f"] if callable(v) else ["v", encode(v)]
        if op == "call":
            fn = getattr(module(req["module"]), req["name"])
            return encode(fn(*decode(req["args"]), **decode(req["kwargs"])))
        if op in ("eval", "truth"):
            v = eval(compile(req["expr"], "<expr>", "eval"), vars(module(req["module"])))
            return bool(v) if op == "truth" else encode(v)
        raise ValueError(f"unknown op {op!r}")

    while True:
        line = lines.readline()
        if line is None:
            return
        try:
            out = {"ok": handle(json.loads(line))}
        except BaseException as e:  # SystemExit from untrusted code is an ordinary failure here
            try:
                text = str(e)[:1000]
            except BaseException:
                text = ""
            out = {"error": [type(e).__name__, text]}
        try:
            data = json.dumps(out, allow_nan=False)
        except BaseException:
            data = json.dumps({"error": ["ValueError", "result could not be transferred"]})
        _write_all(wfd, (data + "\n").encode())


# ============================================================================ harness side
class RemoteError(Exception):
    """An exception raised by untrusted code that has no safe built-in counterpart."""


_NO_MAP = frozenset({"StopIteration", "StopAsyncIteration"})


def _remote_exception(name, text):
    import builtins

    t = getattr(builtins, name, None) if name not in _NO_MAP else None
    if isinstance(t, type) and issubclass(t, Exception):
        try:
            return t(text)
        except Exception:
            pass
    return RemoteError(f"{name}: {text}")


class RemoteFunction:
    """Callable proxy: arguments and results cross as plain data."""

    def __init__(self, untrusted, module, name):
        self._u, self._module, self.__name__ = untrusted, module, name

    def __call__(self, *args, **kwargs):
        raw = self._u.request(op="call", module=self._module, name=self.__name__,
                              args=encode(list(args)), kwargs=encode(dict(kwargs)))
        return decode(raw)

    def __repr__(self):
        return f"<untrusted function {self._module}.{self.__name__}>"


class _Importer:
    """Meta-path hook: importing ``<package>.x`` in the harness yields proxies for the untrusted
    module (functions become :class:`RemoteFunction`, other values are copied)."""

    def __init__(self, untrusted, packages):
        import importlib.machinery  # before the hook is installed: find_spec must not import

        self.u, self.packages, self._spec = untrusted, tuple(packages), importlib.machinery.ModuleSpec

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] not in self.packages:
            return None
        return self._spec(fullname, self, is_package=True)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []
        entries = self.u.request(op="import", module=module.__name__)
        if type(entries) is not list:
            raise ImportError("malformed module listing")
        names = []
        for e in entries:
            if type(e) is not list or len(e) < 2 or type(e[0]) is not str:
                raise ImportError("malformed module listing")
            setattr(module, e[0], RemoteFunction(self.u, module.__name__, e[0]) if e[1] == "f" else decode(e[2]))
            names.append(e[0])
        module.__all__ = names


class Untrusted:
    """Harness-side handle on a confined server process that runs untrusted code.

    ``read``: extra readable paths; ``path``: directories put on the server's ``sys.path``
    (readable too); ``budget``: seconds for the whole session.
    """

    def __init__(self, read=(), path=(), budget=10.0, mem_mb=512, lib_source=None, require_kernel=True):
        import subprocess
        import tempfile
        import time

        self.deadline = time.monotonic() + budget
        self.dir = os.path.realpath(tempfile.mkdtemp(prefix="untrusted_"))
        self._importers = []
        cfg = {"read": default_read_roots(list(read) + list(path)), "path": [os.path.realpath(p) for p in path],
               "mem_mb": mem_mb, "cpu_s": int(budget) + 1, "parent": os.getpid(), "require_kernel": require_kernel}
        src = (lib_source or _lib_source()) + f"\nserve({cfg!r})\n"
        env = {"PATH": "/usr/bin:/bin", "HOME": self.dir, "TMPDIR": self.dir, "LANG": "C.UTF-8",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        if os.environ.get("OA_ALLOW_NO_KERNEL_SANDBOX") == "1":
            env["OA_ALLOW_NO_KERNEL_SANDBOX"] = "1"
        self.proc = subprocess.Popen([sys.executable, "-I", "-B", "-c", src], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.dir, env=env,
                                     close_fds=True)
        self._w = self.proc.stdin.fileno()
        os.set_blocking(self._w, False)
        self._r = _Lines(self.proc.stdout.fileno())

    def request(self, **msg):
        _write_all(self._w, (json.dumps(msg, allow_nan=False) + "\n").encode(), self.deadline)
        line = self._r.readline(self.deadline)
        if line is None:
            raise RemoteError("untrusted process exited")
        resp = json.loads(line)
        if type(resp) is not dict:
            raise RemoteError("malformed response")
        if "error" in resp:
            err = resp["error"]
            if type(err) is list and len(err) == 2 and all(type(e) is str for e in err):
                raise _remote_exception(err[0], err[1])
            raise RemoteError("malformed error")
        if "ok" not in resp:
            raise RemoteError("malformed response")
        return resp["ok"]

    def load(self, code, module="candidate"):
        """Execute untrusted source as a module in the server."""
        self.request(op="load", module=module, code=code)

    def function(self, name, module="candidate"):
        return RemoteFunction(self, module, name)

    def lookup(self, name, module="candidate"):
        r = self.request(op="lookup", module=module, name=name)
        if type(r) is list and r[:1] == ["f"]:
            return RemoteFunction(self, module, name)
        if type(r) is list and len(r) == 2 and r[0] == "v":
            return decode(r[1])
        raise RemoteError("malformed lookup")

    def eval(self, expr, module="candidate"):
        """Value of an untrusted expression evaluated in the module (copied as plain data)."""
        return decode(self.request(op="eval", module=module, expr=expr))

    def truth(self, expr, module="candidate"):
        """``bool(expr)`` as computed by the untrusted side (the only value that crosses)."""
        return self.request(op="truth", module=module, expr=expr) is True

    def importer(self, *packages):
        """Make ``import <package>...`` in the harness return proxies for untrusted modules."""
        imp = _Importer(self, packages)
        sys.meta_path.insert(0, imp)
        self._importers.append(imp)
        return imp

    def close(self):
        import shutil

        for imp in self._importers:
            if imp in sys.meta_path:
                sys.meta_path.remove(imp)
        try:
            self.proc.kill()
            self.proc.wait(5)
        except Exception:
            pass
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except Exception:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _lib_source():
    src = globals().get("_OA_LIB_SRC")
    if src is None:
        with open(__file__) as f:
            src = f.read()
    return src


def tag_big_ints(x):
    """Integers too long for a decimal JSON literal (the host keeps Python's digit guard) travel
    as ``{"$oa_int": hex}``; ``_exec.run_isolated`` turns them back into integers."""
    t = type(x)
    if t is int and x.bit_length() > 4000:
        return {"$oa_int": format(x, "x")}
    if t is list or t is tuple:
        return [tag_big_ints(v) for v in x]
    if t is dict:
        return {k: tag_big_ints(v) for k, v in x.items()}
    return x


def harness_main(code, read, path, budget, mem_mb, main):
    """Epilogue of generated harness scripts: prints ``OA-RESULT <nonce> <json>`` on success."""
    nonce = sys.stdin.readline().strip()
    try:
        with Untrusted(read=read, path=path, budget=budget, mem_mb=mem_mb) as u:
            if code is not None:
                u.load(code)
            text = json.dumps(tag_big_ints(main(u)))
    except BaseException as e:
        msg = f"{type(e).__name__}: {e}"
        sys.stdout.write("\nOA-ERROR " + json.dumps(msg[:2000]) + "\n")
        sys.stdout.flush()
        os._exit(3)
    sys.stdout.write(f"\nOA-RESULT {nonce} {text}\n")
    sys.stdout.flush()
    os._exit(0)
