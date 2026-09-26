"""Landlock launcher: confine this process with a kernel file-system allow-list, then exec the command.

Run as ``python -I -S -c <this file's source> <config JSON> -- argv...`` (:func:`so_arena.core.sandbox.wrap`);
it is stdlib-only and never imports the package, which the command must not see. It applies, irreversibly and
inherited by everything the command starts:

1. ``PR_SET_NO_NEW_PRIVS`` (no privilege gain through set-uid programs);
2. a Landlock ruleset (Linux >= 5.13): reads only beneath ``read`` roots, writes only beneath ``write`` and
   ``scratch`` roots, except the ``holes`` inside ``read`` and ``write`` roots - Landlock rules only add access, so a root holding a hole is granted
   entry by entry around it (symbolic links are skipped: a rule would grant their target). With ABI >= 4 and
   no ``network``, TCP bind and connect are denied; with ABI >= 6, signals and abstract Unix sockets are scoped
   to the sandbox;
3. with ``seccomp`` (x86-64 and arm64): no sockets (unless ``network``), no ptrace or cross-process memory
   access, no new namespaces, io_uring, BPF, perf events, keyrings, userfaultfd or mounts - the layer that
   stands in for the network namespace and the private process view where there are no namespaces.

Anything that fails to apply fails closed: the launcher exits with status 126 and a message on stderr, and the
command never runs. After confinement it checks that Landlock enforces (the root directory must be unreadable).
"""

import json
import os
import stat
import struct
import sys

REFUSED = 126
SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446  # the same on every architecture Landlock supports
# file-system access rights (landlock.h) and the ABI version that introduced the later ones
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV  # the rights a rule on a non-directory may hold
READ = EXECUTE | READ_FILE | READ_DIR

# (audit arch, syscalls denied with EPERM, the socket syscall) per machine
_SECCOMP = {
    "x86_64": (0xC000003E, [101, 310, 311, 312, 425, 426, 427, 321, 298, 248, 249, 250, 272, 308, 323, 165, 166, 155], 41),
    "aarch64": (0xC00000B7, [117, 270, 271, 272, 425, 426, 427, 280, 241, 217, 218, 219, 97, 268, 282, 40, 39, 41], 198),
}


def _libc():
    import ctypes

    return ctypes, ctypes.CDLL(None, use_errno=True)


def landlock_abi() -> int:
    """The Landlock ABI version the kernel supports (0: unavailable - not Linux, too old, or not enabled)."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        ctypes, libc = _libc()
        v = libc.syscall(ctypes.c_long(SYS_CREATE), None, ctypes.c_long(0), ctypes.c_long(1))  # LANDLOCK_CREATE_RULESET_VERSION
        return int(v) if v > 0 else 0
    except (OSError, AttributeError):
        return 0


def seccomp_supported() -> bool:
    return sys.platform.startswith("linux") and os.uname().machine in _SECCOMP


def _refuse(msg: str) -> None:
    sys.stderr.write(f"so-arena sandbox: {msg}; refusing to run the command\n")
    sys.stderr.flush()
    os._exit(REFUSED)


def _handled_fs(abi: int) -> int:
    rights = (1 << 13) - 1  # ABI 1: EXECUTE ... MAKE_SYM
    if abi >= 2:
        rights |= REFER
    if abi >= 3:
        rights |= TRUNCATE
    if abi >= 5:
        rights |= IOCTL_DEV
    return rights


def _landlock(cfg: dict, abi: int) -> None:
    ctypes, libc = _libc()
    L = ctypes.c_long
    handled = _handled_fs(abi)
    net = 0 if cfg.get("network") or abi < 4 else 1 | 2  # BIND_TCP | CONNECT_TCP, with no rule allowing any port
    scoped = 1 | 2 if abi >= 6 else 0  # ABSTRACT_UNIX_SOCKET | SIGNAL
    attr = struct.pack("=QQQ", handled, net, scoped)[: 8 if abi < 4 else 16 if abi < 6 else 24]
    buf = ctypes.create_string_buffer(attr, len(attr))
    rs = libc.syscall(L(SYS_CREATE), buf, L(len(attr)), L(0))
    if rs < 0:
        _refuse(f"cannot create a Landlock ruleset ({os.strerror(ctypes.get_errno())})")
    holes = sorted({os.path.normpath(h) for h in cfg.get("holes", ())})

    def add(path: str, access: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return  # nothing there to grant
        try:
            acc = access & handled
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                acc &= FILE_RIGHTS
            if not acc:
                return
            rule = ctypes.create_string_buffer(struct.pack("=Qi", acc, fd), 12)
            if libc.syscall(L(SYS_ADD_RULE), L(rs), L(1), rule, L(0)) != 0:  # LANDLOCK_RULE_PATH_BENEATH
                _refuse(f"cannot add a Landlock rule for {path} ({os.strerror(ctypes.get_errno())})")
        finally:
            os.close(fd)

    def grant(path: str, access: int) -> None:
        """``access`` beneath ``path`` except beneath the holes: around a hole, entry by entry."""
        path = os.path.normpath(path)
        if path in holes:
            return
        prefix = path.rstrip("/") + "/"
        if not any(h.startswith(prefix) for h in holes):
            add(path, access)
            return
        try:
            entries = list(os.scandir(path))
        except OSError:
            return
        for e in entries:
            if not e.is_symlink():
                grant(e.path, access)

    for p in cfg.get("read", ()):
        grant(p, READ)
    for p, mode in cfg.get("files", ()):  # single files, e.g. devices: "r" or "rw"
        add(p, READ_FILE | WRITE_FILE | TRUNCATE if mode == "rw" else READ_FILE)
    for p in cfg.get("write", ()):
        grant(p, handled)
    for p in cfg.get("scratch", ()):  # a sandbox's own fresh file systems (under namespaces): wholly writable
        add(p, handled)
    if cfg.get("proc_self"):
        add(f"/proc/{os.getpid()}", READ)  # its own entry only (the command keeps this pid): no other process's
    if libc.prctl(L(38), L(1), L(0), L(0), L(0)) != 0:  # PR_SET_NO_NEW_PRIVS
        _refuse("cannot set no_new_privs")
    if libc.syscall(L(SYS_RESTRICT), L(rs), L(0)) != 0:
        _refuse(f"cannot enforce the Landlock ruleset ({os.strerror(ctypes.get_errno())})")
    os.close(rs)


def _seccomp(cfg: dict) -> None:
    ctypes, libc = _libc()
    L = ctypes.c_long
    arch, deny, sock = _SECCOMP[os.uname().machine]
    deny = deny + ([] if cfg.get("network") else [sock])
    LD, JEQ, JSET, RET = 0x20, 0x15, 0x45, 0x06
    ALLOW, EPERM, KILL = 0x7FFF0000, 0x00050001, 0x80000000
    prog = []

    def op(code: int, jt: int, jf: int, k: int) -> None:
        prog.append(struct.pack("=HBBI", code, jt, jf, k))

    op(LD, 0, 0, 4)  # seccomp_data.arch
    op(JEQ, 1, 0, arch)
    op(RET, 0, 0, KILL)  # a foreign system-call ABI (e.g. int 0x80): killed
    op(LD, 0, 0, 0)  # seccomp_data.nr
    if arch == 0xC000003E:
        op(JSET, 0, 1, 0x40000000)  # the x32 ABI
        op(RET, 0, 0, EPERM)
    for n in deny:
        op(JEQ, 0, 1, n)
        op(RET, 0, 0, EPERM)
    op(RET, 0, 0, ALLOW)
    code = b"".join(prog)
    buf = ctypes.create_string_buffer(code, len(code))

    class Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fp = Fprog(len(prog), ctypes.addressof(buf))
    if libc.prctl(L(38), L(1), L(0), L(0), L(0)) != 0 or libc.prctl(L(22), L(2), ctypes.byref(fp), L(0), L(0)) != 0:
        _refuse(f"cannot install the seccomp filter ({os.strerror(ctypes.get_errno())})")


def main(args: list) -> None:
    if len(args) < 3 or args[1] != "--":
        _refuse("usage: <config JSON> -- argv...")
    cfg, argv = json.loads(args[0]), args[2:]
    abi = landlock_abi()
    if abi < max(1, int(cfg.get("min_abi", 1))):
        _refuse(f"Landlock ABI {cfg.get('min_abi', 1)} or later is needed, the kernel has {abi or 'none'}")
    if cfg.get("seccomp") and not seccomp_supported():
        _refuse(f"no seccomp filter for machine {os.uname().machine}")
    _landlock(cfg, abi)
    if cfg.get("seccomp"):
        _seccomp(cfg)
    if "/" not in cfg.get("read", ()) and "/" not in cfg.get("write", ()):
        try:
            os.listdir("/")
            _refuse("Landlock is not enforcing (the root directory is readable)")
        except PermissionError:
            pass
    try:
        if cfg.get("chdir"):
            os.chdir(cfg["chdir"])
        os.execvp(argv[0], argv)
    except OSError as e:
        sys.stderr.write(f"so-arena sandbox: cannot run {argv[0]!r}: {e}\n")
        os._exit(127)


if __name__ == "__main__":
    main(sys.argv[1:])
