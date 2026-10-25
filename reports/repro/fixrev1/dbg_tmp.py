from oversight_arena.domains._exec import run_isolated, run_python
code = "import tempfile, os\ndef f(n):\n    with tempfile.NamedTemporaryFile('w', delete=False) as fh: fh.write('x')\n    return 1\ndef g(n):\n    open('scratch.txt','w').write('x'); return open('scratch.txt').read()\n"
print(run_isolated("def main(u):\n    return u.function('f')(1)\n", code=code))
print(run_isolated("def main(u):\n    return u.function('g')(1)\n", code=code))
print(run_python(code + "\nprint(f(1), g(1))\n").short())
