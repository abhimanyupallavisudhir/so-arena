"""Reproduce selected observations at the commits in task-1-comparison.md.

Run with the review environment's Python (dependencies from attempts 1 and 2):
    python reviews/review_probes.py /path/a1 /path/a2 /path/a3 /path/a4

Each probe imports one checkout in a separate process. No API calls are made.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


PROBES = {
    1: """
        from oversight_arena.elicitation.optimize import Candidate, OptimizationTrace
        from oversight_arena.core.strategy import Strategy
        from oversight_arena.domains.code import passes_hidden

        candidates = [
            Candidate(id=str(r), strategy=Strategy(name=str(r)), iteration=0,
                      parent=None, reward=r, reward_se=0, gt={}, per_task={}, n=1)
            for r in [-1.0, 0.0]
        ]
        chosen = [c.reward for c in OptimizationTrace(candidates).best()]
        print('Best of rewards [-1, 0]:', chosen, '(correct selection: [0])')
        accepted = passes_hidden(
            "print('PASSED'); raise SystemExit(0)",
            'def check(f):\\n    assert f(2) == 4',
            'double',
        )
        print('Program defining no double function passes hidden test:', accepted)
        assert chosen == [-1.0] and accepted is True
    """,
    2: """
        import numpy as np
        import pandas as pd
        from so_arena.analysis.optimization import (
            GameTree, TreeNode, TreeLeaf, BestOfN, Uniform, evaluate_tree, pool_curve,
        )

        nodes = {
            'root': TreeNode(id='root', key='coin', role='coin', children=['g0', 'g1']),
            'g0': TreeNode(id='g0', key='same-hidden-information', role='guesser',
                           children=['00', '01']),
            'g1': TreeNode(id='g1', key='same-hidden-information', role='guesser',
                           children=['10', '11']),
        }
        leaves = {
            str(a) + str(b): TreeLeaf(id=str(a) + str(b),
                                     rewards={'guesser': float(a == b)})
            for a in [0, 1] for b in [0, 1]
        }
        tree = GameTree(item_id='hidden-coin', mechanism='guess', root='root',
                        nodes=nodes, leaves=leaves)
        value = evaluate_tree(tree, {'guesser': BestOfN(2)}).rewards['guesser']
        print('Blind fair-coin guessing: maximum feasible 0.5; tree reports', value)
        pool = pd.DataFrame({'item_id': ['x', 'x'], 'reward': [0., 1.],
                             'value': [1., np.nan]})
        row = pool_curve(pool, selections=[Uniform()]).to_dict('records')[0]
        print('Quality with half the selected mass unlabelled:', row)
        assert value == 1.0 and row['value'] == 1.0
    """,
    4: """
        import asyncio
        from oversight_arena import Action, Role, Task, run
        from oversight_arena.mechanisms import Debate

        async def main():
            counts = []
            for simultaneous in (False, True):
                calls = []

                async def tool(arguments):
                    calls.append(arguments)
                    return {'ok': True}

                async def speaker(obs):
                    if any(e.kind == 'tool' and e.actor == obs.role for e in obs.events):
                        return Action('Evidence checked')
                    return Action(data={'tool_calls': [{'name': 'check', 'arguments': {}}]})

                async def judge(obs):
                    return Action(data={'scores': {'proposer': .5, 'critic': .5}})

                record = await run(
                    Task('t', 'Test'),
                    (Role('proposer', tools=('check',)), Role('critic', tools=('check',)),
                     Role('judge', trainable=False)),
                    {'proposer': speaker, 'critic': speaker, 'judge': judge},
                    Debate(rounds=1, simultaneous=simultaneous), name='debate',
                    tools={'check': tool},
                )
                print('Simultaneous:', simultaneous, 'status:', record.status,
                      'executed tools:', len(calls))
                assert record.status == 'complete', record.error
                counts.append(len(calls))
            assert counts == [2, 0]

        asyncio.run(main())
    """,
}


def main():
    if len(sys.argv) != 5:
        raise SystemExit('Usage: review_probes.py ATTEMPT1 ATTEMPT2 ATTEMPT3 ATTEMPT4')
    roots = [Path(arg).resolve() for arg in sys.argv[1:]]
    for attempt, program in PROBES.items():
        root = roots[attempt - 1]
        print(f'Attempt {attempt}', flush=True)
        subprocess.run(
            [sys.executable, '-c', textwrap.dedent(program)],
            cwd=root,
            env={**os.environ, 'PYTHONPATH': str(root / 'src')},
            check=True,
            timeout=60,
        )


if __name__ == '__main__':
    main()
