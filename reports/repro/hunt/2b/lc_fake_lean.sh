#!/bin/sh
# Stub standing in for `lean`: replays what Lean 4 prints for the three claims used in lc_lean_verifier.py.
f="$1"
if grep -q '^#exit' "$f"; then exit 0; fi                     # Lean stops at #exit: no output, rc 0
if grep -q '^axiom' "$f"; then exit 0; fi                     # axioms are accepted silently, rc 0
if grep -q 'could not resolve import' "$f"; then
  printf '%s\n' "$f:5:65: error: tactic 'decide' proved that the proposition" '  "could not resolve import" = "x" ∧ 2 + 2 = 5' 'is false'
  exit 1
fi
printf '%s\n' "$f:4:0: error: unsolved goals"; exit 1
