"""The sandbox's MAX_OPERATIONS budget is per invocation of a sandbox-defined function / lambda, not shared
across every call a tool makes: a `map_fn` invoked once per chunk of a large collection must not exhaust the
step's budget part-way through (it used to fail every chunk after ~255k of a 545k-chunk map)."""

from __future__ import annotations

import pytest

from skunk.sandbox.local_python_executor import MAX_OPERATIONS, InterpreterError, LocalPythonExecutor

# a callback whose body costs a few hundred operations; 30k calls is ~8x MAX_OPERATIONS in total
_LAMBDA = 'fn = lambda text, meta: {"n": len([c for c in text if c == "a"]), "year": int(meta["doc_id"].split("_")[0])}'
_DEF = '''
def fn(text, meta):
    n = 0
    for c in text:
        if c == "a":
            n += 1
    return {"n": n, "year": int(meta["doc_id"].split("_")[0])}
'''
_TEXT = "a" * 200


def _executor() -> LocalPythonExecutor:
    ex = LocalPythonExecutor(additional_authorized_imports=[])
    ex.send_tools({})  # installs the base python tools (len, range, sum, ...), as the agent does
    return ex


# the callbacks' body, evaluated inline (no function call) to measure what one invocation costs
_BODY = 'n = len([c for c in text if c == "a"])\nyear = int(meta["doc_id"].split("_")[0])'


def _ops_of_body() -> int:
    ex = _executor()
    ex.send_variables({"text": _TEXT, "meta": {"doc_id": "1939_01_1"}})
    ex(_BODY)
    return ex.state["_operations_count"]["counter"]


@pytest.mark.parametrize("definition", [_LAMBDA, _DEF], ids=["lambda", "def"])
def test_each_native_call_gets_a_fresh_budget(definition):
    ex = _executor()
    fn = ex(definition + "\nfn").output
    # comfortably more total work than MAX_OPERATIONS, spread over many calls (the map tool's usage pattern)
    n_calls = 30_000
    assert _ops_of_body() * n_calls > 2 * MAX_OPERATIONS  # would have failed part-way under the shared counter
    before = ex.state["_operations_count"]["counter"]
    for i in range(n_calls):
        assert fn(_TEXT, {"doc_id": f"{1939 + i % 80}_01_{i}"}) == {"n": 200, "year": 1939 + i % 80}
    # and the calls did not drain the step's own counter
    assert ex.state["_operations_count"]["counter"] == before


def test_runaway_work_inside_one_call_is_still_caught():
    ex = _executor()
    fn = ex("fn = lambda n: sum([i for i in range(n)])\nfn").output
    with pytest.raises(InterpreterError, match="max number of operations"):
        fn(MAX_OPERATIONS)


def test_runaway_loop_calling_a_helper_is_still_caught():
    # the loop's own per-iteration work is still charged to the step, whatever the helper costs
    ex = _executor()
    with pytest.raises(InterpreterError, match="max number of operations"):
        ex("def f(i):\n    return i\nfor i in range(20000000):\n    x = [i, i, i, i, i, i, i, i]\n    f(i)\n")


def test_nested_helper_calls_do_not_drain_the_outer_budget():
    ex = _executor()
    code = '''
def inner(x):
    return sum([i for i in range(x)])
def outer(k):
    return [inner(1000) for _ in range(k)]
outer(2000)
'''
    out = ex(code)
    assert out.output == [499500] * 2000
