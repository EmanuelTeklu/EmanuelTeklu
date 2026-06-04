"""Five realistic agent-trace regimes producing Context objects with a stable
context, a changing user tail, volatile surface noise, and invalidation events.

Regimes:
  coding    -- repeated repo files + dev tools, changing code questions
  legal     -- stable contract docs, changing clause questions
  research  -- stable papers, changing research questions
  multitool -- many tools with shuffled order + volatile call IDs
  memory    -- versioned user/project memory that occasionally updates
"""
from __future__ import annotations
import random
from .canonicalize import Context

# larger stable contexts -> prefill-dominated, realistic agent prompts (~1-2k tok)
_REPO = [("repo/util.py", "def load(p):\n    return open(p).read()\n" * 80),
         ("repo/model.py", "class Net:\n    def forward(self,x):\n        return x\n" * 80)]
_DEV_TOOLS = [{"name": "run_tests", "description": "run the test suite", "args": {"path": "str"}},
              {"name": "grep", "description": "search the repo", "args": {"pattern": "str"}},
              {"name": "edit", "description": "edit a file", "args": {"path": "str", "patch": "str"}}]
_CONTRACT = [("docs/msa.txt", "This Master Services Agreement governs all work orders. " * 120),
             ("docs/sla.txt", "Uptime shall be 99.9% measured monthly. " * 90)]
_LEGAL_TOOLS = [{"name": "cite", "description": "cite a clause", "args": {"section": "str"}}]
_PAPERS = [("papers/attention.txt", "Attention weights tokens by scaled dot products. " * 120),
           ("papers/scaling.txt", "Loss follows a power law in compute and data. " * 100)]
_RES_TOOLS = [{"name": "search", "description": "search papers", "args": {"q": "str"}},
              {"name": "summarize", "description": "summarize a doc", "args": {"id": "str"}}]
_MANY_TOOLS = [{"name": f"tool_{i}", "description": f"does task {i}", "args": {"x": "str"}}
               for i in range(8)]

SYS = "You are a careful assistant. Use tools only when needed and cite sources."


def _vol(rng):
    rid = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    ts = f"2026-0{rng.randint(1,9)}-{rng.randint(10,28)}T{rng.randint(10,23)}:00:00Z"
    return f"req_id={rid} ts={ts} "


def _regime_spec(name):
    return {
        "coding": (_DEV_TOOLS, _REPO, ["fix the failing test", "why is load slow?",
                                       "add a docstring", "refactor forward"]),
        "legal": (_LEGAL_TOOLS, _CONTRACT, ["what is the SLA uptime?",
                                            "who owns IP?", "termination terms?"]),
        "research": (_RES_TOOLS, _PAPERS, ["summarize attention", "what is the scaling law?",
                                           "compare the two papers"]),
        "multitool": (_MANY_TOOLS, _REPO, ["use tool 3", "chain tools 1 and 5",
                                           "what can you do?"]),
        "memory": (_RES_TOOLS, _PAPERS, ["recall my preference", "continue the project",
                                         "what did we decide?"]),
    }[name]


def generate(name, n_requests=120, turns=8, seed=0, mem_inval_p=0.12,
             shuffle_tools=True, volatile=True):
    rng = random.Random(hash((name, seed)) % (2**31))
    tools0, docs, questions = _regime_spec(name)
    reqs = []
    sessions = max(1, n_requests // turns)
    for s in range(sessions):
        tools = [dict(t) for t in tools0]
        ds = [(f"{ref}", c) for ref, c in docs]
        memory = [{"block_id": "profile", "version": 1, "content": "prefers concise answers"}]
        for t in range(turns):
            if len(reqs) >= n_requests:
                break
            tl = [dict(x) for x in tools]
            if shuffle_tools:
                rng.shuffle(tl)                              # surface reorder (should-hit)
            dd = ([(f"{ref}?ts={rng.randint(1,9)}", c) for ref, c in ds]
                  if volatile else list(ds))                # doc metadata noise (should-hit)
            mem_event = (name == "memory" or True) and rng.random() < mem_inval_p and t > 0
            if mem_event:
                memory = [dict(memory[0], version=memory[0]["version"] + 1,
                               content=memory[0]["content"] + "; new note")]
            sys = (_vol(rng) if volatile else "") + SYS     # volatile header (should-hit)
            ctx = Context(system=sys, tools=tl, docs=dd, memory=[dict(memory[0])],
                          user=f"{name} q{s}-{t}: " + rng.choice(questions))
            reqs.append((s, t, ctx, int(mem_event)))
    return reqs


REGIMES = ["coding", "legal", "research", "multitool", "memory"]
