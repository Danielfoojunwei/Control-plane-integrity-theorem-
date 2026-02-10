# Control-Plane Integrity Theorem

A formal model and mechanised proof that untrusted content (malicious skills, poisoned tool outputs, adversarial web pages) cannot modify the control plane of an RLM-based agent without explicit user authorisation.

## Theorem statement

**Control-Plane Integrity Against Malicious Skills.** Let two executions share the same initial state S\_0 and identical sequences of trusted inputs. Suppose the verifier enforces that any update to the control plane must be justified solely by untainted IR nodes whose provenance is SYS or confirmed USER. Then, for any sequences of untrusted skill content or tool output:

1. The control-plane state remains equal at all steps: P\_t^(1) = P\_t^(2).
2. No new integrations or permission changes are introduced unless authorised by the user.

## Architecture

```
src/
  principals.py   -- Principal hierarchy (SYS, USER, WEB, SKILL, TOOL_OUTPUT)
  ir.py           -- Information Representation graph with taint propagation
  state.py        -- Agent state model: S_t = (P_t, M_t, B_t, G_t)
  verifier.py     -- Three-phase control-plane update verifier
  agent.py        -- Stepwise agent execution engine
  theorem.py      -- Formal theorem statement, lemmas, and mechanised proof

tests/
  test_principals.py  -- Principal trust classification tests
  test_ir.py          -- IR graph and taint propagation tests
  test_verifier.py    -- Verifier rule enforcement tests
  test_agent.py       -- Agent execution engine tests
  test_theorem.py     -- Theorem proof and all five evaluation scenarios
```

## Model

The agent state at time t is the tuple **S\_t = (P\_t, M\_t, B\_t, G\_t)** where:

| Component | Description |
|-----------|-------------|
| **P\_t** (Control plane) | Permissions, integrations, and policies |
| **M\_t** (Memory) | Persistent storage |
| **B\_t** (Behaviour) | System prompt extensions and reminders |
| **G\_t** (IR graph) | Information-flow graph with taint tracking |

### Principals

| Principal | Trust | Description |
|-----------|-------|-------------|
| SYS | Trusted | System / platform policies |
| USER | Trusted | Direct, confirmed user input |
| WEB | Untrusted | Scraped web pages, emails |
| SKILL | Untrusted | Installed skill file content |
| TOOL\_OUTPUT | Untrusted | External tool return values |

### Verification rules

1. **V1 (Taint-free):** All nodes in the transitive dependency set must have taint = 0.
2. **V2 (Trusted provenance):** Every justifying node must have SYS or USER principal.
3. **V3 (User confirmation):** Integration or permission changes require explicit user confirmation.

## Evaluation scenarios

The test suite implements five evaluation scenarios:

1. **Malicious skill injection** -- A skill attempts to add a Telegram bot integration. Blocked.
2. **Tool output poisoning** -- A tool returns "grant admin permissions". Blocked.
3. **Supply-chain malware** -- Skills attempt to write to SOUL.md/AGENTS.md. Blocked.
4. **False-positive rate** -- Legitimate user-initiated changes with confirmation. Allowed.
5. **Policy coverage** -- Sequential multi-vector attacks. All blocked; control plane unchanged.

## Running

```bash
python -m pytest tests/ -v
```

## License

Apache 2.0
