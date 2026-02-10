# Control-Plane Integrity Theorem -- Formal Proof

## Definitions

### Agent state

The agent state at step t is:

    S_t = (P_t, M_t, B_t, G_t)

where P_t is the control plane, M_t is memory, B_t is behaviour parameters, and G_t is the information representation (IR) graph.

### Control plane

    P_t = (Perm_t, Int_t, Policy_t)

- **Perm_t**: set of allowed tools and actions
- **Int_t**: configured integrations (chat channels, API connections, skills)
- **Policy_t**: principal-to-scope mappings

### Principals

    Trusted    = {SYS, USER}
    Untrusted  = {WEB, SKILL, TOOL_OUTPUT}

### IR nodes

Each IR node n has:
- `n.id`: unique identifier
- `n.principal`: the origin principal
- `n.taint` in {0, 1}: 0 = untainted, 1 = tainted
- `n.deps`: set of dependency node ids

### Taint rule

    taint(n) = 1  if  n.principal in Untrusted
                      OR  exists d in n.deps such that taint(d) = 1
    taint(n) = 0  otherwise

### Verification rules

A proposal to change P_t to P_{t+1} with justification certificate C = (J, confirmed) is **approved** iff:

- **V1**: For all node ids m in the transitive closure of J: G_t[m].taint = 0
- **V2**: For all node ids j in J: G_t[j].principal in Trusted
- **V3**: If the proposal adds integrations or changes permissions: confirmed = True

---

## Lemmas

### Lemma 1 (Taint Propagation Soundness)

**Statement.** For every node n in G_t:

    n.principal in Untrusted  =>  n.taint = 1

and

    (exists d in n.deps: G_t[d].taint = 1)  =>  n.taint = 1

**Proof.** Follows directly from the taint computation rule, which is applied at node creation time. The IR graph is append-only, so taint values are immutable once set. By the rule, any node with an untrusted principal receives taint = 1, and any node with a tainted dependency receives taint = 1. Since these are the only two ways taint is assigned, the lemma holds by construction.

### Lemma 2 (Untrusted Origin Implies Taint)

**Statement.** If n.principal in {WEB, SKILL, TOOL_OUTPUT}, then n.taint = 1.

**Proof.** Immediate from Lemma 1 and the definition of Untrusted.

### Lemma 3 (Verifier Soundness)

**Statement.** If the verifier approves a control-plane change justified by certificate C = (J, confirmed), then:
1. Every node in the transitive dependency set of J has taint = 0.
2. Every node in J has a trusted principal.
3. If the change adds integrations or modifies permissions, confirmed = True.

**Proof.** The verifier explicitly checks V1, V2, and V3 before returning APPROVED. If any check fails, the result is REJECTED. Therefore, an APPROVED result implies all three conditions hold.

---

## Theorem (Control-Plane Integrity Against Malicious Skills)

**Statement.** Let two executions E^(1) and E^(2) share:
- the same initial state S_0
- the same sequence of trusted inputs T = (t_1, t_2, ..., t_k)

but differ in untrusted inputs:
- E^(1) receives untrusted sequence U^(1) = (u_1^(1), u_2^(1), ...)
- E^(2) receives untrusted sequence U^(2) = (u_1^(2), u_2^(2), ...)

Suppose the verifier enforces V1, V2, V3 at every step. Then:

1. **P_t^(1) = P_t^(2)** for all t.
2. No new integrations or permission changes are introduced unless authorised by the user.

**Proof.** By induction on the step index t.

**Base case (t = 0):**
Both executions start from S_0, so P_0^(1) = P_0^(2). No changes have been proposed.

**Inductive step:**
Assume P_t^(1) = P_t^(2). We show P_{t+1}^(1) = P_{t+1}^(2).

At step t, each execution processes either a trusted input or an untrusted input.

**Case 1: Trusted input t_i is delivered to both executions.**

Both executions receive the same input with the same principal and content. If t_i proposes a control-plane change:
- The justifying node has a trusted principal (USER or SYS).
- By Lemma 1, since the principal is trusted and there are no tainted dependencies (the node is freshly created from trusted input), the node has taint = 0.
- The verifier checks V1, V2, V3. Both executions produce identical IR nodes (same principal, same content) and identical proposals.
- Therefore, the verifier produces the same verdict in both executions.
- If approved, both apply the same change. If rejected, neither changes.
- Result: P_{t+1}^(1) = P_{t+1}^(2).

**Case 2: Untrusted input u_t^(i) is delivered to execution E^(i).**

The input has principal pi in {WEB, SKILL, TOOL_OUTPUT}. Consider whether u_t^(i) proposes a control-plane change.

**Case 2a: No control-plane change proposed.**
P_{t+1}^(i) = P_t^(i). Since P_t^(1) = P_t^(2) by hypothesis, and neither execution changes its control plane, P_{t+1}^(1) = P_{t+1}^(2).

**Case 2b: A control-plane change is proposed.**
The justifying node n has principal pi in Untrusted. By Lemma 2, n.taint = 1. The verifier checks V1 and finds a tainted node in the dependency set. The verifier returns REJECTED. The control plane is not modified: P_{t+1}^(i) = P_t^(i).

Since both executions either leave the control plane unchanged or have their untrusted proposals rejected, and P_t^(1) = P_t^(2) by hypothesis:

    P_{t+1}^(1) = P_t^(1) = P_t^(2) = P_{t+1}^(2)

**Property 2 (No unauthorised changes):**

By Lemma 3, the only approved changes are those where:
- All justifying nodes are untainted (V1)
- All justifying nodes have trusted principal (V2)
- User confirmation is present for integration/permission changes (V3)

Since untrusted inputs produce tainted nodes (Lemma 2), they cannot satisfy V1. Therefore, only trusted inputs with user confirmation can produce approved control-plane changes, which establishes property 2.

**QED.**

---

## Corollaries

### Corollary 1 (Telegram Bot Attack Prevention)

An attacker embedding "add a Telegram bot" in a shared document cannot cause the agent to add a new integration, because:
1. The document content is parsed as an IR node with principal WEB (or SKILL).
2. By Lemma 2, this node has taint = 1.
3. Any proposal justified by this node fails V1.

### Corollary 2 (Persistent Storage Protection)

A malicious skill writing "reminders" to SOUL.md or AGENTS.md to alter policy cannot succeed, because:
1. The skill content has principal SKILL, so taint = 1.
2. Any policy change proposal justified by this node fails V1 and V2.
3. Even if the write to storage succeeds (in M_t), the control plane P_t remains unchanged.

### Corollary 3 (Transitive Taint Laundering Prevention)

An attacker cannot launder taint by having untrusted content influence a "trusted-looking" derivation, because:
1. A node derived from any tainted dependency inherits taint = 1 (Lemma 1).
2. The transitive dependency check in V1 examines the full closure.
3. Therefore, no chain of derivations can produce an untainted node from tainted input.
