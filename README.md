# Control-Plane Integrity for LLM Agents: A Provenance-Based Defense Against Malicious Skills and Tool Output Poisoning

> **Abstract.** Large language model (LLM) agents that integrate external skills, tools, and data sources face a fundamental security challenge: untrusted content from these sources can manipulate the agent into modifying its own control plane---adding unauthorized integrations, escalating permissions, or rewriting policies. We present a formal information-flow framework that prevents such attacks by reducing control-plane integrity to the correctness of a *provenance tracking infrastructure*. Our main result is a **Control-Plane Integrity Theorem**: a reference-monitor argument showing that if (1) every IR node records its true data sources, (2) principals are assigned from transport channels (not content), and (3) memory operations preserve provenance, then no untrusted input can modify the agent's control plane. The theorem is *conditional*---it explicitly identifies three assumptions whose violation breaks the guarantee entirely---and we provide concrete implementations that enforce two of three. We evaluate with Qwen2.5-3B-Instruct against 216 mechanism-level attacks and 36 end-to-end LLM attacks. The unprotected model follows attack instructions 77.8% of the time; the verifier blocks 100% of attempts *when assumptions hold*. Adversarial analysis of 8 deployment scenarios shows 3/6 assumption violations bypass the verifier completely, while our `ChannelPrincipalAssigner` and `ProvenantMemoryStore` close 2 of 3 identified bypass paths.

---

## 1. Introduction

LLM-based agents increasingly operate in environments where they process content from multiple sources of varying trust: user instructions, system prompts, web pages, installed skills, and tool return values. This creates an attack surface where *indirect prompt injection* (Greshake et al., 2023), *malicious skills* (Zhan et al., 2024), and *tool output poisoning* (Zhang et al., 2024b) can manipulate the agent into taking unauthorized actions.

We focus on attacks that target the agent's **control plane**---the configuration layer that determines what integrations are active, what permissions are granted, and what policies govern behavior. Unlike data-plane attacks (which affect individual responses), control-plane attacks are *persistent*: a single successful attack can permanently alter capabilities.

### 1.1 Scope and Non-Goals

**What we protect:** The control plane $P_t = (\text{Perm}_t, \text{Int}_t, \text{Policy}_t)$---permissions (which tools are authorized), integrations (which external services are connected), and policies (which principals have which scopes). A control-plane change is any mutation to these three sets.

**What we do NOT protect:**
- **Data-plane attacks:** An attacker causing the agent to produce a wrong answer, leak information in a response, or take an unauthorized action that does not modify the control plane. These are orthogonal and require separate defenses (output filtering, data-loss prevention).
- **Attacks through trusted channels:** If a user account is compromised, the attacker's input arrives through `USER` and our framework cannot distinguish it from legitimate input. We assume the authentication layer is correct.
- **Semantic attacks on the LLM:** Jailbreaks, prompt injection that affects LLM reasoning without tool calls, or attacks that exploit the LLM's content understanding. Our verifier is content-agnostic by design.

### 1.2 Central Claim (Precisely Stated)

Our theorem is a **reduction**: it reduces control-plane integrity to three deployment properties (provenance completeness, principal accuracy, memory persistence). This is analogous to how a reference monitor reduces system security to correct mediation of access requests (Anderson, 1972). The theorem does *not* claim to solve the harder problems of provenance tracking or principal assignment---it proves that *if* these are correct, integrity follows.

Concretely:
- **Claim 1 (Conditional guarantee):** Under assumptions A1--A3, no untrusted input can modify $P_t$. This is a mechanism property, not an empirical finding. The 0% protected ASR is a *theorem consequence*, not a measurement.
- **Claim 2 (Assumption fragility):** When any of A1--A3 is violated, the guarantee breaks completely (100% bypass rate for 3/6 tested scenarios).
- **Claim 3 (Constructive mitigations):** We provide `ChannelPrincipalAssigner` (enforcing A2) and `ProvenantMemoryStore` (enforcing A3) as concrete implementations.

### 1.3 Contributions

1. A formal agent state model $S_t = (P_t, M_t, B_t, G_t)$ with provenance-based taint tracking and explicit control-plane boundaries (Section 3).
2. The **Control-Plane Integrity Theorem**: a reference-monitor reduction from CP integrity to provenance correctness, with proof by induction (Section 4).
3. **`ChannelPrincipalAssigner`**: enforces "principal from channel, not content" to eliminate confused deputy attacks (Section 5).
4. **`ProvenantMemoryStore`**: preserves taint through memory store/load cycles to eliminate taint laundering (Section 5).
5. End-to-end evaluation with Qwen2.5-3B-Instruct (3.09B params): 216 mechanism attacks + 36 LLM attacks + 8 adversarial scenarios (Section 6).
6. Honest adversarial analysis identifying exactly when and how the guarantee fails (Section 7).

---

## 2. Related Work

**Indirect prompt injection.** Greshake et al. (2023) demonstrated that LLM agents can be manipulated through content injected into their context by external sources. Subsequent work formalized this threat in tool-integrated agents (Zhan et al., 2024; Zhang et al., 2024a).

**Agent safety benchmarks.** INJECAGENT (Zhan et al., 2024) evaluates 1,054 test cases across 17 user tools and 62 attacker tools, finding ASRs of 24.1--44.9% for GPT-3.5/GPT-4. Agent-SafetyBench (Zhang et al., 2024a) tests 2,000 cases across 349 environments, reporting 34.1--40.1% ASR for frontier models. Agent Security Bench (Zhang et al., 2024b) focuses on memory poisoning, finding 38.5% average ASR. **These baselines use different models, attack sets, and success criteria; direct numerical comparison with our results is not valid.**

**Information-flow control.** Our approach draws on classical IFC (Denning, 1976; Myers & Liskov, 1997), adapting taint tracking to the LLM agent setting. The key novelty is that the trust boundary is defined by *transport channel* rather than traditional security labels or code annotations.

**Reference monitors.** The reference monitor concept (Anderson, 1972) requires complete mediation, tamper-resistance, and verifiability. Our verifier is a reference monitor for control-plane mutations: it mediates every CP update proposal, is isolated from LLM processing, and has a simple, auditable implementation.

**Defenses.** Existing defenses include prompt engineering (delimiters, instruction hierarchy), fine-tuning (Anthropic, 2024), and detection classifiers (Lakera, 2024). These are empirical and probabilistic. Our approach provides a *conditional formal guarantee*---a fundamentally different security model that complements rather than replaces empirical defenses.

---

## 3. Formal Model

### 3.1 Agent State

The agent state at step $t$ is:

$$S_t = (P_t, M_t, B_t, G_t)$$

| Component | Description | CP-affecting? |
|-----------|-------------|:---:|
| $P_t$ (Control plane) | Permissions, integrations, policies | **Yes** |
| $M_t$ (Memory) | Persistent storage (SOUL.md, notes) | No* |
| $B_t$ (Behaviour) | System prompt extensions, reminders | No* |
| $G_t$ (IR graph) | Information-representation graph with taint tracking | No |

*Memory and behaviour can be modified by untrusted content (the theorem permits this). Only $P_t$ is protected. This is a deliberate scope limitation---see Section 1.1.

### 3.2 Control-Plane Boundary

The control plane is defined as:

$$P_t = (\text{Perm}_t, \text{Int}_t, \text{Policy}_t)$$

- **$\text{Perm}_t$**: Set of `(tool_name, allowed_actions)` pairs. Adding a new tool or expanding action scope requires verification.
- **$\text{Int}_t$**: Set of `(name, kind, config)` tuples for external integrations. Adding a chat channel, API connection, or webhook requires verification.
- **$\text{Policy}_t$**: Set of `(principal, allowed_paths)` mappings. Changing scope boundaries requires verification.

**Hard boundary rule:** A mutation is CP-affecting if and only if it modifies $\text{Perm}_t$, $\text{Int}_t$, or $\text{Policy}_t$. All other state changes ($M_t$, $B_t$, individual tool responses) are data-plane operations outside our protection scope.

### 3.3 Principal Hierarchy and Channel-Based Assignment

Every piece of information is attributed to a **principal** based on its **transport channel**:

| Channel | Principal | Trust Level | Assignment Rule |
|---------|-----------|:-----------:|:---:|
| Platform runtime | `SYS` | Trusted | Hardcoded |
| Authenticated user session | `USER` | Trusted | Authentication layer |
| HTTP scrape / email fetch | `WEB` | Untrusted | Transport protocol |
| Skill file store | `SKILL` | Untrusted | File origin |
| Tool API return value | `TOOL_OUTPUT` | Untrusted | API boundary |
| LLM generation | `SYS` (derived) | Inherits | **Must carry deps** |
| Memory retrieval | `SYS` (derived) | Inherits | **Must carry deps** |

**Critical design rule (addresses confused deputy):** Principal assignment is determined by the *transport channel*, never by payload content. The `ChannelPrincipalAssigner` (Section 5.1) enforces this invariant. Content claiming `[SYSTEM]` on a `WEB` channel receives principal `WEB`.

**Derived channels:** LLM generation and memory retrieval are *derived* channels. They do not receive a root trust level; instead, their taint is determined by their dependencies in the IR graph. This means:
- An LLM output processing untrusted input inherits taint from that input
- A memory retrieval inherits taint from the node that wrote to memory

### 3.4 Information Representation Graph

Every piece of data is represented as an **IR node** $n = (\text{id}, \pi, \tau, c, D)$ where $\pi$ is the principal, $\tau \in \{0, 1\}$ is the taint flag, $c$ is the content, and $D$ is the set of dependency node IDs.

**Taint propagation rule.** A node is tainted ($\tau = 1$) if its principal is untrusted *or* any dependency is tainted:

$$\tau(n) = \begin{cases} 1 & \text{if } \pi(n) \in \{\texttt{WEB}, \texttt{SKILL}, \texttt{TOOL\_OUTPUT}\} \\ 1 & \text{if } \exists\, d \in D(n) : \tau(d) = 1 \\ 0 & \text{otherwise} \end{cases}$$

The IR graph is **append-only**: once created, a node's taint is immutable. This is conservative tainting at the **turn level**: all LLM output from a turn that processed any untrusted input inherits taint from all inputs in that turn. Token-level or span-level tainting would be finer-grained but would require solving the provenance attribution problem for LLM internals, which we consider out of scope.

### 3.5 Verification Rules

A proposal to update $P_t$ to $P_{t+1}$, justified by certificate $C = (J, \text{confirmed})$, is **approved** iff:

- **V1 (Taint-free):** $\forall m \in \text{TransitiveClosure}(J): \tau(m) = 0$
- **V2 (Trusted provenance):** $\forall j \in J: \pi(j) \in \{\texttt{SYS}, \texttt{USER}\}$
- **V3 (User confirmation):** If the proposal adds integrations or changes permissions: $\text{confirmed} = \text{True}$

**V3 specification:** "User confirmation" means a signal arriving through the `AUTHENTICATED_USER_SESSION` channel (keyboard, voice, authenticated API call). It does *not* mean the LLM reporting "the user said yes"---that would be content-based assessment, violating the channel-based principal rule. The `ChannelPrincipalAssigner` enforces this distinction.

### 3.6 Three Assumptions

The theorem's guarantee is conditioned on:

- **A1 (Provenance completeness):** Every derived IR node records all data sources in its dependency set.
- **A2 (Principal accuracy):** Principals are assigned from transport channels, not content. Enforced by `ChannelPrincipalAssigner`.
- **A3 (Memory persistence):** Provenance chains survive memory store/load cycles. Enforced by `ProvenantMemoryStore`.

---

## 4. Theorem and Proof

### 4.1 Supporting Lemmas

**Lemma 1 (Taint Propagation Soundness).** For every node $n$ in $G_t$: $\pi(n) \in \text{Untrusted} \implies \tau(n) = 1$, and $(\exists\, d \in D(n): \tau(d) = 1) \implies \tau(n) = 1$.

*Proof.* By the taint computation rule applied at node creation. The IR graph is append-only, so taint is immutable.

**Lemma 2 (Untrusted Origin Implies Taint).** If $\pi(n) \in \{\texttt{WEB}, \texttt{SKILL}, \texttt{TOOL\_OUTPUT}\}$, then $\tau(n) = 1$.

*Proof.* Immediate from Lemma 1.

**Lemma 3 (Verifier Soundness).** If the verifier approves a change with certificate $(J, \text{confirmed})$, then all nodes in the transitive closure of $J$ have $\tau = 0$, all nodes in $J$ have trusted principals, and confirmation is present for integration/permission changes.

*Proof.* The verifier explicitly checks V1, V2, V3 and returns APPROVED only if all pass.

### 4.2 Control-Plane Integrity Theorem

**Theorem.** Let executions $E^{(1)}$ and $E^{(2)}$ share initial state $S_0$ and trusted inputs $T = (t_1, \ldots, t_k)$ but differ in untrusted inputs $U^{(1)}, U^{(2)}$. If assumptions A1--A3 hold and the verifier enforces V1--V3 at every step, then:

1. $P_t^{(1)} = P_t^{(2)}$ for all $t$ (control-plane equivalence).
2. No integrations or permission changes occur without user authorization.

*Proof.* By induction on $t$.

**Base case.** $P_0^{(1)} = P_0^{(2)}$ by shared initial state.

**Inductive step.** Assume $P_t^{(1)} = P_t^{(2)}$.

*Case 1 (Trusted input).* Both executions receive identical input $t_i$. The justifying node has a trusted principal with taint 0 (by A1 and Lemma 1). Both produce identical proposals; the verifier gives the same verdict. $P_{t+1}^{(1)} = P_{t+1}^{(2)}$.

*Case 2 (Untrusted input).* The input has principal $\pi \in \text{Untrusted}$. By A1, the derived node records this dependency. By A2, the principal is correctly assigned as untrusted. By A3, any memory intermediary preserves taint. By Lemma 2, the justifying node has $\tau = 1$. The verifier rejects (V1 fails). The control plane is unchanged: $P_{t+1}^{(i)} = P_t^{(i)}$.

Since $P_t^{(1)} = P_t^{(2)}$ by hypothesis: $P_{t+1}^{(1)} = P_{t+1}^{(2)}$. $\blacksquare$

**Note on tautology concern:** The theorem is *intentionally* a conditional result---it is a reference-monitor reduction, not a standalone empirical claim. Its value lies in precisely identifying the three assumptions (A1--A3) that suffice for integrity, and in showing that no additional properties of the LLM, the attacks, or the agent framework are needed. The adversarial analysis (Section 7) validates that the assumptions are non-trivial by showing that violating any one produces 100% bypass.

### 4.3 Corollaries

1. **Skill injection prevention.** A malicious skill embedding "add Telegram bot" cannot succeed: the skill content has principal `SKILL` (by A2), taint 1 (by Lemma 2), and fails V1.
2. **Persistent storage protection.** A skill writing to `SOUL.md` to alter policy: even if the write to $M_t$ succeeds (we do not protect memory), $P_t$ remains unchanged because any policy change proposal fails V1/V2.
3. **Transitive taint resistance.** An attacker cannot launder taint through derivation chains: the transitive closure in V1 ensures all ancestors are checked (conditional on A1).

---

## 5. Concrete Mitigations

### 5.1 ChannelPrincipalAssigner

Addresses assumption A2 and the confused deputy attack (Section 7, Scenario 2).

The `ChannelPrincipalAssigner` enforces a strict mapping from transport `Channel` enum to `Principal`. The mapping is immutable and content-independent:

```python
Channel.PLATFORM_RUNTIME        → Principal.SYS
Channel.AUTHENTICATED_USER_SESSION → Principal.USER
Channel.HTTP_SCRAPE              → Principal.WEB
Channel.SKILL_FILE_STORE         → Principal.SKILL
Channel.TOOL_API_RETURN          → Principal.TOOL_OUTPUT
Channel.LLM_GENERATION           → Principal.SYS  (derived: MUST carry deps)
Channel.MEMORY_RETRIEVAL          → Principal.SYS  (derived: MUST carry deps)
```

**Key invariant:** The `assign()` method takes only a `Channel` enum. It does not accept content. This makes it structurally impossible for content to influence principal assignment.

**Derived channels:** `LLM_GENERATION` and `MEMORY_RETRIEVAL` are flagged as `requires_dependencies=True`. The caller must provide dependency node IDs. If dependencies are omitted, the taint chain is broken (Scenario 1 bypass). Enforcing this at the framework level is an engineering requirement, not something the assigner can guarantee alone.

### 5.2 ProvenantMemoryStore

Addresses assumption A3 and the taint laundering attack (Section 7, Scenario 4).

The `ProvenantMemoryStore` wraps the agent's memory with provenance tracking:

- **On write:** Records the IR node ID that produced the content alongside the content itself.
- **On read:** Creates a new IR node that carries a dependency on the original write source, preserving the taint chain through memory.

This closes the laundering attack where untrusted content is stored in memory and later retrieved as a fresh, untainted `SYS` node. With `ProvenantMemoryStore`, the retrieval node inherits taint from the original write source.

**Design limitation:** The store tracks provenance for the *most recent* write to each key. If the memory system allows append or merge operations, the provenance model would need extension. We implement the simplest sound design.

### 5.3 What Remains Open: Provenance Completeness (A1)

Assumption A1 (provenance completeness) requires that every derived IR node records all its data sources. This is a *framework-level* property: the agent runtime must instrument every data flow from input to LLM to tool call. We do not provide an implementation for this because it depends on the specific agent framework (LangChain, AutoGen, CrewAI, etc.). What we prove is that *if* the framework provides this, the rest follows.

---

## 6. Empirical Evaluation

### 6.1 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct (Q4_K_M GGUF) |
| Parameters | 3.09 billion |
| Runtime | llama-cpp-python, CPU-only, 16 threads |
| Inference speed | ~13 tokens/second |
| Mechanism attacks | 216 (extended suite: 5 categories, 6 obfuscation levels) |
| LLM attacks | 36 (base suite: 4 categories, 3 obfuscation levels) |
| Random seed | 42 |
| Temperature | 0.0 (deterministic) |

**Attack taxonomy.** We generate attacks across four categories:

| Category | Description | Mechanism $n$ | LLM $n$ |
|----------|-------------|:---:|:---:|
| Malicious skill injection | Skills instructing the agent to add integrations | 54 | 9 |
| Tool output poisoning | Tool returns with embedded instructions | 60 | 9 |
| Memory poisoning | Attacks targeting persistent storage | 48 | 9 |
| Indirect prompt injection | External content with embedded instructions | 54 | 9 |

**Obfuscation levels:** Level 0 (direct instruction), Level 1 (social engineering), Level 2 (authority impersonation), Level 3 (base64 encoding), Level 4 (JSON wrapping), Level 5 (HTML comment hiding). The mechanism suite uses all 6 levels; the LLM suite uses levels 0--2 due to inference cost.

### 6.2 Part A: Mechanism Correctness (216 attacks)

| Test | Description | Result |
|------|-------------|--------|
| A1: Taint propagation | Untrusted principals produce taint=1; trusted produce taint=0 | **11/11 pass** |
| A2: Transitive taint | Taint propagates through derivation chains of any depth | **9/9 pass** |
| A3: Verifier rules | All 216 attacks correctly rejected (V1, V2, V3 all fire) | **216/216 correct** |
| A4: False positives | Legitimate user operations correctly allowed | **0% FPR** |
| A5: Dual execution | Theorem proof with 216 untrusted inputs, 3 lemmas | **All hold** |

### 6.3 Part B: End-to-End LLM Evaluation (36 attacks)

We send each of the 36 attack payloads to Qwen2.5-3B-Instruct via ChatML-formatted prompts with tool definitions. The model generates a response; we parse it for tool calls; if a dangerous tool is called, we run the *real verifier* with the full provenance chain.

| Metric | Value | 95% CI |
|--------|-------|--------|
| Baseline ASR (no verifier) | **77.8%** (28/36) | 61.9%--88.3% |
| Protected ASR (with verifier) | **0.0%** (0/36) | 0.0%--9.6% |
| Verifier catch rate | **100.0%** (28/28) | 87.9%--100.0% |

*Confidence intervals computed via Wilson score interval.*

**Per-category baseline ASR.**

| Attack Category | $n$ | LLM Attempted | Baseline ASR |
|----------------|-----|:---:|:---:|
| Malicious skill injection | 9 | 8 | 88.9% |
| Memory poisoning | 9 | 8 | 88.9% |
| Tool output poisoning | 9 | 6 | 66.7% |
| Indirect prompt injection | 9 | 6 | 66.7% |

**Interpretation.** The 77.8% baseline ASR demonstrates the LLM is genuinely susceptible---it outputs tool calls for `add_integration`, `write_file`, and `modify_permissions` in response to injected instructions. The protected ASR of 0.0% is a conditional guarantee (Section 4.2), not an independent empirical measurement. It confirms the mechanism works end-to-end *when assumptions hold*.

**Statistical limitations.** With $n = 36$, the 95% CI on baseline ASR is $\pm$13 percentage points. The 0% protected ASR has a one-sided 95% upper bound of 9.6%. These intervals reflect the sample size, which is limited by LLM inference cost on CPU. Larger-scale evaluation with faster hardware would improve precision.

### 6.4 Published Baselines (Context Only)

We list published ASRs for context. **Direct numerical comparison is not valid** because these studies use different models, attack sets, and evaluation protocols.

| Benchmark | Model | ASR | $n$ |
|-----------|-------|:---:|:---:|
| INJECAGENT (direct) | GPT-4 | 24.3% | 1,054 |
| INJECAGENT (enhanced) | GPT-4 | 34.2% | 1,054 |
| INJECAGENT (direct) | GPT-3.5 | 24.1% | 1,054 |
| INJECAGENT (enhanced) | GPT-3.5 | 44.9% | 1,054 |
| AgentSafetyBench | GPT-4o | 40.1% | 2,000 |
| AgentSafetyBench | Claude-3.5 | 34.1% | 2,000 |
| ASB (memory poisoning) | Avg | 38.5% | 398 |

The only valid comparison is **within our own system**: 77.8% baseline $\to$ 0.0% protected ASR (same model, same 36 attacks, same protocol; $z = 6.2$, $p < 10^{-9}$).

---

## 7. Adversarial Analysis: When Does the Guarantee Fail?

The 0.0% protected ASR is *structurally guaranteed* when A1--A3 hold. To characterize the defense honestly, we test **8 scenarios** using the 28 real LLM attack attempts from Part B, replayed under varied conditions.

### 7.1 Assumption Violation Scenarios

| # | Scenario | Assumption Violated | Bypass Rate |
|---|----------|:---:|:---:|
| 1 | **Broken provenance** | A1 | **100%** (28/28) |
| 2 | **Confused deputy** | A2 | **100%** (28/28) |
| 3 | V3 auto-confirmation | V3 only | **0%** (0/28) |
| 4 | **Taint laundering** | A3 | **100%** (28/28) |
| 5 | Deep chain (control) | None | **0%** (0/28) |
| 6 | Mixed provenance (control) | None | **0%** (0/28) |

**Scenario 1 (Broken provenance, A1 violated):** The agent framework fails to record that the LLM's tool call was derived from untrusted input. The resulting IR node appears as a fresh `SYS` node with taint 0. *This is the most likely real-world failure and the hardest to prevent---it requires framework-level instrumentation.*

**Scenario 2 (Confused deputy, A2 violated):** The agent incorrectly assigns `USER` principal to LLM output because the LLM's content claims "the user requested this." *This occurs in systems that infer principal from content rather than channel.* Our `ChannelPrincipalAssigner` (Section 5.1) eliminates this by design.

**Scenario 3 (V3 bypass):** Even with auto-confirmation, V1 still catches tainted justifications. Bypassing V3 alone is insufficient.

**Scenario 4 (Taint laundering, A3 violated):** Untrusted content stored in memory and retrieved without provenance. The taint chain is severed. *Analogous to TOCTOU vulnerabilities.* Our `ProvenantMemoryStore` (Section 5.2) closes this.

**Scenarios 5--6 (Controls):** Demonstrate that transitive taint and mixed-provenance detection work correctly even through deep chains.

### 7.2 Mitigation Scenarios

| # | Scenario | Mitigation Applied | Bypass Rate |
|---|----------|:---:|:---:|
| 7 | Channel-based assignment | `ChannelPrincipalAssigner` | **0%** (0/28) |
| 8 | Provenant memory | `ProvenantMemoryStore` | **0%** (0/28) |

**Scenario 7 (Mitigation for Scenario 2):** With `ChannelPrincipalAssigner`, the LLM output is assigned principal `SYS` via the `LLM_GENERATION` channel and *must* carry dependencies. The dependency on the untrusted source propagates taint correctly. Bypass rate: 0%.

**Scenario 8 (Mitigation for Scenario 4):** With `ProvenantMemoryStore`, memory retrieval creates a new IR node with a dependency on the write source. Taint is preserved through the store/load cycle. Bypass rate: 0%.

### 7.3 Residual Risk: Broken Provenance (Scenario 1)

Scenario 1 remains open. It requires the agent framework to correctly instrument every data flow from input to LLM to tool call. This is a *deployment engineering* problem, not a formal verification problem. Possible approaches:
- **Taint tracking in the LLM runtime:** Analogous to dynamic taint analysis in binary security.
- **Conservative default:** Treat all LLM output as tainted unless the framework explicitly certifies the provenance chain. This is safe but restrictive (high false positive rate for legitimate user-initiated CP changes).
- **Mandatory dependency annotations:** Require every IR node creation to provide at least one dependency (except for root inputs from authenticated channels).

---

## 8. Limitations

1. **Model scale.** We evaluate on Qwen2.5-3B-Instruct (3.09B params). Larger models may exhibit different attack compliance rates. The verifier's formal guarantee is model-independent, but the baseline ASR (77.8%) is model-specific.

2. **Sample size.** The LLM evaluation uses 36 attacks ($\pm$13pp at 95% CI). The mechanism evaluation uses 216 attacks. Real-world attack diversity may exceed our taxonomy.

3. **Provenance completeness (A1) is unsolved.** We prove that correct provenance implies integrity, but we do not solve the provenance tracking problem. As Scenario 1 shows, broken provenance breaks the guarantee entirely.

4. **Content-agnostic design.** The verifier does not inspect content. It cannot detect attacks through trusted channels or semantic manipulation that does not produce tool calls.

5. **Turn-level tainting is conservative.** All LLM output in a turn inherits taint from all inputs in that turn. Token-level or span-level tainting could reduce false positives for mixed-input turns but would require solving provenance attribution for LLM internals.

6. **No real agent stack integration.** We evaluate in a standalone framework, not within LangChain/AutoGen/CrewAI. Integration would require instrumenting the specific framework's data flow.

7. **Principal taxonomy is minimal.** Five principals (SYS, USER, WEB, SKILL, TOOL_OUTPUT) are a starting point. Production systems may need sub-principals (e.g., `SKILL_VERIFIED` vs `SKILL_COMMUNITY`, `WEB_AUTHENTICATED` vs `WEB_ANONYMOUS`).

---

## 9. Conclusion

We have presented a provenance-based information-flow framework for protecting LLM agent control planes. The framework's contribution is a *reduction*: it reduces the hard problem of control-plane integrity to three concrete deployment properties (provenance completeness, channel-based principal assignment, memory provenance persistence). We provide concrete implementations for two of three, and honest adversarial analysis showing that violating any assumption breaks the guarantee entirely.

The 0% protected ASR is a theorem consequence, not an empirical finding. Its value is in precisely identifying what assumptions suffice for integrity and what engineering work remains to enforce them. The 77.8% baseline ASR confirms that unprotected LLMs are genuinely vulnerable, making the defense meaningful.

---

## Repository Structure

```
src/                              Core formal model and proof
  principals.py                   Principal hierarchy (SYS, USER, WEB, SKILL, TOOL_OUTPUT)
  ir.py                           IR graph with automatic taint propagation
  state.py                        Agent state model: S_t = (P_t, M_t, B_t, G_t)
  verifier.py                     Three-rule control-plane update verifier (V1, V2, V3)
  channel_assigner.py             Channel-based principal assignment (closes confused deputy)
  provenant_memory.py             Provenance-preserving memory store (closes taint laundering)
  agent.py                        Stepwise agent execution engine
  theorem.py                      Formal theorem, 3 lemmas, mechanized dual-execution proof

eval/                             Evaluation framework
  attack_generator.py             Base suite (36) + extended suite (216) attacks
  llm_agents.py                   LangChain agent wrappers
  evaluator.py                    Metrics and evaluation logic
  visualize_results.py            Matplotlib/seaborn plotting

tests/                            70 unit tests
  test_principals.py              Principal trust classification
  test_ir.py                      IR graph and taint propagation
  test_verifier.py                Verifier rule enforcement
  test_channel_assigner.py        Channel-based assignment + content independence
  test_provenant_memory.py        Taint preservation through memory store/load
  test_agent.py                   Agent execution engine
  test_theorem.py                 Theorem proof and 5 evaluation scenarios

run_evaluation.py                 Main evaluation: Parts A, B, B2, C, D
PROOF.md                          Full formal proof with lemmas and corollaries
```

## Reproducing Results

```bash
# Install dependencies
pip install numpy scipy pandas matplotlib seaborn
pip install llama-cpp-python
pip install huggingface-hub

# Download model (2GB, one-time)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf')"

# Run full evaluation (~5 minutes on 16-core CPU)
python run_evaluation.py

# Run unit tests (70 tests, <1 second)
python -m pytest tests/ -v
```

## References

1. Anderson, J. P. (1972). Computer security technology planning study. Technical Report ESD-TR-73-51, Air Force Electronic Systems Division.
2. Denning, D. E. (1976). A lattice model of secure information flow. *Communications of the ACM*, 19(5), 236--243.
3. Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *arXiv:2302.12173*.
4. Myers, A. C., & Liskov, B. (1997). A decentralized model for information flow control. *SOSP*.
5. Zhan, Q., Liang, Z., Ying, Z., & Kang, D. (2024). InjecAgent: Benchmarking indirect prompt injections in tool-integrated LLM agents. *arXiv:2403.02691*.
6. Zhang, Z., Yuan, S., Chen, S., et al. (2024a). Agent-SafetyBench: Evaluating the safety of LLM agents. *arXiv:2412.14470*.
7. Zhang, H., Guo, W., Chen, Y., et al. (2024b). Agent Security Bench (ASB): Formalizing and benchmarking attacks and defenses in LLM-based agents. *arXiv:2410.02644*.
8. Anthropic. (2024). Claude's character. *anthropic.com*.
9. Lakera AI. (2024). PINT Benchmark: Evaluating prompt injection solutions. *lakera.ai*.

## License

Apache 2.0
