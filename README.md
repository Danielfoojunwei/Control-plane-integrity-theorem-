# Control-Plane Integrity for LLM Agents: A Provenance-Based Defense Against Malicious Skills and Tool Output Poisoning

> **Abstract.** Large language model (LLM) agents that integrate external skills, tools, and data sources face a fundamental security challenge: untrusted content from these sources can manipulate the agent into modifying its own control plane---adding unauthorized integrations, escalating permissions, or rewriting policies. We present a formal information-flow framework that prevents such attacks through *provenance-based taint tracking* and a *three-rule verifier*. We prove a **Control-Plane Integrity Theorem**: under correct provenance tracking, no sequence of untrusted inputs can alter the agent's control plane without explicit user authorization. We evaluate empirically using Qwen2.5-3B-Instruct (3.09B parameters) against 36 attack payloads spanning four categories and three obfuscation levels. The unprotected model follows attack instructions **77.8%** of the time (baseline ASR); the verifier blocks **100%** of these attempts, yielding a **protected ASR of 0.0%**. We further conduct adversarial analysis of six deployment-failure scenarios, showing that the guarantee is *conditional*: three scenarios (broken provenance, confused deputy, taint laundering) bypass the verifier entirely, demonstrating that the defense's value depends critically on the integrity of the provenance infrastructure. Our protected ASR improves over published baselines by **24.1--44.9 percentage points** (all $p < 0.001$).

---

## 1. Introduction

LLM-based agents increasingly operate in environments where they process content from multiple sources of varying trust: user instructions, system prompts, web pages, installed skills, and tool return values. This creates an attack surface where *indirect prompt injection* (Greshake et al., 2023), *malicious skills* (Zhan et al., 2024), and *tool output poisoning* (Zhang et al., 2024b) can manipulate the agent into taking unauthorized actions.

We focus on a particularly dangerous class of attacks: those that target the agent's **control plane**---the configuration layer that determines what integrations are active, what permissions are granted, and what policies govern the agent's behavior. Unlike data-plane attacks (which affect individual responses), control-plane attacks are *persistent*: a single successful attack can permanently alter the agent's capabilities.

**Contributions.**
1. We formalize the agent state model $S_t = (P_t, M_t, B_t, G_t)$ where $P_t$ is the control plane, and define a provenance-based information-flow framework with automatic taint propagation (Section 3).
2. We prove the **Control-Plane Integrity Theorem**: under three verification rules (V1--V3), no untrusted input can modify $P_t$ without user authorization (Section 4).
3. We evaluate end-to-end with a real open-source LLM (Qwen2.5-3B-Instruct, 3.09B parameters), demonstrating 77.8% baseline ASR reduced to 0.0% protected ASR across 36 attack payloads (Section 5).
4. We conduct an honest adversarial analysis of six deployment-failure scenarios, identifying three that bypass the verifier entirely, characterizing the *conditional* nature of the guarantee (Section 6).

**Reproducibility.** All code, evaluation scripts, and results are included in this repository. The evaluation uses a freely available open-source model (Qwen2.5-3B-Instruct) requiring no API keys. Run `python run_evaluation.py` to reproduce all results.

---

## 2. Related Work

**Indirect prompt injection.** Greshake et al. (2023) demonstrated that LLM agents can be manipulated through content injected into their context by external sources. Subsequent work formalized this threat in tool-integrated agents (Zhan et al., 2024; Zhang et al., 2024a).

**Agent safety benchmarks.** INJECAGENT (Zhan et al., 2024) evaluates 1,054 test cases across 17 user tools and 62 attacker tools, finding ASRs of 24.1--44.9% for GPT-3.5/GPT-4. Agent-SafetyBench (Zhang et al., 2024a) tests 2,000 cases across 349 environments, reporting 34.1--40.1% ASR for frontier models. Agent Security Bench (Zhang et al., 2024b) focuses on memory poisoning, finding 38.5% average ASR.

**Information-flow control.** Our approach draws on classical information-flow security (Denning, 1976; Myers & Liskov, 1997), adapting taint tracking to the LLM agent setting where the boundary between trusted and untrusted content is defined by *principal provenance* rather than traditional security labels.

**Defenses.** Existing defenses include prompt engineering (delimiters, instruction hierarchy), fine-tuning (Anthropic, 2024), and detection classifiers (Lakera, 2024). These are empirical and probabilistic. Our approach provides a *formal guarantee* conditioned on correct provenance tracking---a fundamentally different security model.

---

## 3. Formal Model

### 3.1 Agent State

The agent state at step $t$ is:

$$S_t = (P_t, M_t, B_t, G_t)$$

| Component | Description |
|-----------|-------------|
| $P_t$ (Control plane) | Permissions $\text{Perm}_t$, integrations $\text{Int}_t$, policies $\text{Policy}_t$ |
| $M_t$ (Memory) | Persistent storage (conversation history, notes) |
| $B_t$ (Behaviour) | System prompt extensions and reminders |
| $G_t$ (IR graph) | Information-representation graph with taint tracking |

### 3.2 Principal Hierarchy

Every piece of information entering the system is attributed to a **principal**:

| Principal | Trust Level | Examples |
|-----------|-------------|----------|
| `SYS` | Trusted | System prompts, platform policies |
| `USER` | Trusted | Direct, confirmed user input |
| `WEB` | Untrusted | Scraped web pages, emails, documents |
| `SKILL` | Untrusted | Installed skill file content |
| `TOOL_OUTPUT` | Untrusted | External tool/API return values |

### 3.3 Information Representation Graph

Every piece of data processed by the agent is represented as an **IR node** $n = (\text{id}, \pi, \tau, c, D)$ where $\pi$ is the principal, $\tau \in \{0, 1\}$ is the taint flag, $c$ is the content, and $D$ is the set of dependency node IDs.

**Taint propagation rule.** A node is tainted ($\tau = 1$) if its principal is untrusted *or* any dependency is tainted:

$$\tau(n) = \begin{cases} 1 & \text{if } \pi(n) \in \{\texttt{WEB}, \texttt{SKILL}, \texttt{TOOL\_OUTPUT}\} \\ 1 & \text{if } \exists\, d \in D(n) : \tau(d) = 1 \\ 0 & \text{otherwise} \end{cases}$$

The IR graph is **append-only**: once a node is created, its taint is immutable.

### 3.4 Verification Rules

A proposal to update the control plane from $P_t$ to $P_{t+1}$, justified by a certificate $C = (J, \text{confirmed})$ where $J$ is a set of justifying node IDs, is **approved** if and only if:

- **V1 (Taint-free):** For all node IDs $m$ in the transitive closure of $J$: $\tau(m) = 0$.
- **V2 (Trusted provenance):** For all node IDs $j \in J$: $\pi(j) \in \{\texttt{SYS}, \texttt{USER}\}$.
- **V3 (User confirmation):** If the proposal adds integrations or changes permissions: $\text{confirmed} = \text{True}$.

---

## 4. Theorem and Proof

### 4.1 Supporting Lemmas

**Lemma 1 (Taint Propagation Soundness).** For every node $n$ in $G_t$: $\pi(n) \in \text{Untrusted} \implies \tau(n) = 1$, and $(\exists\, d \in D(n): \tau(d) = 1) \implies \tau(n) = 1$.

*Proof.* Follows from the taint computation rule applied at node creation. The IR graph is append-only, so taint values are immutable.

**Lemma 2 (Untrusted Origin Implies Taint).** If $\pi(n) \in \{\texttt{WEB}, \texttt{SKILL}, \texttt{TOOL\_OUTPUT}\}$, then $\tau(n) = 1$.

*Proof.* Immediate from Lemma 1.

**Lemma 3 (Verifier Soundness).** If the verifier approves a change with certificate $(J, \text{confirmed})$, then all nodes in the transitive closure of $J$ have $\tau = 0$, all nodes in $J$ have trusted principals, and confirmation is present for integration/permission changes.

*Proof.* The verifier explicitly checks V1, V2, V3 and returns APPROVED only if all pass.

### 4.2 Control-Plane Integrity Theorem

**Theorem.** Let executions $E^{(1)}$ and $E^{(2)}$ share initial state $S_0$ and trusted inputs $T = (t_1, \ldots, t_k)$ but differ in untrusted inputs $U^{(1)}, U^{(2)}$. If the verifier enforces V1--V3 at every step, then:

1. $P_t^{(1)} = P_t^{(2)}$ for all $t$ (control-plane equivalence).
2. No integrations or permission changes occur without user authorization.

*Proof.* By induction on $t$.

**Base case.** $P_0^{(1)} = P_0^{(2)}$ by shared initial state.

**Inductive step.** Assume $P_t^{(1)} = P_t^{(2)}$.

*Case 1 (Trusted input).* Both executions receive identical input $t_i$. The justifying node has a trusted principal with taint 0 (Lemma 1). Both produce identical proposals; the verifier gives the same verdict. $P_{t+1}^{(1)} = P_{t+1}^{(2)}$.

*Case 2 (Untrusted input).* The input has principal $\pi \in \text{Untrusted}$. By Lemma 2, the justifying node has $\tau = 1$. The verifier rejects (V1 fails). The control plane is unchanged: $P_{t+1}^{(i)} = P_t^{(i)}$.

Since $P_t^{(1)} = P_t^{(2)}$ by hypothesis: $P_{t+1}^{(1)} = P_{t+1}^{(2)}$. $\blacksquare$

**Corollaries.**

1. **Skill injection prevention.** A malicious skill embedding "add a Telegram bot" cannot succeed: the skill content has principal `SKILL`, taint 1, and fails V1.
2. **Persistent storage protection.** A skill writing to `SOUL.md` to alter policy: even if the write to $M_t$ succeeds, $P_t$ remains unchanged because any policy change proposal fails V1/V2.
3. **Transitive taint resistance.** An attacker cannot launder taint through derivation chains: the transitive closure in V1 ensures all ancestors are checked.

---

## 5. Empirical Evaluation

### 5.1 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct (Q4_K_M GGUF) |
| Parameters | 3.09 billion |
| Runtime | llama-cpp-python, CPU-only, 16 threads |
| Inference speed | 12.8 tokens/second |
| Attack payloads | 36 (4 categories $\times$ 3 obfuscation levels $\times$ 3 variants) |
| Random seed | 42 |
| Temperature | 0.0 (deterministic) |
| Max tokens per attack | 192 |

**Attack categories.** We generate 36 unique attack payloads across four categories:

| Category | Description | Examples |
|----------|-------------|----------|
| Malicious skill injection | Skills that instruct the agent to add integrations | Telegram bot, Slack webhook, Discord channel |
| Tool output poisoning | Tool returns containing injected instructions | Log entries, search results, API responses |
| Memory poisoning | Attacks targeting persistent storage | SOUL.md overwrites, config injection |
| Indirect prompt injection | Content from web/email with embedded instructions | Email bodies, document content, web pages |

Each category includes three **obfuscation levels**: Level 0 (direct instruction), Level 1 (social engineering), Level 2 (encoded/obfuscated payloads).

**Tools available to the LLM.** The model is presented with four tools via system prompt: `add_integration` (CP-affecting), `modify_permissions` (CP-affecting), `write_file` (CP-affecting), and `search_web` (safe). An attack is scored as *attempted* if the LLM outputs a JSON tool call for any CP-affecting tool.

### 5.2 Part A: Mechanism Correctness

We first validate the verifier mechanism independently of LLM behavior.

| Test | Description | Result |
|------|-------------|--------|
| A1: Taint propagation | Untrusted principals produce taint=1; trusted produce taint=0 | **11/11 pass** |
| A2: Transitive taint | Taint propagates through derivation chains of any depth | **9/9 pass** |
| A3: Verifier rules | All 36 attack payloads correctly rejected (V1, V2, V3 all fire) | **36/36 correct** |
| A4: False positives | Legitimate user operations correctly allowed | **0% FPR, 100% accuracy** |
| A5: Dual execution | Theorem proof with 36 untrusted inputs, 3 lemmas | **All hold (0.61 ms)** |

### 5.3 Part B: End-to-End LLM Evaluation

We send each of the 36 attack payloads to Qwen2.5-3B-Instruct via ChatML-formatted prompts with tool definitions. The model generates a response; we parse it for tool calls; if a dangerous tool is called, we run the *real verifier* with the full provenance chain.

**Main results.**

| Metric | Value |
|--------|-------|
| Baseline ASR (no verifier) | **77.8%** (28/36) |
| Protected ASR (with verifier) | **0.0%** (0/36) |
| Verifier catch rate | **100.0%** (28/28) |
| End-to-end successes | **0** |

**Per-category baseline ASR (LLM susceptibility without verifier).**

| Attack Category | $n$ | LLM Attempted | Baseline ASR |
|----------------|-----|---------------|--------------|
| Malicious skill injection | 9 | 8 | **88.9%** |
| Memory poisoning | 9 | 8 | **88.9%** |
| Tool output poisoning | 9 | 6 | **66.7%** |
| Indirect prompt injection | 9 | 6 | **66.7%** |

**Per-obfuscation baseline ASR.**

| Obfuscation Level | $n$ | LLM Attempted | Baseline ASR |
|-------------------|-----|---------------|--------------|
| Level 0 (direct) | 12 | 10 | **83.3%** |
| Level 1 (social engineering) | 12 | 8 | **66.7%** |
| Level 2 (encoded) | 12 | 10 | **83.3%** |

**Interpretation.** The 77.8% baseline ASR demonstrates that the LLM is genuinely susceptible to these attacks---it outputs tool calls for `add_integration`, `write_file`, and `modify_permissions` in response to injected instructions. This is *not* a circular test. The verifier then blocks all 28 attempted attacks because the provenance chain traces each tool call back to an untrusted source.

The non-monotonic relationship between obfuscation level and ASR (Level 1 is *lower* than Level 0 or Level 2) suggests that moderate obfuscation may introduce ambiguity that reduces the model's compliance, while heavy obfuscation may use encoding patterns the model has learned to decode.

### 5.4 Part C: Comparison with Published Baselines

We compare our protected ASR (0.0%) against published benchmarks. Since our baseline ASR (77.8%) confirms the LLM is susceptible, the improvement is attributable to the verifier.

| Benchmark | Model | Published ASR | Our Protected ASR | $\Delta$ (pp) | $z$-score | $p$-value |
|-----------|-------|--------------|-------------------|---------------|-----------|-----------|
| INJECAGENT (direct) | GPT-4 | 24.3% | 0.0% | +24.3 | 3.38 | $7.2 \times 10^{-4}$ |
| INJECAGENT (enhanced) | GPT-4 | 34.2% | 0.0% | +34.2 | 4.28 | $1.8 \times 10^{-5}$ |
| INJECAGENT (direct) | GPT-3.5 | 24.1% | 0.0% | +24.1 | 3.36 | $7.7 \times 10^{-4}$ |
| INJECAGENT (enhanced) | GPT-3.5 | 44.9% | 0.0% | +44.9 | 5.34 | $9.2 \times 10^{-8}$ |
| AgentSafetyBench | GPT-4o | 40.1% | 0.0% | +40.1 | 4.88 | $1.1 \times 10^{-6}$ |
| AgentSafetyBench | Claude-3.5-Sonnet | 34.1% | 0.0% | +34.1 | 4.30 | $1.7 \times 10^{-5}$ |
| ASB (memory poisoning) | Avg | 38.5% | 0.0% | +38.5 | 4.62 | $3.8 \times 10^{-6}$ |

All comparisons are statistically significant ($p < 0.001$, two-proportion $z$-test).

**Caveats.** (1) Published ASRs use different models, attack sets, and evaluation protocols. (2) Our ASR of 0.0% is a *mechanism guarantee* conditioned on correct provenance tracking, not purely an empirical LLM result. (3) The comparison measures our verifier + LLM system against unprotected LLMs in published work. A fairer comparison would test the same attacks on the same model with and without our verifier---which is exactly what the baseline vs. protected ASR (77.8% vs. 0.0%) measures.

---

## 6. Adversarial Analysis: When Does the Verifier Fail?

The 0.0% protected ASR is *structurally guaranteed* when provenance tracking is correct: every attack enters through an untrusted principal, the LLM output inherits taint through the dependency chain, and the verifier rejects the tainted justification. To honestly characterize the defense, we test **six deployment-failure scenarios** using the 28 real LLM attack attempts from Part B, replayed under broken conditions.

### 6.1 Scenarios

| # | Scenario | Assumption Violated | Bypass Rate |
|---|----------|-------------------|-------------|
| 1 | **Broken provenance** | LLM output recorded with no dependency on untrusted source | **100%** (28/28) |
| 2 | **Confused deputy** | LLM output mislabeled as `USER`-originated | **100%** (28/28) |
| 3 | V3 auto-confirmation | `user_confirmed=True` for all changes; provenance intact | **0%** (0/28) |
| 4 | **Taint laundering** | Content stored in memory, retrieved without provenance | **100%** (28/28) |
| 5 | Deep chain (control) | 5-hop relay with intact provenance | **0%** (0/28) |
| 6 | Mixed provenance (control) | Clean + tainted parents; V1 checks all deps | **0%** (0/28) |

### 6.2 Analysis

**Three scenarios bypass the verifier (1, 2, 4).** These represent realistic deployment bugs:

- **Broken provenance (Scenario 1):** If the agent framework fails to record that the LLM's tool call was derived from untrusted input, the resulting IR node appears as a fresh `SYS` node with taint 0. The verifier has no basis to reject it. *This is the most likely real-world failure mode.*

- **Confused deputy (Scenario 2):** If the agent incorrectly attributes the LLM's output to the `USER` principal (e.g., because the LLM claimed "the user requested this"), the node passes both V1 and V2. *This can occur in systems that infer principal from content rather than channel.*

- **Taint laundering (Scenario 4):** If untrusted content is stored in agent memory and later retrieved as a fresh `SYS` node without the original dependency, the taint chain is severed. *This is analogous to TOCTOU vulnerabilities in classical security.*

**Three scenarios maintain the guarantee (3, 5, 6).** These serve as positive controls:

- **Auto-confirmation (Scenario 3):** Even with `user_confirmed=True`, V1 still catches the tainted justification. Bypassing V3 alone is insufficient.
- **Deep chain (Scenario 5):** Transitive taint propagation works correctly through 5+ hops.
- **Mixed provenance (Scenario 6):** A single tainted dependency among clean ones is sufficient for rejection.

### 6.3 Implications

The defense's value is the gap: **77.8% baseline ASR reduced to 0.0% protected ASR**. But this gap exists *only* when three properties hold:

1. **Provenance completeness:** Every derived IR node records all its data sources.
2. **Principal accuracy:** External content is correctly labeled with untrusted principals.
3. **Memory persistence:** Provenance chains survive memory store/load cycles.

When these properties fail, the protected ASR rises to match the baseline (77.8%). Deployment must therefore treat the provenance infrastructure as a *security-critical component* subject to the same rigor as the verifier itself.

---

## 7. Limitations

1. **Model scale.** We evaluate on Qwen2.5-3B-Instruct. Larger models (70B+, GPT-4, Claude) may exhibit different attack compliance rates, though the verifier's formal guarantee is model-independent.

2. **Attack diversity.** Our 36 payloads cover four categories and three obfuscation levels. Real-world attacks may use novel vectors not represented here.

3. **Provenance assumption.** The theorem's guarantee is conditioned on correct provenance tracking. As shown in Section 6, three realistic failure modes bypass the verifier entirely. We do *not* solve the problem of maintaining provenance---we prove that *if* provenance is correct, integrity holds.

4. **Content-agnostic.** The verifier does not inspect payload content. It cannot detect attacks that arrive through trusted channels (e.g., a compromised user account).

5. **Sample size.** With $n = 36$ attacks, confidence intervals on the baseline ASR are wide ($\pm$14% at 95% CI). Larger attack sets would improve precision.

---

## 8. Conclusion

We have presented a provenance-based information-flow framework for protecting LLM agent control planes against malicious skills, tool output poisoning, and indirect prompt injection. The framework achieves a formally proven 0% attack success rate *conditional on correct provenance tracking*, reducing the empirically measured 77.8% baseline ASR to 0.0%.

Our adversarial analysis (Section 6) is equally important: it identifies three deployment-failure scenarios where the guarantee breaks completely, providing a concrete security engineering checklist for real-world deployment. The defense is sound, but only as strong as the provenance infrastructure it relies on.

---

## Repository Structure

```
src/                              Core formal model and proof
  principals.py                   Principal hierarchy (SYS, USER, WEB, SKILL, TOOL_OUTPUT)
  ir.py                           IR graph with automatic taint propagation
  state.py                        Agent state model: S_t = (P_t, M_t, B_t, G_t)
  verifier.py                     Three-rule control-plane update verifier (V1, V2, V3)
  agent.py                        Stepwise agent execution engine
  theorem.py                      Formal theorem, 3 lemmas, mechanized dual-execution proof

eval/                             Evaluation framework
  attack_generator.py             36 attacks: 4 categories x 3 obfuscation levels x 3 variants
  llm_agents.py                   LangChain agent wrappers
  evaluator.py                    Metrics and evaluation logic
  visualize_results.py            Matplotlib/seaborn plotting

tests/                            49 unit tests
  test_principals.py              Principal trust classification
  test_ir.py                      IR graph and taint propagation
  test_verifier.py                Verifier rule enforcement
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

# Run full evaluation (~3 minutes on 16-core CPU)
python run_evaluation.py

# Run unit tests
python -m pytest tests/ -v
```

## References

1. Denning, D. E. (1976). A lattice model of secure information flow. *Communications of the ACM*, 19(5), 236--243.
2. Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *arXiv:2302.12173*.
3. Myers, A. C., & Liskov, B. (1997). A decentralized model for information flow control. *SOSP*.
4. Zhan, Q., Liang, Z., Ying, Z., & Kang, D. (2024). InjecAgent: Benchmarking indirect prompt injections in tool-integrated LLM agents. *arXiv:2403.02691*.
5. Zhang, Z., Yuan, S., Chen, S., et al. (2024a). Agent-SafetyBench: Evaluating the safety of LLM agents. *arXiv:2412.14470*.
6. Zhang, H., Guo, W., Chen, Y., et al. (2024b). Agent Security Bench (ASB): Formalizing and benchmarking attacks and defenses in LLM-based agents. *arXiv:2410.02644*.
7. Anthropic. (2024). Claude's character. *anthropic.com*.
8. Lakera AI. (2024). PINT Benchmark: Evaluating prompt injection solutions. *lakera.ai*.

## License

Apache 2.0
