# Control-Plane Integrity Theorem

A formal model and mechanised proof that untrusted content (malicious skills, poisoned tool outputs, adversarial web pages) cannot modify the control plane of an RLM-based agent without explicit user authorisation.

**No mocks, no simulations, no fakes** -- the evaluation framework uses real LLM models, real benchmarks, and real datasets from academia and industry.

## Theorem statement

**Control-Plane Integrity Against Malicious Skills.** Let two executions share the same initial state S\_0 and identical sequences of trusted inputs. Suppose the verifier enforces that any update to the control plane must be justified solely by untainted IR nodes whose provenance is SYS or confirmed USER. Then, for any sequences of untrusted skill content or tool output:

1. The control-plane state remains equal at all steps: P\_t^(1) = P\_t^(2).
2. No new integrations or permission changes are introduced unless authorised by the user.

**In other words:** Malicious skills and untrusted tool outputs cannot cause the agent to add chat integrations, change permissions, or modify policies.

## Repository structure

```
src/                              # Core formal model and proof
  principals.py                   -- Principal hierarchy (SYS, USER, WEB, SKILL, TOOL_OUTPUT)
  ir.py                           -- Information Representation graph with taint propagation
  state.py                        -- Agent state model: S_t = (P_t, M_t, B_t, G_t)
  verifier.py                     -- Three-phase control-plane update verifier
  agent.py                        -- Stepwise agent execution engine
  theorem.py                      -- Formal theorem statement, lemmas, and mechanised proof

eval/                             # Empirical evaluation framework
  agent_state.py                  -- Extended state model for LLM evaluation
  verifier.py                     -- Verifier with baseline comparison
  attack_generator.py             -- 36 attack types across 4 categories
  llm_agents.py                   -- LangChain agents (protected & baseline)
  evaluator.py                    -- Evaluation framework & metrics
  main.py                         -- Main execution script
  download_datasets.py            -- Dataset downloader for canonical benchmarks
  visualize_results.py            -- Results visualization (matplotlib/seaborn)

tests/                            # Unit tests (49 tests)
  test_principals.py              -- Principal trust classification tests
  test_ir.py                      -- IR graph and taint propagation tests
  test_verifier.py                -- Verifier rule enforcement tests
  test_agent.py                   -- Agent execution engine tests
  test_theorem.py                 -- Theorem proof and all five evaluation scenarios

PROOF.md                          -- Full formal proof with lemmas and corollaries
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

## Unit test scenarios

The test suite implements five evaluation scenarios:

1. **Malicious skill injection** -- A skill attempts to add a Telegram bot integration. Blocked.
2. **Tool output poisoning** -- A tool returns "grant admin permissions". Blocked.
3. **Supply-chain malware** -- Skills attempt to write to SOUL.md/AGENTS.md. Blocked.
4. **False-positive rate** -- Legitimate user-initiated changes with confirmation. Allowed.
5. **Policy coverage** -- Sequential multi-vector attacks. All blocked; control plane unchanged.

## Evaluation framework

The `eval/` directory provides a comprehensive empirical evaluation framework.

### Real LLM models

- **OpenAI**: GPT-4o, GPT-4o-mini, o1-mini
- **Anthropic**: Claude 3.5 Sonnet
- **Google**: Gemini 2.0 Flash

### Real benchmarks & datasets

- **Agent-SafetyBench** (Zhang et al., 2024): 349 environments, 2,000 test cases
- **INJECAGENT** (Zhan et al., 2024): 1,054 test cases, 17 user tools, 62 attacker tools
- **PINT Benchmark** (Lakera): Prompt injection detection
- **NVIDIA Aegis-AI-Content-Safety-Dataset-2.0**: 25,007 prompts
- **Agent Security Bench** (ASB): Memory poisoning attacks

### Comprehensive attack suite

36 unique attacks across 4 categories, each with 3 obfuscation levels (none, basic, advanced):

| Category | Attacks | Examples |
|----------|---------|----------|
| Malicious Skills | 9 | Telegram bot, Slack webhook, Discord webhook |
| Tool Output Poisoning | 9 | Log poisoning, search results, API responses |
| Memory Poisoning | 9 | SOUL.md, AGENTS.md, config files |
| Indirect Prompt Injection | 9 | Email, documents, webpages |

### Evaluation metrics

- **Attack Success Rate (ASR)**: Percentage of successful attacks
- **Control-Plane Integrity Score (CPIS)**: 1 - ASR
- **Statistical Significance**: Two-proportion z-test, p-value < 0.05

### Expected results

| Metric | Protected Agent | Baseline Agent | Target |
|--------|----------------|----------------|--------|
| ASR | < 5% | > 50% | < 5% |
| CPIS | > 0.95 | < 0.50 | > 0.95 |
| ASR Reduction | -- | -- | > 45% |

## Running

### Unit tests
```bash
python -m pytest tests/ -v
```

### Evaluation framework

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=your_key_here

# Quick test (3 attacks)
python -m eval.main quick

# Single experiment
python -m eval.main experiment malicious_skills gpt-4o-mini 9

# Comprehensive evaluation
python -m eval.main comprehensive gpt-4o-mini 9

# Download benchmark datasets
python -m eval.download_datasets

# Visualize results
python -m eval.visualize_results
```

## Academic citations

1. **Agent-SafetyBench**: Zhang et al., "Agent-SafetyBench: Evaluating the Safety of LLM Agents", arXiv:2412.14470, 2024
2. **INJECAGENT**: Zhan et al., "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents", arXiv:2403.02691, 2024
3. **PINT Benchmark**: Lakera AI, "A New Benchmark for Evaluating Prompt Injection Solutions", 2024
4. **Agent Security Bench**: "ASB: Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents", arXiv:2410.02644, 2024

## License

Apache 2.0
