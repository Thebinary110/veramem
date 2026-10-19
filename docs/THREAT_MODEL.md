# VeraMem threat model

This document describes what VeraMem protects, who it protects it from, where the trust
boundaries are, and how each threat is handled. Threats are grouped by STRIDE category.
Every entry names the code that mitigates it, the test that covers it, and the risk that
remains. Some guarantees are known to break today. Those entries are marked **OPEN**, and
each one has a strict-xfail test that will start failing as soon as the bug is fixed.

`SECURITY.md` lists the guarantees in short form. This document is the reasoning behind
them.

## 1. Design premise

**Model output is untrusted.** VeraMem asks a model to extract memories, reconcile them,
answer questions, distil experiences and de-identify shared content. That model reads
tool output, web pages and documents, so it can be prompt-injected at any time. VeraMem's
guarantees therefore have to come from deterministic code that runs *after* the model:

- grounding (the quote must exist in the source message)
- role-based authority (set by the source role, never by the model)
- scope-restricted reconciliation (the model only sees the caller's own neighbours, under
  short local ids)
- the SQL access predicate, applied before ranking
- the citation lock
- the PII and canary gate
- the hash-chained ledger

`tests/test_adversarial.py` checks these guarantees against a fully compromised model,
`MaliciousLLM`. It is built on `veramem.providers.ScriptedLLM` and returns hostile output
for every prompt.

## 2. Assets

| Asset | Where it lives | Why it matters |
|---|---|---|
| A user's memories (facts, preferences, events, experiences, summaries) | `memories` table, `memories_fts`, `entity_links`, `evidence` | Confidentiality between tenants. Integrity matters too, because agents act on these memories. |
| Raw episodes (verbatim messages, tool output, documents) | `episodes`, `episodes_fts` | The most sensitive data. This is the evidence every memory points back to. |
| Authority labels | `memories.authority` | They decide whether `check_action` lets a memory bind a protected tool call. |
| Ledger | `ledger` (hash chain) | Detects tampering and keeps an audit trail. Erasure leaves only hash tombstones. |
| Shared experience pool | `memories` with `visibility` `team` or `public` | Readable by other tenants, so it must never carry personal data. |
| API keys and JWT secrets | `api_keys` (SHA-256 hashes), env `VERAMEM_API_KEY`, `VERAMEM_JWT_SECRET` | Control who may act as which user. |
| Query vectors | `traces.qvec`, `outcomes.qvec` | Embeddings of past queries. They can be partly inverted back to query text, so they count as personal data. |
| Traces and outcome stats | `traces`, `memories.successes/failures` | They drive ranking, so poisoning them shifts what users see. |
| Pseudonym secret | `meta.pseudonym_secret` | Links shared items to contributors for withdrawal. If it leaks, contributors can be re-identified. |

## 3. Actors

| Actor | Capabilities assumed |
|---|---|
| **End user** (legitimate) | Writes and reads their own memories through the SDK, REST, or MCP. |
| **Other tenant** (curious or malicious) | A legitimate account on the same deployment. Crafts queries, ids, and shared content to read or alter another user's data. |
| **Malicious tool or document content** | Text that reaches `add()` with role `tool` or `document`, such as a web page, an email, or an API response. It fully controls its own wording, and it is the main source of prompt injection. |
| **Compromised or prompt-injected LLM** | Returns arbitrary JSON or text for any prompt: invented quotes and ids, wrong kinds, huge or malformed output. It sees only what VeraMem puts in the prompt. |
| **Operator with DB file access** | Can read and write the SQLite file directly. |
| **Network attacker** | Can reach the REST port. Can sniff traffic only if TLS is not terminated in front of VeraMem. |
| **MCP client (agent)** | The model inside Claude Desktop, Cursor, or another client. It calls the `remember`, `recall`, `correct`, `forget` and other tools, and it may itself be prompt-injected. |

## 4. Trust boundaries

1. **LLM output to VeraMem.** All model output is untrusted. It is parsed by
   `providers.llm.parse_json`, then validated in `write.Writer._extract` and `_decide`,
   `answer.Answerer.answer`, `learning.Learner._manage_experience`, and
   `sharing.Sharer.share`.
2. **Message content to VeraMem.** Tool and document content is untrusted. Authority comes
   from the *role* the integrator assigns (`write._role`, `types.Authority.for_source`).
   **Assumption:** the integrator labels roles correctly.
3. **REST caller to server.** Identity is established in `server/app.py` together with
   `auth.py`. See §6.
4. **MCP client to MCP server.** The user is fixed by the `VERAMEM_USER` environment
   variable, not by the model (`mcp_server.build`). The `role` argument of `remember` *is*
   chosen by the model.
5. **Process to SQLite file.** The file is trusted storage. The ledger can detect changes to
   it but cannot prevent them.

## 5. Threats

Columns: **Threat** · **Mitigation in code** · **Tests** · **Residual risk**.

### S: Spoofing

| # | Threat | Mitigation | Tests | Residual risk |
|---|---|---|---|---|
| S1 | A REST caller passes another user's `user_id`. | Per-user API keys pinned to one user and agent, admin keys that may act for anyone, and JWT `sub` as the user (`auth.py`, the `scope()` helper in `server/app.py`). The CLI refuses non-loopback binds without auth. | `test_auth.py::test_user_key_is_pinned_to_its_user`, `test_interfaces.py::test_rest_api_key` | **In open mode (no keys, no JWT), `user_id` is trusted.** Any local caller can act as anyone. The admin key (`VERAMEM_API_KEY`) can act for any user by design, so it must be kept server-side only. |
| S2 | A tool or document claim presents itself as the user's own statement ("the user prefers ..."). | `Authority.for_source(ep.source)` assigns authority from the role of the *episode the quote is located in*, never from the model. A non-user source cannot produce `Kind.PREFERENCE` (`write.Writer._extract`). | `test_tool_claim_disguised_as_user_preference_stays_unendorsed`, `test_non_user_source_cannot_create_preference`, `test_role_policy_authority_labels`, `test_fuzz.py::test_well_typed_hostile_extraction_keeps_guarantees` | See T3: the *text* is not bound to the quote. |
| S3 | The model spoofs a source label (`"source": "m7"`) to move a claim to another message. | `_extract` locates the quote itself. If the label is wrong, it searches only the current batch for the quote and takes authority from the message where the quote is actually found. | `test_tool_claim_disguised_as_user_preference_stays_unendorsed` (spoofed `m7`) | See T3. |
| S4 | An MCP agent calls `remember(role="user")` to store tool content as user-authored. | None. The role argument is model-chosen. | none | **Residual.** A prompt-injected MCP agent can create AUTHORIZED memories for the configured user. Integrators who need strict authority should pin the role server-side. |
| S5 | A lookalike user id (`"victim "`, `"VICTIM"`, a zero-width character, `"' OR '1'='1"`) reaches the victim's data. | Ids are compared exactly through bound parameters (`policy.read_filter`, `episode_filter`, `check_write`). FTS owner tokens are hashes of the exact id. | `test_lookalike_user_ids_are_distinct_tenants` | Normalising ids (case, Unicode) is up to the integrator. |

### T: Tampering

| # | Threat | Mitigation | Tests | Residual risk |
|---|---|---|---|---|
| T1 | The model invents memories that nobody said. | Every candidate needs a verbatim quote, which `grounding.locate` must find (≥ 0.88 fuzzy) in the source. Otherwise the candidate is rejected (`AddResult.rejected`). | `test_ungrounded_quotes_are_rejected`, `test_ungrounded_candidate_is_rejected`, `test_hostile_model_battery_keeps_all_invariants` | Fuzzy matching can ground a quote that differs slightly from the source. |
| T2 | A tool message drives an UPDATE or DELETE of an authorized memory. | `Writer._update` and `_retract` refuse when the candidate's authority rank is lower than the target's. The update is kept as a separate, conflicting memory, and the retract becomes a NOOP. `_confirm` never lowers authority. | `test_tool_driven_update_and_delete_cannot_touch_authorized_memory`, `test_low_authority_cannot_overwrite_user_fact` | None known for this path. |
| T3 | **OPEN.** The model attaches a tool's claim to a benign *user* quote ("sounds good") in the same batch, which yields an AUTHORIZED preference. | Not mitigated. `_extract` binds only the quote to the source, never the candidate text. `grounding.support_ratio` exists but is never called. | `test_tool_claim_cannot_borrow_a_user_quote` (xfail) | High for agents that mix user and tool turns in one `add()`. Suggested fix: require `support_ratio(text, quote)` ≥ τ for AUTHORIZED candidates, or cap authority at the weakest source in the batch whose words appear in the text. |
| T4 | **OPEN.** The reconcile manager rewrites an authorized memory into arbitrary text (UPDATE with free-form `text`). | Only the authority guard applies. The merged text is stored without any grounding check (`Writer._decide` and `_update`). | `test_reconcile_merged_text_must_be_grounded` (xfail) | Same class as T3. Suggested fix: ground the merged text against the union of old and new evidence. |
| T5 | Reconcile ops reference other users' ids or invented ids. | The model only sees short local ids (`"0"`, `"1"`, ...) mapped to same-scope neighbours (`Writer._scope_where`, `_decide`). Any other id resolves to `None` and the op becomes ADD. | `test_reconcile_cannot_reach_other_users_or_invented_ids`, `test_dedup_does_not_cross_users`, `test_dedup_neighbours_never_include_other_users` | None known. |
| T6 | **OPEN.** Any tenant rewrites, retracts, or erases a *shared* experience. | `policy.Policy.check_write` allows any principal when both `user_id` and `agent_id` are NULL, and every shared item is stored that way (`Sharer.share` calls `_manage_experience` with `None, None`). `Memory.update` then writes a new **AUTHORIZED public** version, which `check_action` allows. | `test_outsider_cannot_rewrite_shared_experience`, `test_outsider_cannot_retract_shared_experience`, `test_non_member_cannot_erase_team_experience` (all xfail) | **High.** This poisons the shared pool and upgrades authority, and it bypasses the privacy gate. Per-user API keys do not help, because the caller is legitimately themselves. Suggested fix: treat NULL-owner rows as admin-only for writes, and never derive AUTHORIZED from an update to a non-private row. |
| T7 | An operator edits the SQLite file (memory text, episodes, ledger rows). | `Memory.verify()` re-hashes the ledger chain, episode content, and memory text against the latest `text_sha256`. | `test_ledger_detects_tampering`, `test_ledger_detects_rewritten_history`, `test_episode_tampering_detected`; `verify()["ok"]` is asserted after every adversarial scenario | **Detection only.** Someone with write access can rewrite the whole chain consistently, because there is no external anchor. They can also change `authority`, `visibility` or `user_id`, which are not covered by `text_sha256`. Mitigate with OS permissions or disk encryption, and by publishing the ledger head externally. |
| T8 | Feedback poisoning: rating someone else's trace, or flooding feedback. | `Memory.feedback(user_id=...)` checks trace ownership. Each trace can be rated only once. The citation lock limits credit to ids in the trace. The ranker is deployed only if it beats relevance-only ranking at a 95% bound. | `test_feedback_on_another_users_trace_is_denied`, `test_feedback_citation_lock_and_single_rating`, `test_ranker_trained_on_uninformative_feedback_is_not_deployed` | Without `user_id`, the SDK call is unchecked, so the integrator must pass it. A user can still bias the ranking of their own memories. |

### R: Repudiation

| # | Threat | Mitigation | Tests | Residual risk |
|---|---|---|---|---|
| R1 | A write happens with no record of who made it. | Every write appends a ledger entry with actor, op, target, and a payload hash (`ledger.Ledger.append`). The append runs in a `BEGIN IMMEDIATE` transaction so concurrent processes can't fork the chain. | `test_ledger_chain_survives_concurrent_processes`, `test_threads_keep_ledger_and_isolation_intact` | The actor is whatever the caller claims (see S1). Grants and `MERGE` have no actor. |

### I: Information disclosure

| # | Threat | Mitigation | Tests | Residual risk |
|---|---|---|---|---|
| I1 | Another tenant retrieves the victim's memories through search, at any `k` or granularity. | `policy.read_filter` is compiled into SQL *before* vectors or BM25 are scored. `store.fts_query` also pins the FTS `owner` column to the caller's scope tokens. | `_assert_isolated` (run at k ∈ {1, 5, 50, 100} × 5 granularities in most adversarial tests), `test_private_isolation_across_users`, `test_fuzz.py::test_search_never_crosses_tenants` | None known. |
| I2 | Raw episodes leak, either directly or through the retry ladder (`source_episodes`). | `policy.episode_filter` applies to the raw channel and to every expansion step. | `test_raw_episodes_are_isolated_at_any_k` | None known. |
| I3 | FTS5 operator injection (`"`, `NEAR`, `*`, `OR`, `owner:`, `{col}:`) widens the scope or crashes the query. | `store.fts_query` keeps only `[A-Za-z0-9_]` tokens, quotes each one, ORs them inside `text : (...)`, and ANDs the result with `owner : (caller scopes)`. | `test_owner_column_filter_in_query_cannot_widen_scope`, `test_fts_query_quotes_every_term`, `test_hostile_queries_do_not_crash_or_leak`, `test_fuzz.py::test_fts_query_is_always_valid_fts5`, `test_victim_owner_token_in_query_does_not_widen` | Owner tokens are unsalted hashes of user ids, so anyone can compute them. That is fine because the SQL predicate is the real boundary. Non-ASCII words get no lexical channel (recall only, not security). |
| I4 | The compromised model is *shown* another user's data. | Prompts are built only from access-filtered rows: same-scope neighbours in reconcile, filtered hits in answer. | `MaliciousLLM.saw` assertions in `test_reconcile_cannot_reach_other_users_or_invented_ids`, `test_answer_strips_citations_never_shown`, `test_hostile_model_battery_keeps_all_invariants` | The model provider sees everything the caller can see. Choose a provider accordingly. |
| I5 | An answer cites ids it was never shown, such as another user's memory id. | Citation lock: only labels from this answer's rendered context map to ids (`Answerer.answer`). | `test_answer_strips_citations_never_shown`, `test_answer_citation_lock_and_abstention` | The answer *text* is model output and can say anything. It cannot contain other tenants' data, because the model never saw any (I4). |
| I6 | Shared experiences leak PII or the owner's private values. | Preferences can never be shared (`set_shareable`, `check_visibility`). After the model abstracts the content, a deterministic gate checks it for PII regexes and for content-word n-gram canaries from the owner's own memories and turns (`privacy.check_shareable`). Shared items are UNENDORSED, carry no evidence, and record only an HMAC pseudonym. | `test_share_abstraction_leak_is_blocked_by_deterministic_gate`, `test_shared_items_never_carry_authority_or_evidence`, `test_privacy_gate_blocks_pii_and_canaries`, `test_preferences_cannot_be_shared` | Regexes and canaries are heuristics: paraphrased values ("a city in Bavaria") pass. |
| I7 | **OPEN.** The shared-pool manager's output skips the privacy gate. | The gate runs on the abstracted `content`. `Learner._manage_experience` then stores the manager's `content` (`final`) without checking it again. | `test_shared_pool_manager_output_is_also_gated` (xfail) | **High** once the pool is non-empty. Suggested fix: run `check_shareable` on `final` inside the shared branch. |
| I8 | **OPEN.** A non-member reads a retracted or superseded *team* experience through `get` or `history`. | `Memory.get` only denies non-active rows when they are `private`. | `test_non_member_cannot_read_retracted_team_experience` (xfail) | Medium: needs the id, but `history` also walks the version chain. |
| I9 | Probing someone else's memory ids through the action gate. | `check_action` reports memories the caller can't read as `missing`. | `test_action_gate_does_not_leak_other_users_memories` | None known. |
| I10 | Data persists after erasure. | `forget` hard-deletes rows, FTS entries, evidence, outcome rows, and traces, rebuilds FTS, runs `VACUUM`, and truncates the WAL (`Sharer.forget`, `store.scrub`). The ledger keeps only hashes. | `test_forget_user_erases_and_withdraws`, `test_forget_erases_outcome_rows` | Backups, model-provider logs, and the `ResponseCache` file are out of scope. The cache stores prompts verbatim, so disable it or protect it. |
| I11 | Query vectors in `traces` and `outcomes` reveal what a user asked. | Traces are deleted when the user is forgotten. Outcome rows are deleted with their memory. | `test_forget_erases_outcome_rows` | Readable by anyone with DB access. Embeddings can be partly inverted. |

### D: Denial of service

| # | Threat | Mitigation | Tests | Residual risk |
|---|---|---|---|---|
| D1 | Malformed model output (non-JSON, wrong types, `null`) crashes `add`, `answer`, `learn`, or `share`. | `parse_json` with one repair attempt. `LLMError` falls back to raw turns, ADD, abstention, or rejection. | `test_malformed_extraction_output_is_survived`, `test_malformed_reconcile_output_is_survived`, `test_malformed_answer_output_is_survived`, `test_malformed_share_output_shares_nothing`, `test_fuzz.py::test_parse_json_only_raises_llm_error` | **OPEN** for valid JSON of the wrong *type*. See D2. |
| D2 | **OPEN.** Well-formed JSON of the wrong type raises out of the public API. | Not mitigated. Crashing inputs: `{"memories": 7}`, an int `entities` field, `{"operations": 5}`, `"operations"` as a string or a list of non-dicts, `"fact": 1e999`, `{"selected": 5}`, a list returned by the experience manager. | `test_extraction_non_iterable_fields_do_not_crash`, `test_reconcile_non_dict_ops_do_not_crash`, `test_answer_non_iterable_selected_does_not_crash`, `test_experience_manager_list_output_does_not_crash`, `test_fuzz.py::test_arbitrary_extraction_json_never_crashes` (all xfail) | An injected tool message can make every `add()` that includes it fail, though the episode is already stored. Suggested fix: validate model output with pydantic schemas. |
| D3 | **OPEN.** An out-of-range time (`"when": 1e300` from the model, or `timestamp: 1e300` or `-1e18` from the caller) is stored, then crashes `_fmt` and `time.gmtime`. | Not mitigated. The episode is committed before extraction fails, and every later `search().context()` or `answer()` for that user raises. | `test_out_of_range_when_does_not_poison_search`, `test_rest_out_of_range_timestamp_is_rejected_and_does_not_poison` (xfail) | **Persistent per-user DoS.** In open mode or with an admin key, any caller can trigger it for any user. Suggested fix: clamp or reject timestamps outside years 1 to 9999 in `write._ts`. |
| D4 | **OPEN.** Message shapes the REST layer doesn't expect (`messages: 123`, `[123]`, `null`, `[null]`, a non-string role, a non-list trajectory) return HTTP 500. | `AddBody.messages` and `LearnBody.trajectory` are typed `Any`. `normalise_messages` and `_role` assume the right types. | `test_rest_bad_message_shapes_are_4xx`, `test_rest_bad_trajectory_is_4xx` (xfail). Other hostile bodies return 4xx: `test_rest_hostile_bodies_never_500`, `test_rest_malformed_json_and_ids` | Low (the error is noisy, not unsafe). Suggested fix: a strict pydantic union for `messages`. |
| D5 | Huge inputs: 1 MB messages, 1 MB model text, 300 memories in one call, 10k citations. | Entities are capped at 8, FTS terms at 32, citations deduplicated through the label map. The server enforces a body-size limit and a rate limit. | `test_giant_and_numerous_llm_outputs`, `test_hostile_message_content_is_stored_safely[...len1000000]` | There is no cap on how many memories one extraction can create, or on memory text length. A hostile model can bloat a user's store. |

### E: Elevation of privilege

| # | Threat | Mitigation | Tests | Residual risk |
|---|---|---|---|---|
| E1 | A low-authority memory binds a protected action. | `check_action` allows only active AUTHORIZED memories. | `test_action_gate`, `test_tool_claim_disguised_as_user_preference_stays_unendorsed`, `test_shared_items_never_carry_authority_or_evidence` | Breaks transitively through T3, T4, and T6. |
| E2 | Consolidation raises authority (merge or summary). | `Authority.weakest` for merges and summaries. Dedup keeps the higher-authority row and adds evidence without upgrading. `_confirm` upgrades only on a direct restatement from a higher-authority source (the AuthMem H+ case). | `test_user_restatement_legitimately_raises_authority`, `test_hostile_model_battery_keeps_all_invariants` | A user restating a tool claim upgrades it on purpose. |
| E3 | The owner-only write rule is bypassed. | `policy.check_write` for `update`, `delete`, `forget`, and `set_shareable`. | `test_manual_update_and_delete_by_owner_only`, `test_lookalike_user_ids_are_distinct_tenants` | **OPEN** for NULL-owner (shared) rows. See T6. |
| E4 | A user key escalates to admin or to another user. | `auth.lookup` gives keys a fixed role, user, and agent. `scope()` returns 403 on mismatch. Admin-only endpoints are checked by `admin_only`. | `test_auth.py` | Admin keys are all-powerful, so treat them like root credentials. |

## 6. REST identity model

Per-user API keys were added alongside this document. The intended model:

- **User-scoped key.** Bound to exactly one `user_id`, and optionally one `agent_id`, when it
  is created (`veramem keys create`). Only its SHA-256 hash is stored. A request may leave
  out `user_id`, in which case the key's user is used. If the request names a different
  user or agent, it gets a 403. A user key can never act for anyone else, and it cannot
  call admin endpoints (grants, consolidate across users, verify, ranker training).
- **Admin key.** `VERAMEM_API_KEY` or a key created with `role="admin"`. May act for any
  `user_id`. It is meant for a trusted backend that authenticates end users itself and
  passes their ids through. Never ship it to a client.
- **JWT / OIDC.** `sub` becomes the user, with the same pinning as a user key.
- **Open mode** (nothing configured). Every caller is anonymous and `user_id` is trusted.
  This is allowed only on loopback.

Even with per-user keys, T6, I7, and I8 are reachable by *legitimate* tenants, because
those bugs are in the authorisation rules for shared rows, not in identity.

## 7. Explicitly out of scope / not mitigated

1. **Write access to the DB file.** The ledger detects edits but cannot prevent them (T7).
   Columns that `text_sha256` doesn't cover (`authority`, `visibility`, `user_id`, `status`)
   can be changed without detection.
2. **Prompt injection inside a user's own messages** that affects that same user's
   extraction. The extractor reads those messages with user authority, so injected
   instructions in them become AUTHORIZED memories. T3 widens this: injection in a tool turn
   can also borrow a user quote from the same batch.
3. **Role labelling by the integrator.** If you pass tool output as `role="user"`, it gets
   user authority. The same applies to the MCP `remember(role=...)` argument (S4).
4. **The model provider** sees every prompt (I4), including everything the caller is
   allowed to read.
5. **Transport security.** Terminate TLS in front of `veramem serve`.
6. **Semantic leaks through shared experiences** that pass the regex and canary gate (I6).
7. **Resource exhaustion by a hostile model** within one user's own store (D5).

## 8. Running the adversarial suite

```bash
.venv/bin/pytest tests/test_adversarial.py tests/test_fuzz.py -q -rxX
```

`test_fuzz.py` needs `hypothesis` (tested with 6.170.1) and skips itself when it is not
installed. Every `xfail(strict=True)` test above corresponds to an **OPEN** entry. Fixing
the bug turns the test into an XPASS, which strict mode reports as a failure. When that
happens, remove the marker and update this document.
