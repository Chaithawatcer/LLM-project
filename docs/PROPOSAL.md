# Proposal - Cyber Advisor

**Problem and audience.** Security analysts and IT staff who must decide what to patch first need two different kinds of answers: *facts about a list* ("how many Microsoft entries are in CISA's Known Exploited Vulnerabilities catalog? which vendors have ransomware-linked entries?") and *guidance from authoritative documents* ("what does NIST recommend for emergency patching, or for assets that cannot be patched?"). A plain RAG chatbot answers the second kind but cannot count or rank rows, and a plain chat model invents both. Cyber Advisor is an agent that routes each question to a cited document search, to guarded text-to-SQL over the KEV table, to both, or declines when its sources cannot answer.

**Course techniques.** RAG with citations and a `min_score` floor (L5-L6); hybrid BM25 + dense retrieval and a retrieval evaluation harness (L8); structured/validated output via Pydantic (L2); an agent loop with native tool calling, a step cap and a duplicate-call guard (L10); agentic RAG with text-to-SQL and an engine-level SQL guard (L9, L11); a FastAPI + SSE backend (L3-L4). Tool results are treated as untrusted input (L14).

**Data.** CISA KEV catalog (CSV, CC0, 1 754 rows) and 12 NIST publications (public-domain US-government PDFs). Raw data and the fetch script are committed.

**Evaluation.** 36 hand-checked questions (12 document, 11 SQL, 6 mixed, 7 that must be declined). Metrics: chunk-level Hit@k and MRR for retrieval (baselines: BM25-only and the reference app's naive 800-character windows with a 0.40 floor); execution-match accuracy for SQL (baseline: documents-only RAG, which cannot count); tool-routing accuracy; refusal accuracy and false-refusal rate on a `min_score` sweep. Reported with the sample-size caveat.

**Scope.** Small and finished: no browser automation, no code execution, no MCP. If time allows, an MCP server exposing `query_kev` is the natural extension.
