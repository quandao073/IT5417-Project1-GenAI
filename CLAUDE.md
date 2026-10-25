# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Read WORKLOG First

**`WORKLOG.md` is the project's context file. Read all of it before changing anything**
(it is capped at 150 lines, so this is cheap).

**WORKLOG and `docs/scope-and-non-goals.md` override the plan document.**
`docs/ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md` was written before the data was
measured and before the scope changed; it is history only. Don't implement from it.

### Keep it current and short

Update WORKLOG when you:
- finish a phase, or a self-contained chunk of one
- make, change or reverse a decision
- hit a trap that cost time, or would cost the next agent time
- resolve something listed under *Đang chờ quyết định*

Do **not** update it for routine edits, added tests, or no-op refactors.

Rules:
- **Edit in place, don't append.** Rewrite *Trạng thái hiện tại* in the same edit; a stale
  status is worse than none, because it is trusted
- **Delete what no longer holds**: superseded decisions, fixed traps, closed questions.
  Don't strike them through or keep them "as history"; git keeps history
- No journal: `git log` covers what changed and when; WORKLOG covers what is true now and why
- Stay under 150 lines; if an edit would cross that, cut something stale first. One line per
  decision or trap where possible; details live in the spec or doc it points to
- A measured number goes in `docs/measured-facts.md` (append-only) with the method used to
  measure it, never copied from a paper or from the plan

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.