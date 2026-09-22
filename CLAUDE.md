# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Read WORKLOG First

**`WORKLOG.md` is the project's context file. Read it before changing anything.**

Minimum: *Trạng thái hiện tại*, *Quyết định đang có hiệu lực*, *Bẫy đã biết*.
Those three sections are enough to avoid breaking something.

**WORKLOG overrides the plan document.** `docs/ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md`
was written before the data was measured, so several of its requirements have been
deliberately overridden. Implementing a plan section verbatim without checking
*Quyết định đang có hiệu lực* will produce wrong work.

### Update it when something significant lands

Log it when you:
- finish a phase, or a self-contained chunk of one
- make, change or reverse a decision — especially one that deviates from the plan
- establish a fact by measuring real data
- hit a trap that cost time, or would cost the next agent time
- resolve something listed under *Đang chờ quyết định*

Do **not** log routine edits, added tests, or no-op refactors. `git log` already
covers what changed; WORKLOG covers **why it is the way it is**.

Rules for an entry:
- Newest first, dated, at most ~10 lines
- Answer *why*, not *what*
- **Update *Trạng thái hiện tại* in the same edit.** A stale status header is worse
  than none, because it is trusted
- A measured number goes in *Sự thật đã đo* with the method used to measure it,
  never copied from a paper or from the plan

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