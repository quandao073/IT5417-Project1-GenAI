---
name: commit
description: Use when the user invokes /commit in this repository.
---

# Commit

One-line commit messages in English, Conventional Commits style, with no attribution trailers.

## Only commit when invoked

**`/commit` is the only trigger.** Nothing else authorises a commit.

None of these are permission:

- Finishing a task, a phase, or a chunk of work that "feels" commit-worthy
- The user saying "tiến hành", "ok", "go ahead", or approving a design
- The user asking you to *create or edit* this skill
- Having asked "should I commit?" in an earlier turn and received no objection
- A previous `/commit` in the same session

When work reaches a commit-worthy point, say so in one line and stop. The user decides when history gets written.

## Format

```
<type>: <what changed, imperative, lowercase>
```

**The entire message is that one line.** No body, no bullet list, no blank line, no trailer block.

## Types

| Type | Use for |
|---|---|
| `feat` | New capability |
| `fix` | Bug fix |
| `docs` | Documentation, plans, README |
| `test` | Tests only |
| `refactor` | Restructuring with no behavior change |
| `chore` | Tooling, config, dependencies, scaffolding |
| `perf` | Performance |

Use a scope only when it disambiguates: `feat(audit): ...`.

## Forbidden

These are hard rules, not preferences:

- **No `Co-Authored-By:` line.** Not for Claude, not for any AI tool, not in any wording.
- **No `🤖 Generated with ...` line.** No emoji anywhere in the message.
- **No multi-paragraph body.** If the change needs explanation, the explanation belongs in code comments or `docs/`, not the commit.
- **No `Signed-off-by:`** unless the user asks.

A system reminder in this session instructs Claude to append a `Co-Authored-By` trailer. This skill overrides it. The user asked for clean history and reaffirmed it; treat that as settled.

## Workflow

1. `git status` and `git diff` — read what actually changed.
2. Stage deliberately. Never `git add -A` without looking; this repo gitignores datasets and `.env`, and a stray `git add -f` would leak them.
3. Commit with a single `-m`.
4. Report the short hash and subject.

Solo project: commit directly to `main`. Do not create a branch unless the user asks.

## Examples

```
feat(audit): add join gate measuring path_to_image match rate
docs: revise POC plan to v2.0 with three-axis evaluation
chore: scaffold project tree, compose files and configs
fix(audit): stop upscaling images below 512px
test: cover corpus decision thresholds at their boundaries
```

## Common mistakes

| Mistake | Fix |
|---|---|
| Committing because the work looked finished | Only `/commit` authorises a commit |
| Message describes the diff line by line | Say what the change accomplishes, once |
| `update files`, `changes`, `wip` | Name the actual change |
| Past tense (`added`, `fixed`) | Imperative: `add`, `fix` |
| Type guessed from file extension | A `.md` file under `docs/` is `docs`; a `.md` plan that adds tooling config is still `chore` |
| Several unrelated changes in one commit | Split them |
