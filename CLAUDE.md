# ivrit-py

## Agent-Based Workflow

This project is built around five specialized subagents in `.claude/agents/`. They are the canonical source of truth for how work is done here.

| Agent       | Purpose                                                            |
|-------------|--------------------------------------------------------------------|
| `visionary` | Strategic alignment for direction-changing decisions               |
| `architect` | Technical design when components/boundaries/dependencies change   |
| `coder`     | Implementation following project conventions                       |
| `reviewer`  | Final quality gate — NO HACK, NO OVER-ENGINEERING                 |
| `manager`   | Completion confirmation and credit attribution                     |

## Workflow

For any non-trivial change, dispatch the subagents in this order, skipping any whose "When to Invoke" criteria don't apply:

1. **visionary** — only for feature proposals, strategic pivots, or long-term architectural decisions.
2. **architect** — when adding components, changing boundaries, introducing dependencies, or changing interfaces.
3. **coder** — for all implementation work.
4. **reviewer** — mandatory before declaring any change done. Must return APPROVED.
5. **manager** — only after reviewer approves. Produces the completion summary.

If reviewer requests changes, push back to architect or coder as indicated and re-run the affected phases.

## Ensuring the Right Agent Runs

- **Trust the descriptions.** Each subagent's frontmatter includes `PROACTIVELY` / `MUST BE USED` triggers so Claude Code auto-dispatches them. Don't paraphrase their work inline — delegate.
- **reviewer is non-optional.** No change is complete without a `REVIEW` block returning APPROVED. If you're about to report "done" without one, stop and invoke reviewer.
- **manager closes every workflow.** After reviewer approves, invoke manager to emit the completion block.
- **One agent at a time.** Wait for each subagent's output block before dispatching the next; later agents depend on the previous block as input.
- **When unsure which agent applies**, default to the more senior one (visionary > architect > coder) and let its decision framework decide whether to proceed or downscope.
- **Manual override.** You can always force a specific agent with phrasing like "use the reviewer subagent on this change" — useful if auto-dispatch misses.

Run `/agents` to confirm all five subagents are registered.

## ARCH.md

`ARCH.md` is the source of truth for the package architecture: module layout, public API, abstractions, engines, data flow, and intentional design choices (e.g. "ffmpeg is the only decode path, torchaudio is deliberately avoided").

- **Before any code change**, read `ARCH.md` in full. Do not rely on memory or guesses about the architecture.
- **After any change** that affects what `ARCH.md` documents (modules, classes, public functions, engines, API surface, data flow, design choices), update `ARCH.md` in the same change. A change that breaks `ARCH.md`'s accuracy is incomplete.
- The **architect** subagent is responsible for deciding whether `ARCH.md` needs an update.
