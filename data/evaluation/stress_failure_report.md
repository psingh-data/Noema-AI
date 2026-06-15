# Noema Stress Test Failure Report

Generated: 2026-06-16

Source artifact: `data/evaluation/eval_report.json`

## Summary

The latest stress test shows that Noema's internet and research systems are
working well enough for the current phase. The main quality gap is routing:
messages with multiple intents are still sometimes collapsed into a single
simple mode.

Latest baseline metrics before this routing pass:

| Metric | Result |
|---|---:|
| Intent accuracy | 67.24% |
| Emotion accuracy | 42.10% |
| Internet routing accuracy | 83.10% |
| Research routing accuracy | 98.22% |
| Safety routing accuracy | 100.00% |
| Advice answer rate | 100.00% |
| Decision answer rate | 100.00% |
| Over-reflection rate | 0.00% |
| Bad phrase rate | 0.00% |

## Priority Failures

1. False-positive safety routing

Risk: ordinary phrases such as "I can't go on with this assignment" can be
mistaken for self-harm language if the safety layer only checks keywords.

Fix direction: keep direct self-harm language conservative, but add ordinary
context exclusions for ambiguous phrases.

2. Decision support avoidance

Examples:

- "Should I talk to HR or take a break? What should I do?"
- "Should I change company or switch careers? What should I do?"

Risk: generic advice markers such as "what should I do" can override the named
choice. This makes Noema give advice instead of comparing options.

Fix direction: named decisions must outrank generic advice.

3. Cognitive distortion detection

Examples:

- "If this fails, my whole future is ruined."
- "I always mess everything up."
- "Nobody cares about me."

Risk: practical-advice wording can hide the cognitive pattern, so Noema misses
the chance to gently challenge catastrophizing, mind reading, or all-or-nothing
thinking.

Fix direction: strong cognitive distortion signals should outrank generic
advice unless the user has named a concrete decision.

4. Grief + advice mixed-intent routing

Examples:

- "I miss my sister so much today. Please be practical."
- "A memory of my girlfriend hit me and I broke down."

Risk: Noema may treat grief language as ordinary advice or ordinary reflection,
losing the emotional context.

Fix direction: detect grief-like language even when the user asks for practical
help, and keep the response grief-aware.

5. Multi-intent reasoning

Stress pattern:

The user mentions several life areas at once, expresses emotional weight,
uncertainty about the future, and asks for advice.

Fix direction: add `mixed_complex_life_problem` for messages with 3+ major
topics, emotional content, future uncertainty, and explicit advice request.

## New Intent

`mixed_complex_life_problem`

Trigger:

- 3+ major topics
- emotional content
- future uncertainty
- explicit advice request

Required response:

1. Summarize the situation.
2. Separate issues.
3. Prioritize.
4. Give a recommendation.
5. Ask one question maximum.

## Scope Boundary

No new internet, research, memory, UI, or multi-agent features were added in
this pass. This is a routing-quality correction only.

## Post-Fix Verification

After the routing-quality pass:

| Metric | Result |
|---|---:|
| Intent accuracy | 72.52% |
| Emotion accuracy | 42.03% |
| Internet routing accuracy | 83.50% |
| Research routing accuracy | 98.22% |
| Safety routing accuracy | 100.00% |
| Advice answer rate | 96.52% |
| Decision answer rate | 100.00% |
| Over-reflection rate | 0.00% |
| Bad phrase rate | 0.00% |

Targeted test coverage was added for:

- ordinary-context safety false positives
- named decisions overriding generic advice
- cognitive distortions overriding generic advice
- grief plus advice staying grief-aware
- `mixed_complex_life_problem` response structure
