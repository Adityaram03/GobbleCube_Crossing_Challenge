# Gobblecube AI Builder Take-Home

We're hiring **AI Builders**: engineers who pick up unfamiliar problems,
figure them out with whatever tools help, and ship something that works.
Pick one of the two challenges below, build a working Dockerized
submission, and send us the repo link when it's something you'd put
your name on.

## The two challenges

- **[The ETA Challenge](./eta-challenge-starter/):** ride-hailing ETA
  prediction on public NYC taxi data. Regress a single number: trip
  duration in seconds. Scored on MAE against a held-out 2024 slice.
  The repo baseline lands at ~367 s there.
- **[The Crossing Challenge](./crossing-challenge-starter/):** pedestrian
  crossing-intent + 2-second trajectory for a slow-speed autonomous
  delivery vehicle. Scored on a joint BCE + pixel-ADE composite, each
  term normalized so "do nothing" = 1.0. The repo baseline lands at
  0.74 there.

Pick **one**. One submission per candidate.

## How grading works

The two challenges have **separate leaderboards**. You are not ranked
against candidates who picked the other one. A strong Crossing submission
and a strong ETA submission are treated as equivalent signal for the role.

Each starter README explains what its scoring harness does. Beat the
baseline by as much as you can; we'll tell you how it stacks up.

## Same rules either way

- Submit a **public GitHub repo** containing `predict.py`, a `Dockerfile`,
  your trained weights, and a README. Details and constraints (image size,
  runtime limits, disqualifiers) live in each starter's README. Read the
  one you pick carefully.
- Use whatever AI tooling helps you ship. Claude Code is our in-house
  favourite, but Codex, Cursor, Copilot, plain-API integrations, or no LLM at all are
  fine. We're scoring the submission, not the toolchain. Your git
  history is part of what we look at.
- No external API calls at inference time, no collaborators, no training on
  the Eval set.
- Include the Claude.md/Agents.md and relevant markdown files with the submission

## What we actually care about

A clean submission with an honest README will beat a slightly better score
with no write-up. We read three things, roughly in this order:

1. **Do you ship?** The number on the leaderboard.
2. **Can you learn fast?** Your git log shows the trajectory. First
   commits rarely look like final ones.
3. **Can you reason about a problem that wasn't handed to you as a spec?**
   Your README explains what you tried, what failed, and what the next
   experiment would be if you kept going.

Submit your repo URL and LinkedIn profile to agentic-hiring@gobblecube.ai.