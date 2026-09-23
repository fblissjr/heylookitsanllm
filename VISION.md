# Vision

heylook is a personal inference server for one person's Mac. It runs the
models the owner wants to use, the day they come out, and makes them easy to
reach from a phone, a browser and the owner's other apps. It is a daily
driver and a workbench at the same time.

## What it is for

- Running current open models locally on Apple Silicon, with vision and
  thinking as the normal case, not the special one.
- Being the one place the owner's own tools call for inference, through one
  well-defined API.
- Trying things: a new model, a new quant, a new engine build, a new idea,
  with a short path from download to answer to measurement.

## What it is not

- A multi-user or hosted service.
- A model zoo that implements architectures. Model code belongs upstream.
- A compatibility layer for every client protocol. One wire, done well.
- A place to preserve old behaviour or old data for its own sake.

## Principles

1. **Nothing happens where it cannot be seen.** Every setting in force can be
   explained: its value, where it came from, and what it would be if left
   alone. Every request can say what it cost and what it reused. A silent
   fallback is a bug even when the answer looks right.
2. **Derive, don't copy.** Facts come from where they live: the model's own
   files, its template, the engine's own report. A hand-kept list of anything
   derivable will drift.
3. **Evidence over belief.** Claims about speed, quality or behaviour are
   measured through the real path, with their conditions written down. A
   measurement without its conditions is an anecdote.
4. **Enforce, don't remind.** A rule that matters is a check that runs.
   Anything else is a hope.
5. **Both engines are first class.** MLX and llama.cpp each do something the
   other does not. The user sees one server, one set of controls and one
   report, whichever engine is behind a model.
6. **Follow upstream, don't fork.** Pin exact versions, contribute fixes back,
   and keep the local layer thin. The ecosystem moves faster than any fork can.
7. **Simple where it can be, with judgment.** Remove what is dead or
   duplicated. Keep what earns its place, even when removing it would be
   tidier.
8. **The interface tells the truth.** It discloses cost rather than asking
   for confirmation, and interrupts only before something would be lost. What
   works on the desktop works on the phone.
9. **Private by default.** Conversations live on this machine. Nothing is
   logged or sent anywhere unless the owner turns it on.

## Where it is going

Toward one engine-neutral server: the same contract, cache behaviour and
reporting whether a model runs on MLX or llama.cpp, with less code on each
side of that line, not more. Toward models that are fast in real use, which
means reusing what was already computed (multi-turn context, long system
prompts, images) rather than chasing raw throughput. And toward a codebase
small and legible enough that one person, helped by agents, can keep it
honest.
