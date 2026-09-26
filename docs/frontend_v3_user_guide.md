# v3 user guide

last updated: 2026-09-05

How the UI actually behaves, written for the person using it rather than
the person maintaining it. Where behaviour is surprising, this says so rather
than smoothing it over.

**Scope.** Chat, presets and settings in depth — that is where the state model
lives and where nearly all the confusing behaviour is. The other pages get an
orienting paragraph each. For the build contract see
[frontend_v3_spec.md](./frontend_v3_spec.md); for the developer map see
[frontend_v3.md](./frontend_v3.md).

---

## 1. The one thing to understand first

Almost every "why did it do that?" in this app comes from the same source:
**your settings live in three places at once.** The drawer now names which one
you are editing, but the three layers are still the thing to hold in your head —
they are why applying a preset, editing a knob, and saving a preset each affect
a different thing.

| Layer | Where it lives | What it is for | Changes when |
|---|---|---|---|
| **The draft** | this page, in memory | what the next conversation is created from, when none is open | you move a slider, type a prompt or Apply a preset with no conversation open |
| **The conversation** | the server, on the conversation row | what actually gets sent to the model | you move a slider with a conversation open |
| **The preset** | the server, in a named store | a snapshot you can copy into a conversation, or copy a conversation into | only when you press **Save** or **Save as new** |

The system prompt has the same three layers: a draft parked in your browser
before any conversation exists, the conversation's own prompt, and the prompt a
preset carries. **Clear**, under the prompt box, removes the prompt in one step
(it asks *Clear prompt?* first, because the text is typed work).

**A blank sampler field shows the value it will actually use** (v2.0.21). Leaving
Top-k empty on a gemma model shows that model's own value in grey, because that is
what its config asks for — not the word "auto", which told you nothing. Under the
field's name, small text says where that value comes from: *vendor default* (the
model publisher's), *model file default* (this model's heylook settings) or
*heylook default* (the server's fallback). The moment you
type your own value, the field's name turns accent-coloured, its border matches, and
a small reset control appears beside it that puts the field back to the model's
value. Nothing appears next to fields you have not touched, so the panel stays quiet
and the only controls on screen are the ones you changed. Presets still store "auto"
for a key you never set, and auto still means "use whatever this model resolves to" —
that is why one preset behaves correctly across models with different defaults. A few
fields have no server-side answer (Seed, for one) and honestly still read `auto`.
Turning Thinking on or off does not change any of these numbers. It did until
v2.0.32, when enabling thinking also applied a presence penalty; that overlay was
removed, and the switch now travels alone.

Two consequences worth internalising:

- **The sampler panel is the active conversation's settings**, and it says so:
  under the *Generation* heading it reads **"Applies to this conversation —
  changes save as you make them."** With no conversation open it reads
  **"The next conversation you start is created with these."**, because that
  is what the panel is then: the next conversation is created from exactly
  what the drawer shows. It starts blank, never from the conversation (or
  notebook) that was open before.
  Selecting a different conversation replaces every value in the panel with
  that conversation's stored ones. If you had "temperature 1.3" on screen and
  you click another conversation, the 1.3 is not lost — it is still on the
  first conversation — but the panel now shows something else.
  **Clear all overrides** empties every field, which hands each value back to
  the server's own defaults. On an open conversation that is a change to *that
  conversation*.
- **A preset is a copy, never a link.** Applying one copies values in. Editing
  them afterwards does not change the preset. Changing the preset does not
  change any conversation that was made from it. There is no live binding
  anywhere in this app.

---

## 2. Presets

### What a preset holds

A name, a system prompt (optionally), and the whole sampler panel — temperature,
max tokens, top-p, top-k, and the advanced knobs including thinking and thinking
depth.

Presets are global. They are not per-conversation, per-model, or per-page; chat
and the notebook share one store.

A preset with no system prompt is listed as **"<name> — settings only"**. That
is not an error state — it is a preset that carries sampler values and makes no
claim about the prompt (see *Apply*, below).

### Seeing what is in one

Pick it from the **Preset** dropdown in the settings drawer. The section shows a
collapsed **"<name> system prompt"** block — that is the preset's own text,
read-only.

This matters because directly below the preset section is a **System prompt for
this conversation** box, and that box shows *the conversation's* prompt no
matter which preset is selected. Before the preview existed, clicking through
presets appeared to show the same prompt every time, because you were reading
the conversation's prompt under a dropdown you had just changed.

### Apply

Copies the preset onto the current conversation: sampler values, and the prompt
**if the preset carries one**.

A preset with an empty system prompt makes no claim about the prompt. Applying
it leaves the conversation's prompt exactly as it was. Empty never means "set it
to empty" — this is deliberate, so a preset saved without a prompt can never
silently wipe typed work.

Apply asks for confirmation ("Replace prompt?") only when it would overwrite a
prompt you have with a different one. Sampler values are trivially recoverable;
a system prompt is typed work.

Apply also **stamps** the conversation: from then on the conversation records
that it is running that preset, and the dropdown will open on it next time.

### Save and Save as new

Both buttons write the current conversation's prompt and the entire sampler
panel to a preset. They differ only in *which* preset. (Save was labelled
**Update** before v1.79.62; nothing about what it does changed.)

There are **two** write buttons, and which preset you hit is decided by which
one you press — not by what you type:

- **Save** overwrites the preset showing in the dropdown, the one the preview
  directly above it is displaying. This is the only way to overwrite a preset.
- **Save as new** creates one under the typed name. If that name is already in
  use it refuses and tells you to use Save instead. It can never overwrite.

Save asks for confirmation before it changes a preset's stored prompt, except
in one case:

- **No confirmation** when you are updating the preset the conversation is
  already running. This is the ordinary loop — apply a preset, tweak the prompt,
  update it — and making you confirm it every time would train you to click
  through the confirmation that matters.
- **Confirmation** for everything else that would change a stored prompt,
  including *blanking* one. Updating from a conversation with no system prompt
  writes an empty prompt to the preset, and a preset with no prompt does nothing
  when applied. It stays in the list but stops working, which reads as "my
  preset disappeared."

The confirmation is the button itself changing to **Overwrite prompt?** for a
few seconds. Click it again to go through. Changing the dropdown cancels it, and
so does editing the prompt underneath — the confirmation is a promise about one
specific write, and if the write changes, the promise is void.

### If you change a setting and never apply or save

Nothing is lost and nothing is hidden. The change is on **the conversation**
immediately (a short debounce, then a write to the server) and it is what the
model receives on your next message.

The preset is untouched. The drift line under the dropdown says which
situation you are in, which half moved, and which knobs:

> *Matches current settings.*
> *Prompt differs from "p2" — Apply loads the preset's prompt here; Save writes this one into the preset.*
> *Settings differ from "p2" (temperature, max tokens) — Apply loads the preset's values here; Save writes yours into the preset.*
> *Prompt and settings differ from "p2" (…) — …*

Read that as: your conversation and this preset have diverged. **Apply** discards
your changes in favour of the preset's. **Save** discards the preset's in
favour of yours. There is no merge and no third option.

With **no conversation open** a preset you have only selected is not used —
the next conversation is created from what the drawer shows, so the line says
*"Not applied — the next conversation starts from what is shown here. Apply
loads "p2"."* Apply it and it drifts like anywhere else.

The *Preset* heading names what the open conversation is running, which is not
always what the dropdown shows (the dropdown is for browsing): *Preset · p2*,
*Preset · p2, edited*, or *Preset · none applied*. Beside the model selector, a
chip says the same and appends **(edited)** once anything the preset speaks for
has changed.

### Delete

Removes the preset. Conversations that were running it keep their prompt and
settings — they are copies — they simply stop showing a preset name.

---

## 3. Conversations

**New conversation**, pressed from an open conversation, starts from the
selected preset if there is one (its prompt, its settings, and its stamp). With
no preset selected it starts blank: nothing of the open conversation, prompt or
settings, carries over. With no conversation open it is created from exactly
what the drawer shows, including a cleared prompt.

**Switching** replaces the panel, the system prompt box, the model selector and
the message list with the selected conversation's own. Staged attachments are
cleared — they belonged to the conversation you picked them in.

**The model is per conversation.** Selecting a model saves it onto the open
conversation. Switching models mid-conversation is allowed and disclosed rather
than blocked:

- Images or audio already in the thread that the new model cannot read are
  **dropped from the request**, with a note on each affected message. The
  conversation keeps them; the model just never sees them. This is the one
  thing that stops to ask (Cancel / Switch anyway).
- Losing **thinking** is stated in the status line, not confirmed — a capability
  going away destroys nothing, so it is disclosed rather than gated.
- Attachments you have staged but not yet sent **block** the switch instead.
  The asymmetry is deliberate: history is already committed and dropping it is
  reversible by switching back, whereas a staged file silently vanishing from a
  message you are composing is work you did not know you lost.
- Load cost is disclosed, never confirmed. If the model is not resident you are
  told your first message will load it, and a **Load** button lets you pay that
  cost now. Choosing a model *is* choosing to pay for it.
- **Context size** (gguf models only): a select beside the model picker chooses
  the context the next load runs with. **Auto** is the default and is
  llama-server's own answer, sized from the model and fitted to memory; once
  the model is resident the Auto entry shows the number it chose. The other
  entries are power-of-two steps up to the model's training context, marked
  *(max)*. Picking a different value on a resident model turns Load into
  **Reload**, which restarts that model at the new size; the choice is saved as
  the model's `ctx_size` so the Models page shows the same number. MLX models
  have no fixed context allocation, so the control does not appear for them.
- **Flash attention** (gguf models only): a second select beside it, reading
  "flash attn: auto (on)" once the model has loaded, where the part in
  brackets is what llama-server's auto chose. Leave it on auto; *on* and
  *off* exist to test a new architecture. Like the context size, a change
  shows Reload and is saved with the model. On a desktop, hover it for where
  the value came from; the Models page's engine panel shows the same on any
  device.

**Rename** is inline on the sidebar row. **Clone** copies a conversation and its
messages. **Delete** removes it; deleting one that is generating stops the run
first.

---

## 4. Generating

### Sending

The request is built **on the server** from the stored conversation, not from
what the page happens to be holding. Your prompt keystroke, your slider moves and
your preset apply are all written to the store before the generation reads it.
This is why a settings change takes effect on the next message with no explicit
"apply" step.

On a phone the composer is two rows: the message field spans the full width,
and attach, thinking, prompt preview and Send sit on a row under it. The top
bar is likewise two tiers, with Chats, the model and the gear on the first
line and the context, Load and prompt chips on the second. The same controls
are present as on desktop; only the arrangement changes.

### The wait before the first token

The status line names it. On a model that is not resident this is a multi-GB
load and the single longest wait in the app; it says so rather than showing an
empty bubble. Once the model starts reading your prompt the line counts it up
(*Reading the prompt… 2,048 / 8,192 tokens*), on both engines; a prompt that
is mostly already cached from the last turn has little to count and the line
may go straight from the wait to the first token. The line clears the moment
the first token — or the first *thinking* token — arrives.

### Navigating away mid-generation

**The generation is not cancelled and the answer is not truncated.** Switching
conversations, switching to another page, locking your phone, or backgrounding
the tab all end your *subscription* to the stream. The run keeps going on the
server and commits the complete answer when it finishes.

The app tells you at the moment you leave — switching conversations says
*"<title>" keeps generating*, and switching models says the reply in flight
finishes on the previous model and saves here.

Come back and the whole reply is there. If it is still running when you return,
the status line says so and the Send button reads **Stop**:

> *Still generating — this reply was started elsewhere and is finishing on the
> server.*

Closing the tab or reloading is the one case that warns you first, because a
reload also throws away anything not yet written.

### Stopping

**Stop** is the same button as Send, relabelled while a run is in flight. It
genuinely aborts: the model stops, and whatever had been generated is saved as a
partial message. That partial is a real message — you can edit it, continue it,
or delete it.

This is the difference worth remembering: **walking away keeps the whole answer;
pressing Stop keeps only what had arrived.**

### What each answer cost

Under every answer is a muted line with what its generation measured: tokens,
speed, how much of the prompt was reused from cache, speculative-decode
acceptance when a drafter ran, and peak memory. It is saved with the message,
so it is still there after a reload. When a request reused little or nothing,
the line says why when the server knows ("cache miss: cold"); a reason starting
with "probably" is the server's inference, not something the engine reported.

---

## 5. Editing messages

Every message has an **Edit** action, both yours and the model's.

**Save & Continue is offered but disabled** when the current model cannot do it
— continuing your *own* message needs an MLX model — and its tooltip says why.
It is disabled rather than hidden, and never fires on a guess: the continuation
discards everything after the message, and that would land before the failure.

**Thinking is editable too.** When an assistant message has a reasoning trace,
the editor shows two boxes, captioned *Thinking* and *Response*. Clearing the
thinking box removes the block entirely rather than leaving an empty one.

**What Save & Continue does with the two boxes** (since v1.79.62):

- Thinking present, response **empty** — the model resumes *inside* its
  thinking, from the end of the box. This is what a reply you pressed Stop on
  mid-thought looks like, and the continuation is one trace, not two.
- Response present — the thinking is rendered as finished and the model
  continues the response from the end of that box.

**There are no special tokens in the boxes, and that is not a display
setting.** The store holds the thinking and the response as plain text; the
model's chat template puts the markers (`<|im_start|>`, `<think>`, gemma's
thought channel) around each turn when the prompt is built. **Preview
prompt**, in the editor, shows that exact text for the boxes as they are —
markers highlighted — before you commit to Save & Continue. The eye button
beside the composer does the same for the next message you would send. Both
need the model to be loaded; neither loads it for you.

Resuming inside a thought is engine-level on gguf (llama-server does it
natively) and works on every MLX family the server parses (`<think>`
templates, gemma-4 thought channels, harmony). A template with no thinking
structure at all refuses a thinking-only continue and says to put something
in the response box or regenerate.

Preview prompt is on user messages too: it shows what Save & Regenerate
would send, with the box as it is.

The editor offers up to three buttons:

- **Save** — writes the change and nothing else.
- **Save & Regenerate** (user messages) — saves, discards everything after this
  message, and generates a fresh reply from it.
- **Save & Continue** (both roles) — saves, discards everything after, and lets
  the model carry on writing *from the end of this text*. Use it to steer: edit
  the model's half-finished answer, delete the part that went wrong, and let it
  resume from there.

Both destructive buttons refuse while a response is streaming, and say why.

**Delete** removes a single message and leaves the rest of the thread intact.

### The unsaved row

Rarely, a message will appear with **Retry save** and **Discard** buttons and a
note that it was not saved. This means the message exists on your screen but
never reached the server. While one is present, sending and every destructive
action is blocked, because the page's idea of message order no longer matches
the store and acting on it could truncate the wrong part of the thread on the
server. Resolve it either way and normal operation resumes.

---

## 6. Attachments

Three ways in, all equivalent: the attach button, paste, and drag-and-drop
(desktop only — everything droppable is also reachable through the button and
paste).

The attach button only appears for models that can actually read the file type;
images need a vision model, audio is gguf-only. Dropping or pasting a file onto
a model without the capability refuses immediately and stages nothing, rather
than accepting it and failing later.

Oversized images are downscaled before they go on the wire (longest edge
`MAX_EDGE_PX` in `js/image-prep.js`).

Each staged image shows what it costs the selected model, in tokens, on the
thumbnail. The number comes from the model itself, so it only appears once the
model is loaded; before that the badge is a "?" that says so. Models differ a
lot here: Qwen-family models spend more tokens on a bigger image, gemma-4
spends the same number whatever the size.

**Full res** or **Shrink** appears when the model would resize the image to a
different size; the label says which way, and its tooltip gives the new cost
beside the current one. It resizes your original, once, to exactly that size
and aspect ratio, so the model does no further resampling, and the badge
updates. On a Qwen model on MLX it is usually **Full res**: a choice of detail
over cost, above the 2048px upload cap. It does not appear for llama.cpp
models, which do not report the size they resize to.

**Thinking** is one control at the top of the panel, built from the selected
model's chat template: *Default* (named for what the model does untouched:
"(medium)", "(on)", "(off)"), *Off*, and each depth level the template offers,
in the template's own words and order, its default marked "(default)". *On*
appears only for a template with a switch and no default level; where there is
one, picking it is the same prompt. A level that turns
thinking off (the unsloth Qwen3.8 override's `none`) is folded into *Off*
rather than listed twice. A model with levels and no on/off switch (MiniMax)
lists only its levels; a template that takes any word (gpt-oss) gets a text
box with its known values as suggestions. The row names the template file the
choices come from (and says when it is your own override, written on the Models
page, which beats a `chat_template.jinja` beside the weights; that page says
which files an override hides), and notes when a change mid-conversation
re-processes the whole conversation. *Off* remembers the level you had, so the thinking button
beside the composer, which flips the effective state, turns back on to it. A
level saved on another model (in a preset, say) stays saved but shows as "not
offered by this model" and is not sent. Since v1.79.62 a model that can think
thinks by default unless its own settings say otherwise. Depth levels are
instructions the model may overrun; the token cap below is what bounds
length. The *Advanced* group is folded by default and its heading counts the
values you have changed inside it.

**Thinking token cap**, indented under Thinking, is a hard cap on thinking
tokens, not a level. Past it the engine forces the thinking block shut and the
model goes on to answer. Depth levels are instructions the model may overrun;
the cap is the control that actually bounds how long a reply thinks. A cut can
cost answer quality, which has not been measured. It shows only while the
model will think, only for models whose thinking format the engine can close
(not gpt-oss), and says when the engine may not enforce it (llama.cpp decides
per request; the tooltip gives its reason) or when a value is small enough to
end the thought at once. Empty means no cap.

---

## 7. The other pages

- **Notebook** — a single free-form document instead of a thread. Shares the
  preset bar and system-prompt editor with chat, and behaves identically for
  both.
- **Models** — what is installed, what is resident, and per-model configuration.
  The fields here come from the backend, so this page grows new options
  automatically. Each row's **Engine** panel says what runs the model and
  with what: the library, the context (the ceiling, and what a loaded model
  actually got), the chat template in force, and every setting with its value,
  what auto would pick, and why. A value you set is marked "set"; a value
  reported by the running process is marked "live"; "unknown" and "n/a" say
  why. Tap a line for its reason.

In chat, the chip at the end of the bar names the library running the
selected model; tap it for its context and template, with a link to the full
panel here.
- **Perf** — timing and throughput for past generations, and a Cache table: per
  model, how much of its prompts was reused and why the rest was not.
- **Explore** — per-token inspection of a generation.

Sampler settings are shared across pages. Display preferences (such as showing
special tokens) are kept separately from sampler settings on purpose, so a
display toggle can never be mistaken for something the model receives.

**Show special tokens does nothing on a gguf model**, and the row says so:
llama-server splits the thinking and stops at the end-of-turn token inside its
own process, so no marker ever reaches this app to show. To see the markers,
use the prompt preview — it is the template's own render.

---

## 8. Known rough edges

Written down because they are real, not because they are scheduled. Each is a
place where the interface does not currently say enough for the behaviour to be
guessable.

**Send and Stop are one button.** The tooltip now separates "stop what you are
watching" from "stop the run finishing on the server", but the button face reads
the same in both cases, and a tooltip is not reachable by touch.

**The prompt preview cannot show images on MLX.** The vision path renders
through mlx-vlm and has no text-only render, so the preview shows the text
template. It now says so on the panel, naming how many images are being sent
but not shown -- the picture does reach the model.

**An image can only go on a user message when the model is MLX.** The attach
control is withheld while editing an assistant message on an MLX model, and a
paste or drop there is refused. This is not a UI choice: mlx-vlm can only
render an image marker on a user turn, and would otherwise move the picture to
your latest message and describe it to the model as if it had arrived there.
Switch to a gguf model to put an image on an assistant turn.

**An unsaved chat template survives a reload on a desktop, but not a page
change and not a phone.** Type a template into a model's config panel and the
browser asks before you reload or close the tab. It does not ask when you click
Chat or Notebook in the nav bar: the app changes pages without the browser
noticing, so there is no moment to ask at, and the text is gone. It does not
ask on iOS Safari either, which does not raise that dialog reliably. The draft
is never stored anywhere -- Save is the only thing that keeps it.

**Save & Continue on an assistant message carrying an image works on some
models and not others.** It depends on whether the model's chat template keeps
the image's place in the turn being continued. Where it does not, you get a
clear refusal naming the reason -- remove the attachment to continue that
message, or generate a fresh reply. Continuing a text-only message is
unaffected, and so is regenerating.

### Closed

Kept as a record rather than deleted, so this section reads as a ledger.

- *Thinking offered three names for one prompt, and the budget read like a
  level* — closed in v2.0.172: one *Default* entry, the template's levels with
  its default marked, and the cap as an indented token field that hides while
  thinking is off.
- *A new conversation inherited the open one's settings, and a cleared prompt
  came back from the selected preset* — closed in v2.0.172: new conversations
  start from their preset or blank, the no-conversation draft is created as
  shown, and the prompt box has Clear.

- *Apply vs Update — which way does each one point?* — closed in v1.79.62:
  Update is Save, and the drift line names the preset, the knobs and the
  direction of each button.
- *The thinking checkbox read "off" while a model thought, and there was no
  way to say off* — closed in v1.79.62 by the tri-state control and the
  server reporting the model's default.
- *Nothing showed what wraps the thinking when you edit or continue* —
  closed in v1.79.62 by Preview prompt.
- *Save & Continue on a reply stopped mid-thought started a second thought* —
  closed in v1.79.62; it resumes the first. Gemma-channel and harmony models
  on MLX joined in v1.79.63, and the missing space at the seam ("Ineed") went
  in v1.79.64 — it was a first-token strip on the MLX path, not the model.

- *The sampler panel does not say whose settings it is showing* — closed in
  v1.79.25 by the scope line under the *Sampling* heading, and by renaming
  "Reset to defaults" to "Clear all overrides".
- *"Differs from current settings" does not say what differs* — closed in
  v1.79.25; the line names the prompt, the settings, or both.
- *A preset that lost its prompt looks healthy* — closed in v1.79.25; the
  dropdown marks it "settings only" at the point you choose it.
- *Browsing and choosing a save destination are the same control* — closed in
  v1.79.26. Update targets the dropdown selection; Save as new refuses a name in
  use. There is no typing path to an overwrite.
- *Nothing says the generation survives you leaving* — closed in v1.79.26, which
  also fixed the client reporting **"Stopped."** for a run that was in fact
  still generating.
- *"Save & Continue" appearing and disappearing looks arbitrary* — closed in
  v1.79.27; disabled with a reason instead of absent.
- *Losing thinking on a model switch was announced only when media was also
  being dropped* — found while closing the above, fixed in v1.79.27. A
  text-only conversation switching to a plain model said nothing at all.
