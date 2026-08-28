# v3 user guide

last updated: 2026-08-28

How the `/v3` UI actually behaves, written for the person using it rather than
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
| **Global defaults** | your browser (localStorage) | what a brand-new conversation starts from, when no preset is involved | you move a slider with no conversation open |
| **The conversation** | the server, on the conversation row | what actually gets sent to the model | you move a slider with a conversation open |
| **The preset** | the server, in a named store | a snapshot you can copy into a conversation, or copy a conversation into | only when you press **Update** or **Save as new** |

The system prompt has the same three layers: a draft parked in your browser
before any conversation exists, the conversation's own prompt, and the prompt a
preset carries.

Two consequences worth internalising:

- **The sampler panel is the active conversation's settings**, and it says so:
  under the *Sampling* heading it reads **"Applies to this conversation —
  changes save as you make them."** With no conversation open it reads
  **"Defaults for new conversations."**, because that is what the panel is then
  — the seed the next new conversation starts from.
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

### Update and Save as new

Both buttons write the current conversation's prompt and the entire sampler
panel to a preset. They differ only in *which* preset.

There are **two** write buttons, and which preset you hit is decided by which
one you press — not by what you type:

- **Update** overwrites the preset showing in the dropdown, the one the preview
  directly above it is displaying. This is the only way to overwrite a preset.
- **Save as new** creates one under the typed name. If that name is already in
  use it refuses and tells you to use Update instead. It can never overwrite.

Update asks for confirmation before it changes a preset's stored prompt, except
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
situation you are in, and which half moved:

> *Matches current settings.*
> *Prompt differs — Apply copies it here, Update overwrites it.*
> *Settings differ — …*
> *Prompt and settings differ — …*

Read that as: your conversation and this preset have diverged. **Apply** discards
your changes in favour of the preset's. **Update** discards the preset's in
favour of yours. There is no merge and no third option.

Beside the model selector, a chip names the preset the conversation is running
and appends **(edited)** once anything the preset speaks for has changed.

### Delete

Removes the preset. Conversations that were running it keep their prompt and
settings — they are copies — they simply stop showing a preset name.

---

## 3. Conversations

**New conversation** starts from the selected preset if there is one (its
prompt, its settings, and its stamp). With no preset selected, it inherits the
sampler panel as it stands but *not* the current conversation's system prompt.
A prompt typed before any conversation exists is the one exception: it wins over
a preset, because typing it was the more explicit act.

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

### The wait before the first token

The status line names it. On a model that is not resident this is a multi-GB
load and the single longest wait in the app; it says so rather than showing an
empty bubble. The line clears the moment the first token — or the first
*thinking* token — arrives.

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

Oversized images are downscaled before they go on the wire. **Vision tokens /
image** in the advanced settings controls how much detail the model spends on
each image.

**Thinking depth** lists the union of what different model families accept, so
a value one model takes another will reject. The control says so, and `auto`
always works — it leaves the model's own default alone.

---

## 7. The other pages

- **Notebook** — a single free-form document instead of a thread. Shares the
  preset bar and system-prompt editor with chat, and behaves identically for
  both.
- **Models** — what is installed, what is resident, and per-model configuration.
  The fields here come from the backend, so this page grows new options
  automatically.
- **Perf** — timing and throughput for past generations.
- **Explore** — per-token inspection of a generation.
- **Jspace** — the Jacobian-lens interpretability view; see
  [jspace_guide.md](./jspace_guide.md).

Sampler settings are shared across pages. Display preferences (such as showing
special tokens) are kept separately from sampler settings on purpose, so a
display toggle can never be mistaken for something the model receives.

---

## 8. Known rough edges

Written down because they are real, not because they are scheduled. Each is a
place where the interface does not currently say enough for the behaviour to be
guessable.

**Send and Stop are one button.** The tooltip now separates "stop what you are
watching" from "stop the run finishing on the server", but the button face reads
the same in both cases, and a tooltip is not reachable by touch.

**Thinking depth still offers values a given model will reject.** The control
now warns that the accepted set differs per model, but it cannot narrow itself:
the accepted values live in the model's chat template, and for gguf inside the
GGUF's own metadata, so the backend would have to learn and expose them before
the UI could.

### Closed

Kept as a record rather than deleted, so this section reads as a ledger.

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
