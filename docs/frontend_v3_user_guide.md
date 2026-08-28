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
**your settings live in three places at once, and the screen does not tell you
which one you are looking at.**

| Layer | Where it lives | What it is for | Changes when |
|---|---|---|---|
| **Global defaults** | your browser (localStorage) | what a brand-new conversation starts from, when no preset is involved | you move a slider with no conversation open |
| **The conversation** | the server, on the conversation row | what actually gets sent to the model | you move a slider with a conversation open |
| **The preset** | the server, in a named store | a snapshot you can copy into a conversation, or copy a conversation into | only when you press **Save** |

The system prompt has the same three layers: a draft parked in your browser
before any conversation exists, the conversation's own prompt, and the prompt a
preset carries.

Two consequences worth internalising:

- **The sampler panel is the active conversation's settings.** Selecting a
  different conversation silently replaces every value in that panel with that
  conversation's stored ones. Nothing animates, nothing announces it. If you
  had "temperature 1.3" on screen and you click another conversation, the 1.3
  is not lost — it is still on the first conversation — but the panel now shows
  something else.
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

### Save

Takes the current conversation's prompt and the entire sampler panel, and writes
them to the preset named in the box next to the button. A name that already
exists is overwritten. A new name creates a preset.

**Selecting a preset from the dropdown pre-fills that name box.** That is the
sharp edge in this design: browsing presets aims Save at the one you are
browsing. Save asks for confirmation before it overwrites a preset's stored
prompt, except in one case:

- **No confirmation** when you are saving back onto the preset the conversation
  is already running, and that preset is the one showing in the dropdown. This
  is the ordinary loop — apply a preset, tweak the prompt, save it back — and
  making you confirm it every time would train you to click through the
  confirmation that matters.
- **Confirmation** for everything else that would change a stored prompt,
  including *blanking* one. Saving from a conversation with no system prompt
  writes an empty prompt to the preset, and a preset with no prompt does nothing
  when applied. It stays in the list but stops working, which reads as "my
  preset disappeared."

The confirmation is the button itself changing to **Overwrite prompt?** for a
few seconds. Click it again to go through. Changing the dropdown or the name box
cancels it, and so does editing the prompt underneath — the confirmation is a
promise about one specific write, and if the write changes, the promise is void.

### If you change a setting and never apply or save

Nothing is lost and nothing is hidden. The change is on **the conversation**
immediately (a short debounce, then a write to the server) and it is what the
model receives on your next message.

The preset is untouched. The drift line under the dropdown says which situation
you are in:

> *Differs from current settings — Apply copies it here, Save overwrites it.*

Read that as: your conversation and this preset have diverged. **Apply** discards
your changes in favour of the preset's. **Save** discards the preset's in favour
of yours. There is no merge and no third option.

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
  conversation keeps them; the model just never sees them.
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

**Save & Continue does not always appear on your own messages.** Continuing a
*user* turn only works on MLX models; the gguf/llama-server path has no way to
express it. The button hides rather than offering an action that would fail, and
it also hides while the model's provider is still unknown.

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

**The sampler panel does not say whose settings it is showing.** It is the
active conversation's, it is also the source for the next new conversation, and
selecting a conversation overwrites it without comment. Someone tuning a value,
switching away to check something, and coming back will find different numbers
and no explanation. This is the root of most preset confusion; the preset
machinery is downstream of it.

**Browsing and choosing a save destination are the same control.** Selecting a
preset to look at it pre-fills the save-as name, which is why a stray Save
overwrites the preset you were merely inspecting. The confirmation now catches
this, but a confirmation is a guard on a sharp edge, not the absence of one.
Separating "which preset am I reading" from "which preset am I writing" would
remove the edge instead.

**"Differs from current settings" does not say what differs.** It cannot
distinguish a changed prompt from a nudged temperature, so it reads as alarming
after a trivial edit and identical after a total rewrite.

**Nothing says the generation survives you leaving.** This is genuinely good
behaviour and almost nobody will discover it, because the only place it is
mentioned is a status line you see when you happen to return mid-run. Users
reasonably assume walking away truncates the answer, and so they wait.

**Send and Stop are one button, and it can read Stop for a run you did not
start.** Correct — the run is yours, started in another tab or before a reload —
but there is no cue distinguishing "stop the thing streaming in front of me"
from "stop the thing finishing invisibly."

**"Save & Continue" appearing and disappearing looks arbitrary.** The reason
(user-turn continuation is MLX-only) is invisible at the point of use. A
disabled button with a reason would be more honest than an absent one.

**Thinking depth offers values a given model will reject.** The dropdown lists
the union of what different model families accept, so a wrong-for-this-model
value reaches the model and comes back an error. The control cannot currently
narrow itself per model.

**A preset that lost its prompt looks healthy.** It stays in the list and
applies without complaint; it simply stops affecting the prompt. The list gives
no indication which presets carry a prompt and which do not.
