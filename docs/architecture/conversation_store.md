# Conversation store: what each operation writes

Last updated: 2026-09-08

What the chat frontend actually causes DuckDB to INSERT, UPDATE and DELETE, and
the media-blob lifecycle underneath it. Written after driving the real frontend
against a live gguf server and diffing the store row-by-row after every
operation, because the mechanics were only inferable from code before.

Companion to [config.md](./config.md) (models.toml + the sampler cascade). The
schema itself is `db.py`'s `_SCHEMA_SQL`; this is the behaviour on top of it.

## The row ledger

Observed by snapshotting `GET /v1/conversations/{id}` before and after each UI
action and diffing message ids, positions, content, thinking and media.

| UI action | store effect |
|---|---|
| Send a message | `INSERT` the user row, `INSERT` the assistant row (stamped with the generating `model_id`); conversation `updated_at` touched |
| Edit a message, Save | `UPDATE` that row's `content_blocks` and/or `thinking`. No new row, position unchanged |
| Edit + attach an image | same `UPDATE`; the row's blocks gain an image block, `INSERT` into `media_blobs` if the bytes are new |
| Save & Regenerate | truncate from the edited row, then generate: the tail rows are `DELETE`d and a fresh assistant row `INSERT`ed |
| Regenerate (assistant) | replaces in place -- the row count does not grow |
| Save & Continue | `UPDATE` the same row, extending it. **No new row**, and the existing text/thinking stays as the exact prefix |
| Delete a message | `DELETE` that row, then `_gc_media` for the conversation |
| Delete a conversation | `DELETE` its messages, `DELETE` its `media_blobs`, `DELETE` the conversation |
| Rename | `UPDATE conversations.title` |
| Clone | a new conversation with its own message rows AND its own blob rows |

`updated_at` moves on the conversation for every message write; the sidebar
orders by it while the client's in-memory array is only re-sorted on a fresh
list fetch, so **DOM order is not API order** -- anything outside the app that
needs to identify the active conversation must use the row's `data-id`, never
its index.

## Media blobs

`media_blobs` is keyed `PRIMARY KEY (conversation_id, id)` where `id` is the
SHA-256 of the bytes. Three consequences, all verified live:

- **Content-addressed within a conversation.** The same image attached twice
  produces one row -- the second write is an `INSERT OR IGNORE` on the same
  hash.
- **Duplicated across conversations.** Two conversations holding the same image
  are two rows with the same `id` and different `conversation_id`, each with its
  own copy of `data`. That is what makes deletion safe: `delete_conversation`
  runs `DELETE FROM media_blobs WHERE conversation_id = ?`, so it can never
  reach another conversation's copy. Confirmed: deleting one of two
  conversations sharing an image leaves the other's blob serving its full byte
  length, while the deleted one's URL correctly 404s. The cost is storage --
  N conversations sharing an image store N copies of the bytes.
- **Reference-counted within a conversation.** `_gc_media` drops blobs no
  message in that conversation still references, and runs from every mutation
  that can drop a reference (message edit, message delete, truncation,
  regenerate). Confirmed: with two messages referencing one image, deleting the
  first leaves the blob served; deleting the last referencer 404s it.

A clone therefore survives its parent's deletion intact, images included.

The generate path does NOT read these rows as URLs -- providers cannot fetch our
own relative URLs. `_media_ids_for_wire` collects the ids, `db.get_media_blobs`
loads the bytes, and `_wire_content` inlines them. A referenced blob missing
from that map raises rather than silently dropping the image, so store
corruption surfaces as a 500 before any stream starts.

## Thinking

`messages.thinking` is a column of its own, not part of `content_blocks`. An
assistant row can legitimately carry **empty content and non-empty thinking**:
a generation stopped mid-thought, or one that hit `max_tokens` while still
reasoning, stores exactly `[{"type":"text","text":""}]` with the trace in
`thinking`. Any consumer that treats an empty content block as an empty message
is wrong about a shape the store really holds.

The editor exposes both columns as separate textareas -- `aria-label="Edit
thinking"` renders ABOVE `aria-label="Edit message"`, so DOM order is
thinking-then-content. Both persist on Save.

Editing the thinking and then Save & Continue resumes INSIDE the reasoning
block on gguf (llama.cpp's `COMMON_CHAT_CONTINUATION_REASONING`, reached because
the row carries `reasoning_content` and empty content): the edited trace comes
back as the exact prefix of a longer one, in the same row.

## Context size

Three different fields, routinely conflated:

- `context_length` -- the GGUF header's training context. A **ceiling**, not a
  setting.
- `ctx_size` -- what models.toml asks for. Absent = llama-server's `-c 0`, i.e.
  the training context, then `--fit` shrinking to device memory.
- `context_running` -- what the process actually got, read from `/props` at
  ready. The only field that answers "what is it running at".

The chat page's context select offers power-of-two steps from 4K up to
`context_length`, plus the ceiling itself when it is not a power of two, plus
the stored value so the panel shows what is in force -- and a `Custom…` number
input, so the range is not limited to the steps. `Auto` sends 0.

Picking one **persists**: `POST /v1/admin/models/{id}/reload?ctx_size=N` writes
it as the model's `ctx_size` through the one config writer and then loads. For a
discovered model that materializes an entry. It is a spawn-time flag, so it
takes effect on the next load, and asking is not getting -- `--fit` can still
trim, which is what `context_running` is for.
