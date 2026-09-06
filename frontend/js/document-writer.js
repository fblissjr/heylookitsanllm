// The per-document write path, shared by chat (conversations) and notebook
// (notebooks). One implementation, one place to fix.
//
// Chat and notebook each carried their own copy of this, byte-identical apart
// from which api function they called -- and notebook's said "same shape as
// chat's putSystemPrompt" in a comment, which is the tell. What the copies
// were duplicating is not boilerplate: it is the keepalive ordering rule
// below, an invariant that was hand-maintained in two files.

export function createDocumentWriter({ update, onError }) {
  // The PUT chain, owned here rather than parked on a page's state object.
  // Serialising writes is what stops an older value landing after a newer one.
  let chain = null;

  return {
    // Write the document's system prompt.
    //
    // A hide-time flush (`keepalive`) cannot wait its turn: the page may be
    // unloading, and a request still queued behind an in-flight PUT is never
    // sent at all. Dispatch it NOW -- keepalive is what lets it outlive the
    // document -- and make the chain wait on it instead. It carries the newest
    // value, so landing ahead of an older in-flight PUT is the right order,
    // not a race.
    putSystemPrompt(docId, value, { keepalive = false } = {}) {
      const put = () => update(docId, { system_prompt: value }, { keepalive });
      const next = keepalive ? put() : (chain ?? Promise.resolve()).then(put);
      chain = next.catch((err) => onError(`System prompt save failed: ${err.message}`));
    },

    // Stamp which preset this document is RUNNING. Deliberately not chained
    // with the prompt: it is a different column and a slow prompt PUT should
    // not delay it.
    setAppliedPreset(docId, presetId) {
      update(docId, { applied_preset_id: presetId })
        .catch((err) => onError(`Preset stamp save failed: ${err.message}`));
    },
  };
}
