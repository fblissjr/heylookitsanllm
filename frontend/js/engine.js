// The engine contract (plan W13; server side: src/heylook_llm/providers/contract.py).
// /v1/models and /v1/admin/models carry ONE `engine` object per model, the
// same shape whatever engine runs it. Every leaf is a Fact
// {value, provenance, source}; settings are {value, configured, auto,
// reason, provenance, effect}. Read values through readFact so no page
// grows its own notion of where a value lives.

export function readFact(fact) {
  return fact?.value ?? null;
}

// The context ceiling the model's files declare (gguf header / MLX config.json).
export function contextCeiling(row) {
  return readFact(row?.engine?.context?.length);
}

// What the resident process was sized to (gguf /props at ready); null when
// unloaded, and not applicable on MLX.
export function contextRunning(row) {
  return readFact(row?.engine?.context?.running);
}
