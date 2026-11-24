// Client-side AI preset assistant loop over POST /v1/messages with Anthropic
// tool calling (mirrors `heylook presets assist` in src/heylook_llm/cli.py).
//
// Pure ESM, zero DOM dependencies: runs the tool loop against a snapshot of
// saved presets, stages proposed edits in memory without writing anything, and
// returns normalized preset records for the diff-review dialog to apply.
// Supports both non-streaming (`createMessage`) and SSE streaming (`streamMessage`)
// so the UI shows live thinking/tool-drafting progress.

import { httpError, requestId } from './api.js';

export const PRESET_ASSIST_SYSTEM =
  'You help the user edit saved heylook presets cleanly and accurately. '
  + 'Use list_presets and get_preset to inspect existing presets before making changes. '
  + 'When updating a preset\'s system prompt, weave new wording in naturally and match '
  + 'the existing prompt\'s tone, formatting, markdown structure, and casing. '
  + 'Call propose_preset_update for each preset you want to modify; nothing is saved '
  + 'until the user reviews the before/after diff and confirms. Keep untouched sections intact.';

export const PRESET_ASSIST_TOOLS = [
  {
    name: 'list_presets',
    description: 'List all saved preset names and brief metadata.',
    input_schema: { type: 'object', properties: {} },
  },
  {
    name: 'get_preset',
    description: 'Read the full system_prompt and sampler params of a saved preset by exact name.',
    input_schema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Exact preset name' },
      },
      required: ['name'],
    },
  },
  {
    name: 'propose_preset_update',
    description:
      'Stage a proposed update for an existing preset (or a new preset). '
      + 'Changes are shown to the user as a before/after diff before saving. '
      + 'Omit params to preserve the preset\'s existing sampler params.',
    input_schema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Preset name' },
        system_prompt: {
          type: ['string', 'null'],
          description: 'Full updated system prompt text (null for settings-only).',
        },
        params: {
          type: 'object',
          description: 'Optional full sampler params object; omit to keep existing params.',
        },
      },
      required: ['name'],
    },
  },
];

// Filter out embedding models and sort tool-capable models first.
export function toolCapableModels(models) {
  const generative = (models ?? []).filter((m) => !m?.engine?.embedding);
  return generative
    .map((m) => ({
      ...m,
      supportsTools: Array.isArray(m.capabilities) && m.capabilities.includes('tools'),
    }))
    .sort((a, b) => Number(b.supportsTools) - Number(a.supportsTools));
}

// Assemble one Anthropic Messages response from a sequence of parsed SSE
// `(eventName, data)` frames, invoking `onDelta` for live progress UI.
export function createSseMessageAssembler(onDelta) {
  const blocks = new Map();
  let stopReason = null;
  let thinkingChars = 0;
  let textChars = 0;

  return {
    feed(eventName, data) {
      if (!data || typeof data !== 'object') return;
      if (eventName === 'error' || data.type === 'error') {
        const msg = data.error?.message || data.message || 'Stream error';
        throw new Error(msg);
      }
      if (data.type === 'content_block_start') {
        const idx = data.index ?? blocks.size;
        const cb = data.content_block ?? {};
        if (cb.type === 'tool_use') {
          blocks.set(idx, {
            type: 'tool_use',
            id: cb.id || `toolu_${idx}`,
            name: cb.name || '',
            input: (cb.input && typeof cb.input === 'object') ? { ...cb.input } : {},
            _json: '',
          });
          onDelta?.({ kind: 'tool_start', toolName: cb.name || '' });
        } else if (cb.type === 'thinking') {
          blocks.set(idx, { type: 'thinking', thinking: cb.thinking || '' });
        } else {
          blocks.set(idx, { type: 'text', text: cb.text || '' });
        }
        return;
      }
      if (data.type === 'content_block_delta') {
        const idx = data.index ?? 0;
        const block = blocks.get(idx);
        const delta = data.delta ?? {};
        if (!block) return;
        if (delta.type === 'text_delta' && delta.text) {
          block.text += delta.text;
          textChars += delta.text.length;
          onDelta?.({ kind: 'text', chars: textChars });
        } else if (delta.type === 'thinking_delta' && delta.thinking) {
          block.thinking += delta.thinking;
          thinkingChars += delta.thinking.length;
          onDelta?.({ kind: 'thinking', chars: thinkingChars });
        } else if (delta.type === 'input_json_delta' && delta.partial_json) {
          block._json += delta.partial_json;
          onDelta?.({ kind: 'tool_json', toolName: block.name, chars: block._json.length });
        }
        return;
      }
      if (data.type === 'message_delta' && data.delta?.stop_reason) {
        stopReason = data.delta.stop_reason;
      }
    },
    finish() {
      const content = [...blocks.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([, block]) => {
          if (block.type === 'tool_use') {
            let input = block.input;
            if (block._json) {
              try {
                const parsed = JSON.parse(block._json);
                if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) input = parsed;
              } catch { /* keep partial/default input */ }
            }
            return { type: 'tool_use', id: block.id, name: block.name, input };
          }
          return block;
        });
      return { content, stop_reason: stopReason };
    },
  };
}

// Stream one POST /v1/messages turn via SSE and return the assembled response.
export async function streamAnthropicMessage(body, { signal, onDelta } = {}) {
  const res = await fetch('/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': requestId(),
    },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok) throw await httpError(res);

  const assembler = createSseMessageAssembler(onDelta);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let eventName = null;

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split(/\r?\n/);
      buf = lines.pop() ?? '';
      for (const line of lines) {
        if (!line) {
          eventName = null;
          continue;
        }
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          const raw = line.slice(5).trim();
          if (!raw || raw === '[DONE]') continue;
          assembler.feed(eventName, JSON.parse(raw));
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  return assembler.finish();
}

function cleanParams(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
  const out = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v != null) out[k] = v;
  }
  return out;
}

function normalizePrompt(value) {
  if (value == null) return null;
  const text = String(value).trim();
  return text || null;
}

function executePresetTool(name, input, byName, staged) {
  const args = (input && typeof input === 'object' && !Array.isArray(input)) ? input : {};

  if (name === 'list_presets') {
    const items = [...byName.values()].map((p) => ({
      name: p.name,
      has_system_prompt: Boolean(p.system_prompt),
      prompt_chars: (p.system_prompt ?? '').length,
      params: p.params ?? {},
    }));
    return { text: JSON.stringify({ presets: items }), isError: false, summary: `Listed ${items.length} presets` };
  }

  if (name === 'get_preset') {
    const presetName = String(args.name ?? '').trim();
    const found = byName.get(presetName);
    if (!found) {
      return {
        text: JSON.stringify({ error: `Preset not found: ${presetName}` }),
        isError: true,
        summary: `Preset not found: ${presetName}`,
      };
    }
    return {
      text: JSON.stringify({
        name: found.name,
        system_prompt: found.system_prompt ?? null,
        params: found.params ?? {},
      }),
      isError: false,
      summary: `Read "${found.name}"`,
    };
  }

  if (name === 'propose_preset_update') {
    const presetName = String(args.name ?? '').trim();
    if (!presetName) {
      return {
        text: JSON.stringify({ error: 'name is required' }),
        isError: true,
        summary: 'Missing preset name',
      };
    }
    const existing = byName.get(presetName) ?? null;
    const prevStaged = staged.get(presetName) ?? null;

    let systemPrompt;
    if (Object.hasOwn(args, 'system_prompt')) {
      systemPrompt = normalizePrompt(args.system_prompt);
    } else if (prevStaged) {
      systemPrompt = prevStaged.system_prompt;
    } else {
      systemPrompt = existing ? normalizePrompt(existing.system_prompt) : null;
    }

    let params;
    if (Object.hasOwn(args, 'params') && args.params !== undefined) {
      if (args.params === null) {
        params = {};
      } else if (typeof args.params !== 'object' || Array.isArray(args.params)) {
        return {
          text: JSON.stringify({ error: 'params must be an object' }),
          isError: true,
          summary: `Invalid params for "${presetName}"`,
        };
      } else {
        params = cleanParams(args.params);
      }
    } else if (prevStaged) {
      params = { ...prevStaged.params };
    } else {
      params = existing ? cleanParams(existing.params) : {};
    }

    staged.set(presetName, {
      name: presetName,
      system_prompt: systemPrompt,
      params,
    });
    return {
      text: JSON.stringify({
        staged: true,
        name: presetName,
        status: existing ? 'existing' : 'new',
      }),
      isError: false,
      summary: `Staged update for "${presetName}"`,
    };
  }

  return {
    text: JSON.stringify({ error: `Unknown tool: ${name}` }),
    isError: true,
    summary: `Unknown tool: ${name}`,
  };
}

export async function runPresetAssist({
  createMessage,
  streamMessage,
  model,
  instruction,
  presets,
  targetNames = [],
  maxTurns = 16,
  maxTokens = 4096,
  signal,
  onProgress,
}) {
  const cleanInstruction = String(instruction ?? '').trim();
  if (!model) throw new Error('Select a model first.');
  if (!cleanInstruction) throw new Error('Enter an instruction for the preset assistant.');

  const byName = new Map((presets ?? []).map((p) => [p.name, p]));
  const targets = (targetNames ?? []).filter((n) => byName.has(n));
  const userContent = targets.length
    ? `Target presets (only modify these): ${targets.join(', ')}.\n\n${cleanInstruction}`
    : cleanInstruction;

  const messages = [{ role: 'user', content: userContent }];
  const staged = new Map();
  const assistantNotes = [];
  let turns = 0;

  for (let turn = 1; turn <= maxTurns; turn += 1) {
    if (signal?.aborted) {
      const err = new Error('Cancelled.');
      err.name = 'AbortError';
      throw err;
    }
    turns = turn;
    onProgress?.({ turn, phase: 'thinking', stagedCount: staged.size });

    const payload = {
      model,
      system: PRESET_ASSIST_SYSTEM,
      messages,
      tools: PRESET_ASSIST_TOOLS,
      max_tokens: maxTokens,
      stream: Boolean(streamMessage),
    };

    const response = streamMessage
      ? await streamMessage(payload, {
        signal,
        onDelta: (d) => onProgress?.({
          turn,
          phase: 'stream',
          deltaKind: d.kind,
          toolName: d.toolName,
          chars: d.chars,
          stagedCount: staged.size,
        }),
      })
      : await createMessage(payload, { signal });

    const blocks = Array.isArray(response?.content) ? response.content : [];
    const textParts = blocks
      .filter((b) => b?.type === 'text' && b.text)
      .map((b) => b.text.trim())
      .filter(Boolean);
    const toolUses = blocks.filter((b) => b?.type === 'tool_use');

    if (textParts.length) assistantNotes.push(textParts.join('\n'));
    if (!toolUses.length) break;

    messages.push({ role: 'assistant', content: blocks });
    const toolResults = [];
    for (const call of toolUses) {
      const outcome = executePresetTool(call.name, call.input, byName, staged);
      onProgress?.({
        turn,
        phase: 'tool',
        toolName: call.name,
        summary: outcome.summary,
        stagedCount: staged.size,
      });
      toolResults.push({
        type: 'tool_result',
        tool_use_id: call.id,
        content: outcome.text,
        ...(outcome.isError ? { is_error: true } : {}),
      });
    }
    messages.push({ role: 'user', content: toolResults });
  }

  return {
    proposals: [...staged.values()],
    assistantText: assistantNotes.length ? assistantNotes[assistantNotes.length - 1] : '',
    turns,
  };
}
