# Features Documentation

Last updated: 2026-02-27

## Feature Status Overview

| Category | Implemented | Planned | Out of Scope |
|----------|-------------|---------|--------------|
| Chat | 9 | 2 | 1 |
| Models | 6 | 2 | 0 |
| Storage | 7 | 0 | 0 |
| UI/UX | 11 | 1 | 2 |
| Settings | 13 | 0 | 0 |

---

## Implemented Features

### Chat Features

| Feature | Description | Status |
|---------|-------------|--------|
| Text Chat | Send and receive text messages | Complete |
| Streaming Responses | Real-time token-by-token display | Complete |
| Thinking Mode | Display model reasoning (Qwen3) | Complete |
| Stop Generation | Cancel ongoing response | Complete |
| Message Editing | Edit sent messages | Complete |
| Message Deletion | Delete individual messages | Complete |
| Cascade Delete | Delete message and all following | Complete |
| Regenerate Response | Re-generate assistant response | Complete |
| Image Attachments | Add images to messages (VLM) | Complete |

### Model Management

| Feature | Description | Status |
|---------|-------------|--------|
| Model List | View all available models | Complete |
| Model Loading | Load a model for chat | Complete |
| Model Unloading | Unload current model | Complete |
| Capability Badges | Show model capabilities (Chat, Vision, etc.) | Complete |
| Context Window | Adjust context size before loading | Complete |
| Hot Swap | Switch models mid-conversation | Complete |

### Conversation Management

| Feature | Description | Status |
|---------|-------------|--------|
| Create Conversation | Start new chat sessions | Complete |
| Switch Conversations | Navigate between chats | Complete |
| Delete Conversation | Remove conversation with confirmation | Complete |
| Auto-Title | Generate title from first message | Complete |
| Conversation List | Sidebar with all conversations | Complete |

### Persistence

| Feature | Description | Status |
|---------|-------------|--------|
| IndexedDB Storage | Store conversations in browser | Complete |
| Auto-Save | Debounced save on changes | Complete |
| Load on Startup | Restore conversations on refresh | Complete |
| Theme Persistence | Remember theme preference | Complete |
| Settings Persistence | Remember sampler settings | Complete |
| Export Conversations | Download all conversations as JSON | Complete |
| Import Conversations | Upload JSON file to restore chats | Complete |

### UI/UX

| Feature | Description | Status |
|---------|-------------|--------|
| Dark Theme | Native dark mode | Complete |
| Theme Toggle | Light/Dark/Auto mode selector in header | Complete |
| Responsive Layout | Works on various screen sizes | Complete |
| Collapsible Sidebar | Toggle conversation list | Complete |
| Loading States | Spinners and skeleton states | Complete |
| Error States | Connection failed, retry options | Complete |
| Hover Actions | Show actions on hover | Complete |
| Keyboard Shortcuts | Enter to send, Shift+Enter for newline | Complete |
| Copy Message | Copy message content | Complete |
| Token Usage Display | Show prompt/completion tokens per message | Complete |
| Message Timestamps | Show when messages were sent | Complete |

### Settings

| Feature | Description | Status |
|---------|-------------|--------|
| Temperature | Adjust randomness (0-2) | Complete |
| Top P | Nucleus sampling | Complete |
| Top K | Top-k sampling | Complete |
| Max Tokens | Limit response length | Complete |
| Min P | Minimum probability threshold | Complete |
| Repetition Penalty | Penalize repeated tokens | Complete |
| Presence Penalty | Encourage topic diversity | Complete |
| Frequency Penalty | Reduce word repetition | Complete |
| Seed | Reproducible generation | Complete |
| Settings Panel UI | Sliders and controls for all parameters | Complete |
| Sampler Presets | Balanced, Creative, Precise, Deterministic | Complete |
| System Prompt UI | Edit system prompt with presets | Complete |
| Thinking Mode Toggle | Enable/disable Qwen3 thinking mode | Complete |

---

## Planned Features (Gaps)

### High Priority

| Feature | Description | Complexity | Notes |
|---------|-------------|------------|-------|
| Config Editing | Edit models.toml via UI | High | Requires backend API endpoints |
| Multipart Image Upload | Use /v1/chat/completions/multipart | Low | API ready, UI uses base64 instead |
| Capabilities Detection | Call /v1/capabilities on startup | Low | fetchCapabilities() exists but unused |

### Medium Priority

| Feature | Description | Complexity | Notes |
|---------|-------------|------------|-------|
| Search | Search conversation content | Medium | Requires IndexedDB query |
| Keyboard Shortcuts | Cmd+K palette, Cmd+N new chat | Medium | Common UX pattern |
| Logprobs Display | Enable and display token probabilities | Medium | Types defined, needs UI toggle + display |
| Batch Processing UI | Multi-prompt batch interface | Medium | batchChat() API ready |

### Low Priority

| Feature | Description | Complexity | Notes |
|---------|-------------|------------|-------|
| Markdown Rendering | Render markdown in messages | - | Complete (MarkdownRenderer in composed/) |
| Code Highlighting | Syntax highlight code blocks | Medium | Need highlight.js or similar |
| Conversation Rename | Manual title editing | Low | Update function exists |
| Conversation Pinning | Pin important conversations | Low | Add pinned field |
| Message Reactions | Add reactions to messages | Low | UI polish |
| Typing Indicator | Show when model is "typing" | Low | Already have streaming state |

### Out of Scope (Advanced Backend Features)

| Feature | Description | Notes |
|---------|-------------|-------|
| Embeddings UI | /v1/embeddings endpoint | Specialized use case |
| Hidden States UI | /v1/hidden_states endpoint | Research/debugging use case |
| Audio Transcription UI | /v1/audio/transcriptions | macOS only, requires STT provider |

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Authentication | Local-only application |
| Multi-user | Single user design |
| Cloud Sync | Privacy - local storage only |
| Mobile App | Web-only for now |
| Plugins/Extensions | Keep core simple |

---

## Known Gaps and Limitations

### Technical Gaps

1. **No offline support**: Requires backend connection
2. **No service worker**: Could improve load performance
3. **Code splitting**: Each applet lazy-loaded via React.lazy (7 code-split boundaries)
4. **No i18n**: English only
5. **No accessibility audit**: May have a11y issues

### UX Gaps

1. **No empty state guidance**: Could better guide new users
2. **No onboarding**: Assumes user knows the system
3. **Limited error messages**: Could be more helpful
4. **No undo for delete**: Confirmation modal only protection
5. **No message search**: Can't search within conversation

### Testing Gaps

1. **No visual regression tests**: Could add Chromatic/Percy
2. **No performance tests**: Could add Lighthouse CI
3. **Limited mobile E2E**: Tests run on desktop only
4. **No accessibility tests**: Could add axe-core

---

## Implementation Roadmap

### Phase 1: Polish (v0.2.0) - COMPLETE
- [x] Token display per message
- [x] Export/Import conversations
- [x] Theme toggle in UI
- [x] Message timestamps

### Phase 2: Power User (v0.3.0) - PARTIAL
- [x] System prompt UI with presets
- [x] Sampler presets (Balanced, Creative, Precise, Deterministic)
- [ ] Keyboard shortcuts (Cmd+K palette)
- [ ] Search conversations

### Phase 3: Config (v0.4.0)
- models.toml editing via UI
- Backend API endpoints for config
- Model parameter editing

### Phase 4: Rich Content (v0.5.0)
- Markdown rendering
- Code syntax highlighting
- Message reactions/annotations

---

## Feature Request Template

To request a new feature, include:

1. **Description**: What the feature does
2. **Use Case**: Why it's needed
3. **Priority**: High/Medium/Low
4. **Complexity**: Estimate effort
5. **Dependencies**: Backend changes needed?

---

## Test Coverage

874 tests across 38 test files (Vitest + React Testing Library). Test files live alongside their source in each applet and component directory.

Key test areas:
- Stores (chatStore, modelStore, settingsStore, uiStore, batchStore, comparisonStore, explorerStore, performanceStore, notebookStore, modelsStore)
- Chat applet (ChatInput, ChatView, MessageList, EmptyState, ConfirmDeleteModal, Sidebar)
- Shared components (ModelSelector, MarkdownRenderer, Modal, SettingsPanel, layout components)
- Utility libraries (db, color, id, messages, stale, tokens)
- API client

Test utilities live in `src/test/`: `render.tsx` (renderWithProviders), `mocks.ts` (mockFileReader), `setup.ts`.
