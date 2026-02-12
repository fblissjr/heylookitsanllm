# Architecture Documentation

## System Overview

heylook-frontend is a React single-page application (SPA) that provides a chat interface for the heylookitsanllm local LLM server. The architecture follows a unidirectional data flow pattern with centralized state management.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    React Application                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │   Layout    │  │   Features  │  │     Stores      │   │   │
│  │  │  (Header,   │  │   (Chat,    │  │  (Zustand)      │   │   │
│  │  │  Sidebar)   │  │   Models)   │  │                 │   │   │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │   │
│  │         │                │                   │            │   │
│  │         └────────────────┼───────────────────┘            │   │
│  │                          │                                │   │
│  │                    ┌─────▼─────┐                          │   │
│  │                    │    API    │                          │   │
│  │                    │  Layer    │                          │   │
│  │                    └─────┬─────┘                          │   │
│  └──────────────────────────┼───────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┼───────────────────────────────┐   │
│  │         IndexedDB        │        localStorage            │   │
│  │      (Conversations)     │        (Preferences)           │   │
│  └──────────────────────────┼───────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  heylookitsanllm Backend                         │
│                   (localhost:8080)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   FastAPI   │  │   Router    │  │      Providers          │  │
│  │  Endpoints  │◄─┤   (LRU)     │◄─┤  (MLX, llama.cpp, STT)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│  Components: Header, Sidebar, ChatView, ModelSelector, etc.  │
├─────────────────────────────────────────────────────────────┤
│                      State Layer                             │
│  Stores: chatStore, modelStore, settingsStore, uiStore       │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│  API Client, Streaming Handler, IndexedDB Operations         │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
App
├── ThemeProvider
│   └── Layout
│       ├── Header
│       │   ├── SidebarToggle
│       │   ├── ModelButton (opens ModelSelector)
│       │   └── SettingsButtons
│       ├── Sidebar
│       │   ├── NewChatButton
│       │   └── ConversationList
│       │       └── ConversationItem (per conversation)
│       └── Main Content
│           └── ChatView
│               ├── MessageList
│               │   └── MessageItem (per message)
│               ├── ChatInput
│               └── EmptyState (when no conversation)
├── ModelSelector (slide-out panel)
└── ConfirmDeleteModal
```

## State Management

### Store Architecture (Zustand)

```
┌──────────────────────────────────────────────────────────────────┐
│                         State Stores                              │
├────────────────┬─────────────────┬───────────────┬───────────────┤
│   chatStore    │   modelStore    │ settingsStore │   uiStore     │
├────────────────┼─────────────────┼───────────────┼───────────────┤
│ conversations  │ models[]        │ temperature   │ sidebarOpen   │
│ activeConvId   │ loadedModel     │ topP          │ activePanel   │
│ streaming      │ loadingState    │ topK          │ confirmDelete │
│ editState      │ error           │ maxTokens     │               │
├────────────────┼─────────────────┼───────────────┼───────────────┤
│ CRUD ops       │ fetchModels()   │ updateParam() │ toggleSidebar │
│ sendMessage()  │ loadModel()     │ resetToPreset │ openPanel()   │
│ stopGen()      │ unloadModel()   │               │ closePanel()  │
└────────────────┴─────────────────┴───────────────┴───────────────┘
```

### Data Flow

```
User Action → Component → Store Action → State Update → Re-render
     │                         │
     │                         ├──→ API Call (if needed)
     │                         │         │
     │                         │         ▼
     │                         │    Backend Response
     │                         │         │
     │                         ◄─────────┘
     │
     └──→ IndexedDB Persist (debounced)
```

## API Communication

### Request Flow

```
Component                Store               API Layer              Backend
    │                      │                     │                     │
    │  sendMessage()       │                     │                     │
    │─────────────────────►│                     │                     │
    │                      │  streamChat()       │                     │
    │                      │────────────────────►│                     │
    │                      │                     │  POST /v1/chat/...  │
    │                      │                     │────────────────────►│
    │                      │                     │                     │
    │                      │                     │◄──── SSE stream ────│
    │                      │                     │                     │
    │                      │◄─── onToken() ─────│                     │
    │                      │                     │                     │
    │◄── state update ─────│                     │                     │
    │                      │                     │                     │
```

### Streaming Architecture

```typescript
// SSE Event Stream Processing
EventSource Connection
    │
    ├─── data: {"choices":[{"delta":{"content":"Hello"}}]}
    │         │
    │         └──► onToken("Hello") → appendStreamContent()
    │
    ├─── data: {"choices":[{"delta":{"thinking":"..."}}]}
    │         │
    │         └──► onThinking("...") → appendStreamContent(isThinking=true)
    │
    └─── data: [DONE]
              │
              └──► onComplete() → finalizeStream()
```

## Persistence Layer

### Storage Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                      IndexedDB                               │
│  Database: heylook-db                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Object Store: conversations                           │  │
│  │  Key: id (string)                                      │  │
│  │  Indexes: by-updated (timestamp), by-model (modelId)   │  │
│  │                                                        │  │
│  │  Schema:                                               │  │
│  │  {                                                     │  │
│  │    id: string,                                         │  │
│  │    title: string,                                      │  │
│  │    modelId: string,                                    │  │
│  │    messages: Message[],                                │  │
│  │    systemPrompt?: string,                              │  │
│  │    createdAt: number,                                  │  │
│  │    updatedAt: number                                   │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      localStorage                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  heylook:theme        → "dark" | "light" | "auto"     │  │
│  │  heylook:settings     → SamplerSettings JSON          │  │
│  │  heylook:sidebar      → boolean (collapsed state)     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Persistence Timing

- **Conversations**: Debounced save (500ms) after each change
- **Settings**: Immediate save on change
- **Theme**: Immediate save on toggle
- **Load**: On app startup, conversations loaded from IndexedDB

## Error Handling

### Error Boundary Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Error Handling Layers                     │
├─────────────────────────────────────────────────────────────┤
│  1. API Layer                                                │
│     - Network errors → "Connection failed" state            │
│     - HTTP errors → APIError class with status/body         │
│     - Timeout → AbortController cancellation                │
├─────────────────────────────────────────────────────────────┤
│  2. Store Layer                                              │
│     - Catch API errors → Set error state                    │
│     - Log to console                                        │
│     - Reset loading states                                  │
├─────────────────────────────────────────────────────────────┤
│  3. Component Layer                                          │
│     - Display error states                                  │
│     - Retry actions                                         │
│     - Graceful degradation                                  │
└─────────────────────────────────────────────────────────────┘
```

## Performance Considerations

### Optimization Strategies

1. **Debounced Persistence**: Conversation saves are debounced to avoid excessive IndexedDB writes during streaming

2. **Streaming Chunks**: Messages update incrementally during streaming rather than full re-renders

3. **Lazy Loading**: Model selector panel only renders when opened

4. **Memoization**: Message list items are keyed by message ID for efficient React reconciliation

5. **Abort Controllers**: Streaming can be cancelled to free up backend resources

### Bundle Size

```
Production Build:
├── index.html          1.09 KB
├── index.css          39.96 KB (gzip: 7.55 KB)
└── index.js          250.85 KB (gzip: 76.29 KB)
```

## Security Considerations

1. **No Authentication**: Designed for local use only (localhost)
2. **CORS**: Backend must allow frontend origin
3. **Input Sanitization**: User messages sent as-is to backend
4. **Image Handling**: Base64 images stored in conversations
5. **No Secrets**: No API keys or credentials stored

## Testing Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Pyramid                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                        E2E Tests                             │
│                     (Playwright - 48)                        │
│                    ┌────────────────┐                        │
│                   /                  \                       │
│                  /   Integration      \                      │
│                 /    with Backend      \                     │
│                └────────────────────────┘                    │
│                                                              │
│             ┌────────────────────────────────┐               │
│            /     Component Tests (Vitest)     \              │
│           /        React Testing Library       \             │
│          /              (300+)                  \            │
│         └────────────────────────────────────────┘           │
│                                                              │
│      ┌────────────────────────────────────────────────┐      │
│     /              Unit Tests (Vitest)                 \     │
│    /         Store logic, utilities, API client         \    │
│   /                     (170+)                           \   │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Applet Platform (v1.9.0+)

The frontend uses a lazy-loaded applet architecture with 6 applets:

| Applet | Route | Store | Tests |
|--------|-------|-------|-------|
| Chat | `/` | chatStore (applet-owned) | ~200 |
| Batch | `/batch` | batchStore (applet-owned) | 18 |
| Token Explorer | `/explore` | explorerStore (applet-owned) | 21 |
| Model Comparison | `/compare` | comparisonStore (applet-owned) | 33 |
| Performance | `/perf` | performanceStore (applet-owned) | ~50 |
| Notebook | `/notebook` | notebookStore (applet-owned) | ~50 |

Shared stores (model, settings, UI) remain in `src/stores/`.
Shared components live in `src/components/` (primitives/, composed/, icons/, layout/).
Each applet is lazy-loaded via `React.lazy` for code splitting.

**Total: 711 tests across 31 test files.**

## Future Considerations

See [FEATURES.md](./FEATURES.md) for planned features and known gaps.
