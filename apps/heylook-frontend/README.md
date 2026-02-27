# heylook-frontend

React-based web frontend for the heylookitsanllm local LLM server.

## Overview

A modern, responsive chat interface for interacting with local LLM models via the heylookitsanllm backend. Built with React, TypeScript, Zustand, and Tailwind CSS.

## Features

- **Multi-model support**: Select and switch between MLX text and VLM models
- **Real-time streaming**: Token-by-token response streaming with thinking mode support
- **Conversation management**: Create, switch, rename, and delete conversations
- **Persistent storage**: IndexedDB for conversation history, localStorage for preferences
- **Vision support**: Image attachments for vision-language models
- **Sampler settings**: Adjustable temperature, top_p, top_k, and other parameters
- **Dark theme**: Native dark mode with system preference detection
- **Responsive design**: Works on desktop and mobile browsers

## Quick Start

### Prerequisites

- Node.js 18+
- bun
- heylookitsanllm backend running on localhost:8080

### Installation

```bash
cd apps/heylook-frontend
bun install
```

### Development

```bash
# Start frontend only
bun run dev

# Start frontend + backend together
bun run dev:all

# Start backend separately
bun run dev:backend
```

The app will be available at http://localhost:5173

### Production Build

```bash
bun run build
bun run preview  # Preview production build
```

## Testing

```bash
# Unit and component tests (Vitest)
bun run test        # Watch mode
bun run test:run    # Single run
bun run test:ui     # Vitest UI
bun run test:coverage

# E2E tests (Playwright) - requires backend running
bun run test:e2e
bun run test:e2e:ui
```

### Test Coverage

- **471 unit/component tests** across 17 test files
- **48 E2E tests** across 5 test files
- Stores, components, utilities, and integration flows covered

## Project Structure

```
src/
├── api/                    # Backend API communication
│   ├── client.ts          # HTTP client with error handling
│   ├── endpoints.ts       # API endpoint functions
│   ├── streaming.ts       # SSE streaming handler
│   └── index.ts
├── components/
│   ├── layout/            # App shell components
│   │   ├── Header.tsx     # Top bar with model selector
│   │   ├── Layout.tsx     # Main layout wrapper
│   │   └── Sidebar.tsx    # Conversation list
│   └── modals/
│       └── ConfirmDeleteModal.tsx
├── contexts/
│   └── ThemeContext.tsx   # Dark/light theme provider
├── features/
│   ├── chat/components/   # Chat feature
│   │   ├── ChatInput.tsx  # Message input with attachments
│   │   ├── ChatView.tsx   # Main chat view
│   │   ├── EmptyState.tsx # No conversation state
│   │   └── MessageList.tsx # Message display
│   └── models/components/
│       └── ModelSelector.tsx # Model picker panel
├── lib/
│   └── db.ts              # IndexedDB operations
├── stores/                # Zustand state management
│   ├── chatStore.ts       # Conversations and messages
│   ├── modelStore.ts      # Model state and loading
│   ├── settingsStore.ts   # Sampler parameters
│   ├── uiStore.ts         # UI state (sidebar, panels)
│   └── index.ts
├── types/                 # TypeScript definitions
│   ├── api.ts
│   ├── chat.ts
│   ├── models.ts
│   └── settings.ts
├── App.tsx                # Root component
└── main.tsx               # Entry point
```

## Configuration

### Environment Variables

None required - the app connects to localhost:8080 by default.

To change the backend URL, modify `src/api/endpoints.ts`:

```typescript
const API_BASE = 'http://localhost:8080'
```

### Tailwind Theme

Custom colors defined in `tailwind.config.js`:

- `primary`: Blue accent color
- `background-dark`: Dark theme background
- `surface-dark`: Card/panel backgrounds
- `accent-red`, `accent-green`, `accent-yellow`: Status colors

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed architecture documentation.

## Tech Stack

- **Framework**: React 19 with TypeScript
- **State Management**: Zustand
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **Testing**: Vitest + React Testing Library + Playwright
- **Storage**: IndexedDB (idb) + localStorage

## API Integration

The frontend communicates with the heylookitsanllm backend via:

- `GET /v1/models` - List available models
- `POST /v1/models/load` - Load a model
- `POST /v1/models/unload` - Unload current model
- `POST /v1/chat/completions` - Chat with streaming

See [TECH_SPECS.md](./TECH_SPECS.md) for API details.

## Contributing

1. Run tests before submitting changes
2. Follow existing code style (Prettier, ESLint)
3. Add tests for new functionality
4. Update documentation as needed

## License

MIT License - see root [LICENSE](../../LICENSE) file
