# Technical Specifications

## Technology Stack

### Core Dependencies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Framework | React | 19.x | UI framework |
| Language | TypeScript | 5.x | Type safety |
| State | Zustand | 5.x | State management |
| Styling | Tailwind CSS | 4.x | Utility-first CSS |
| Build | Vite | 7.x | Development server and bundler |
| Storage | idb | 8.x | IndexedDB wrapper |
| Testing | Vitest | 3.x | Unit/component tests |
| E2E Testing | Playwright | 1.x | Browser automation |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| @testing-library/react | Component testing utilities |
| @testing-library/user-event | User interaction simulation |
| jsdom | Browser environment for tests |
| eslint | Code linting |
| concurrently | Run multiple commands |

## API Specification

### Backend Endpoints Used

#### Model Management

```
GET /v1/models
Response: {
  data: [{
    id: string,
    object: "model",
    owned_by: string,
    capabilities: {
      chat: boolean,
      vision: boolean,
      thinking: boolean,
      hidden_states: boolean,
      embeddings: boolean
    }
  }]
}
```

```
GET /v1/capabilities
Response: {
  providers: string[],
  features: {
    streaming: boolean,
    vision: boolean,
    batch: boolean,
    embeddings: boolean,
    hidden_states: boolean
  }
}
```

```
POST /v1/admin/reload
Request: {}
Response: { status: "ok" }
```

#### Chat Completion

```
POST /v1/chat/completions
Request: {
  model: string,
  messages: [{
    role: "system" | "user" | "assistant",
    content: string | ContentItem[]
  }],
  stream?: boolean,
  temperature?: number,
  top_p?: number,
  top_k?: number,
  max_tokens?: number,
  stop?: string[],
  stream_options?: {
    include_usage: boolean
  }
}

ContentItem =
  | { type: "text", text: string }
  | { type: "image_url", image_url: { url: string } }

Response (non-streaming): {
  id: string,
  object: "chat.completion",
  model: string,
  choices: [{
    index: number,
    message: {
      role: "assistant",
      content: string,
      thinking?: string
    },
    finish_reason: string
  }],
  usage: {
    prompt_tokens: number,
    completion_tokens: number,
    total_tokens: number
  }
}

Response (streaming): SSE events with:
data: {
  choices: [{
    delta: {
      content?: string,
      thinking?: string
    }
  }]
}
data: [DONE]
```

#### Multipart Image Upload

```
POST /v1/chat/completions/multipart
Content-Type: multipart/form-data

Fields:
  model: string
  messages: JSON string
  images: File[] (multiple)
  resize_max?: string (default: "1024")
  image_quality?: string (default: "85")
  temperature?: string
  max_tokens?: string
  ...other params

Response: Same as chat completion
```

## Data Models

### Conversation

```typescript
interface Conversation {
  id: string;                    // UUID-like: timestamp-random
  title: string;                 // Auto-generated from first message
  modelId: string;               // Model used for this conversation
  messages: Message[];           // Ordered message list
  systemPrompt?: string;         // Optional system prompt
  createdAt: number;             // Unix timestamp (ms)
  updatedAt: number;             // Unix timestamp (ms)
}
```

### Message

```typescript
interface Message {
  id: string;                    // Unique message ID
  role: 'user' | 'assistant' | 'system';
  content: string;               // Message text
  images?: string[];             // Base64 image data URLs
  thinking?: string;             // Model thinking content (Qwen3)
  tokenCount?: number;           // Completion tokens used
  timestamp: number;             // Unix timestamp (ms)
  isEditing?: boolean;           // Currently being edited
  isRegenerating?: boolean;      // Being regenerated
}
```

### Streaming State

```typescript
interface StreamingState {
  isStreaming: boolean;          // Currently receiving tokens
  content: string;               // Accumulated content
  thinking: string;              // Accumulated thinking
  messageId: string | null;      // Target message being updated
}
```

### Model

```typescript
interface Model {
  id: string;                    // Model identifier
  object: string;                // Always "model"
  owned_by: string;              // Provider name
  capabilities: ModelCapabilities;
}

interface ModelCapabilities {
  chat: boolean;                 // Supports chat completion
  vision: boolean;               // Supports image input
  thinking: boolean;             // Supports thinking mode
  hidden_states: boolean;        // Supports hidden state extraction
  embeddings: boolean;           // Supports embeddings
}
```

### Settings

```typescript
interface SamplerSettings {
  temperature: number;           // 0.0 - 2.0, default 0.7
  topP: number;                  // 0.0 - 1.0, default 0.9
  topK: number;                  // 1 - 100, default 40
  maxTokens: number;             // 1 - 32768, default 2048
  repetitionPenalty: number;     // 0.0 - 2.0, default 1.1
  stop: string[];                // Stop sequences
}
```

## Storage Specifications

### IndexedDB Schema

```
Database: heylook-db
Version: 1

Object Store: conversations
├── keyPath: id
├── Index: by-updated (updatedAt) - for sorting
└── Index: by-model (modelId) - for filtering
```

### localStorage Keys

| Key | Type | Description |
|-----|------|-------------|
| `heylook:theme` | `"dark" \| "light" \| "auto"` | Theme preference |
| `heylook:settings` | JSON | Sampler settings |
| `heylook:sidebar` | boolean | Sidebar collapsed state |

## Build Configuration

### Vite Config

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': 'http://localhost:8080'
    }
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: undefined
      }
    }
  }
})
```

### TypeScript Config

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "react-jsx",
    "skipLibCheck": true,
    "noEmit": true
  }
}
```

### Tailwind Config

```javascript
// tailwind.config.js
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3b82f6',
          hover: '#2563eb'
        },
        'background-dark': '#0f0f0f',
        'surface-dark': '#1a1a1a',
        'accent-red': '#ef4444',
        'accent-green': '#22c55e',
        'accent-yellow': '#eab308'
      }
    }
  }
}
```

## Testing Configuration

### Vitest Config

```typescript
// vitest.config.ts
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html']
    }
  }
})
```

### Playwright Config

```typescript
// playwright.config.ts
export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry'
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true
  }
})
```

## Performance Metrics

### Bundle Analysis

| Chunk | Size | Gzipped |
|-------|------|---------|
| Main JS | 250.85 KB | 76.29 KB |
| CSS | 39.96 KB | 7.55 KB |
| HTML | 1.09 KB | 0.56 KB |
| **Total** | **291.90 KB** | **84.40 KB** |

### Runtime Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| Initial Load | < 2s | First contentful paint |
| Model List | < 500ms | API call + render |
| Message Send | < 100ms | Until streaming starts |
| Streaming Render | 60fps | Token-by-token updates |
| DB Save | < 50ms | Debounced writes |

## Browser Compatibility

| Browser | Minimum Version | Notes |
|---------|-----------------|-------|
| Chrome | 90+ | Primary target |
| Firefox | 90+ | Fully supported |
| Safari | 15+ | Fully supported |
| Edge | 90+ | Fully supported |

### Required Browser APIs

- IndexedDB
- Fetch API
- EventSource (SSE)
- localStorage
- CSS Custom Properties
- ES2020 features

## Security Considerations

1. **Same-Origin**: Frontend and backend on localhost
2. **No Auth**: Designed for local-only use
3. **CORS**: Backend must allow frontend origin
4. **CSP**: No inline scripts/styles required
5. **XSS**: React's built-in escaping
6. **Data**: All data stored locally in browser
