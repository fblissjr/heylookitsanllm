# HeylookAnalytics

A React Native mobile app for real-time analytics and monitoring of heylookitsanllm servers.

## Features

### 📊 Dashboard
- Real-time performance metrics
- Request volume charts
- Response time tracking
- Model usage distribution
- Error rate monitoring
- Live activity feed

### 🔍 Query Interface
- Interactive DuckDB SQL editor
- Query templates library
- Save and share queries
- Query history
- Export results (CSV/JSON)
- Syntax highlighting

### 🧪 A/B Testing
- Create and manage A/B tests
- Visual test progress tracking
- Statistical significance calculation
- Winner determination
- Test result visualization
- Export test data

### 🤖 Model Management
- Model performance comparison
- Multi-model benchmarking
- Cost analysis
- Capability matrix
- Performance radar charts
- Head-to-head comparisons

### 💾 Offline Support
- Local SQLite caching
- Offline query execution
- Data sync when online
- Background data updates

## Setup

### Prerequisites
- Node.js 16+
- React Native development environment
- iOS: Xcode 13+
- Android: Android Studio

### Installation

```bash
cd apps/HeylookAnalytics
npm install

# iOS
cd ios && pod install
cd ..

# Run on iOS
npx react-native run-ios

# Run on Android
npx react-native run-android
```

### Configuration

1. Launch the app
2. Go to Settings tab
3. Enter your heylookitsanllm server URL (default: http://localhost:8080)
4. The app will automatically connect and sync

## Usage

### Dashboard
- Pull down to refresh metrics
- Tap time range buttons (1H, 6H, 24H, 7D) to change view
- Scroll horizontally through metric cards
- Tap on model cards for detailed view

### Running Queries
1. Go to Query tab
2. Write SQL query or select from templates
3. Tap "Execute Query"
4. View results in table format
5. Export or save query for later

### Creating A/B Tests
1. Go to Tests tab
2. Tap the "+" FAB button
3. Fill in test details:
   - Test name
   - Hypothesis
   - Prompt variants
   - Target sample size
4. Start test to begin collecting data

### Model Comparison
1. Go to Models tab
2. Select 2+ models to compare
3. View performance charts
4. Run benchmark tests
5. Export comparison data

## Architecture

### Tech Stack
- **React Native** - Cross-platform mobile framework
- **React Navigation** - Navigation
- **React Native Paper** - Material Design components
- **React Query** - Data fetching and caching
- **React Native Chart Kit** - Charts and visualizations
- **AsyncStorage** - Local data persistence
- **SQLite** - Offline database

### Data Flow
```
heylookitsanllm Server
        ↓
    REST API
        ↓
   ApiProvider
    ↓       ↓
DuckDB  Local SQLite
    ↓       ↓
   UI Components
```

### Key Components

#### Providers
- `ApiProvider` - HTTP client for server communication
- `DuckDBProvider` - Local SQLite for offline support

#### Screens
- `DashboardScreen` - Main metrics view
- `QueryScreen` - SQL query interface
- `TestsScreen` - A/B test management
- `ModelsScreen` - Model comparison
- `SettingsScreen` - App configuration

#### Components
- `MetricCard` - Reusable metric display
- `ModelPerformanceCard` - Model stats view
- `QueryResultsView` - Table/chart results
- `TestDetailsModal` - A/B test details
- `CreateTestModal` - New test creation

## API Integration

The app uses these heylookitsanllm endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Run prompts
- `POST /v1/data/query` - Execute DuckDB queries
- `POST /v1/data/load` - Load datasets
- `GET /v1/performance` - Performance metrics

## Offline Capabilities

- **Query History** - All executed queries saved locally
- **Saved Queries** - Named queries with tags
- **Metric Cache** - Recent metrics stored for offline viewing
- **Test Results** - A/B test data cached locally
- **Auto-sync** - Background sync when connection restored

## Performance Tips

1. **Large Queries** - Use LIMIT clause for better performance
2. **Time Ranges** - Shorter ranges load faster
3. **Model Selection** - Compare max 4 models at once
4. **Chart Updates** - Disable auto-refresh in settings if needed

## Troubleshooting

### Connection Issues
- Verify server URL in settings
- Check server is running
- Ensure device is on same network
- Try IP address instead of localhost

### Performance Issues
- Clear app cache in settings
- Reduce query result limits
- Disable background refresh
- Close unused screens

### Data Sync Issues
- Pull to refresh on dashboard
- Check last sync time in settings
- Manually trigger sync
- Clear local cache if corrupted

## Contributing

This is an example app demonstrating heylookitsanllm analytics capabilities. Feel free to extend and customize!