// src/screens/QueryScreen.tsx
// Interactive DuckDB query interface with smart suggestions

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  FlatList,
  TouchableOpacity,
} from 'react-native';
import {
  TextInput,
  Button,
  Card,
  DataTable,
  FAB,
  Portal,
  Dialog,
  List,
  Chip,
  Searchbar,
  Menu,
  Divider,
  Text,
  Surface,
  IconButton,
  useTheme,
  Snackbar,
} from 'react-native-paper';
import { useMutation, useQuery } from '@tanstack/react-query';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { useApi } from '../providers/ApiProvider';
import { useDuckDB } from '../providers/DuckDBProvider';
import QueryResultsView from '../components/QueryResultsView';
import QueryTemplates from '../components/QueryTemplates';
import QueryHistory from '../components/QueryHistory';

interface SavedQuery {
  id: string;
  name: string;
  query: string;
  description?: string;
  tags: string[];
  createdAt: Date;
  lastUsed?: Date;
}

export default function QueryScreen() {
  const theme = useTheme();
  const api = useApi();
  const db = useDuckDB();
  
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTemplates, setShowTemplates] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [savedQueryName, setSavedQueryName] = useState('');
  const [savedQueryTags, setSavedQueryTags] = useState('');
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  
  const queryInputRef = useRef<any>(null);
  
  // Load saved queries
  const { data: savedQueries, refetch: refetchQueries } = useQuery({
    queryKey: ['saved-queries'],
    queryFn: async () => {
      const saved = await AsyncStorage.getItem('saved_queries');
      return saved ? JSON.parse(saved) : [];
    },
  });
  
  // Execute query mutation
  const executeQuery = useMutation({
    mutationFn: async (sql: string) => {
      const response = await api.query(sql);
      return response;
    },
    onSuccess: (data) => {
      setResults(data);
      setError(null);
      // Save to history
      saveToHistory(query);
    },
    onError: (err: any) => {
      setError(err.message || 'Query failed');
      setResults(null);
    },
  });
  
  const handleExecute = () => {
    if (!query.trim()) return;
    executeQuery.mutate(query);
  };
  
  const saveToHistory = async (sql: string) => {
    try {
      const history = await AsyncStorage.getItem('query_history');
      const items = history ? JSON.parse(history) : [];
      items.unshift({
        id: Date.now().toString(),
        query: sql,
        timestamp: new Date().toISOString(),
      });
      // Keep last 50 queries
      await AsyncStorage.setItem('query_history', JSON.stringify(items.slice(0, 50)));
    } catch (error) {
      console.error('Failed to save to history:', error);
    }
  };
  
  const handleSaveQuery = async () => {
    if (!savedQueryName.trim()) return;
    
    const newQuery: SavedQuery = {
      id: Date.now().toString(),
      name: savedQueryName,
      query: query,
      tags: savedQueryTags.split(',').map(t => t.trim()).filter(Boolean),
      createdAt: new Date(),
    };
    
    const existing = savedQueries || [];
    const updated = [...existing, newQuery];
    
    await AsyncStorage.setItem('saved_queries', JSON.stringify(updated));
    await refetchQueries();
    
    setShowSaveDialog(false);
    setSavedQueryName('');
    setSavedQueryTags('');
    setSnackbarMessage('Query saved successfully');
    setShowSnackbar(true);
  };
  
  const loadQuery = (sql: string) => {
    setQuery(sql);
    setShowTemplates(false);
    setShowHistory(false);
  };
  
  // Query templates
  const templates = [
    {
      category: 'Performance',
      queries: [
        {
          name: 'Slow Requests',
          query: `SELECT * FROM request_logs 
WHERE total_time_ms > 1000 
ORDER BY total_time_ms DESC 
LIMIT 20`,
        },
        {
          name: 'Model Performance',
          query: `SELECT 
  model,
  COUNT(*) as requests,
  AVG(total_time_ms) as avg_time,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_time,
  AVG(tokens_per_second) as avg_tps
FROM request_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY model
ORDER BY requests DESC`,
        },
        {
          name: 'Error Analysis',
          query: `SELECT 
  error_type,
  model,
  COUNT(*) as count,
  MAX(timestamp) as last_seen
FROM request_logs
WHERE error IS NOT NULL
GROUP BY error_type, model
ORDER BY count DESC`,
        },
      ],
    },
    {
      category: 'A/B Testing',
      queries: [
        {
          name: 'Active Tests',
          query: `SELECT * FROM ab_tests WHERE status = 'active'`,
        },
        {
          name: 'Test Results',
          query: `SELECT 
  t.name as test_name,
  v.name as variant_name,
  COUNT(r.id) as runs,
  AVG(r.auto_score) as avg_score,
  AVG(r.response_time_ms) as avg_time
FROM ab_tests t
JOIN test_variants v ON t.id = v.test_id
JOIN test_runs r ON v.id = r.variant_id
GROUP BY t.name, v.name
ORDER BY t.name, avg_score DESC`,
        },
      ],
    },
    {
      category: 'Usage Analytics',
      queries: [
        {
          name: 'Hourly Usage',
          query: `SELECT 
  DATE_TRUNC('hour', timestamp) as hour,
  COUNT(*) as requests,
  COUNT(DISTINCT model) as unique_models
FROM request_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour`,
        },
        {
          name: 'Token Usage',
          query: `SELECT 
  model,
  SUM(prompt_tokens) as total_prompt_tokens,
  SUM(completion_tokens) as total_completion_tokens,
  SUM(prompt_tokens + completion_tokens) as total_tokens
FROM request_logs
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY model
ORDER BY total_tokens DESC`,
        },
      ],
    },
  ];
  
  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView style={styles.scrollView}>
        {/* Query Input */}
        <Card style={styles.queryCard}>
          <Card.Content>
            <View style={styles.queryHeader}>
              <Text variant="titleMedium">SQL Query</Text>
              <View style={styles.queryActions}>
                <IconButton
                  icon="content-save"
                  size={20}
                  onPress={() => setShowSaveDialog(true)}
                />
                <IconButton
                  icon="history"
                  size={20}
                  onPress={() => setShowHistory(!showHistory)}
                />
                <IconButton
                  icon="book-open"
                  size={20}
                  onPress={() => setShowTemplates(!showTemplates)}
                />
              </View>
            </View>
            
            <TextInput
              ref={queryInputRef}
              mode="outlined"
              multiline
              numberOfLines={6}
              value={query}
              onChangeText={setQuery}
              placeholder="SELECT * FROM request_logs LIMIT 10"
              style={styles.queryInput}
              contentStyle={styles.queryInputContent}
            />
            
            <Button
              mode="contained"
              onPress={handleExecute}
              loading={executeQuery.isPending}
              disabled={!query.trim()}
              style={styles.executeButton}
            >
              Execute Query
            </Button>
          </Card.Content>
        </Card>
        
        {/* Templates */}
        {showTemplates && (
          <Card style={styles.templatesCard}>
            <Card.Title title="Query Templates" />
            <Card.Content>
              {templates.map((category) => (
                <View key={category.category}>
                  <Text variant="titleSmall" style={styles.categoryTitle}>
                    {category.category}
                  </Text>
                  {category.queries.map((template) => (
                    <List.Item
                      key={template.name}
                      title={template.name}
                      onPress={() => loadQuery(template.query)}
                      left={(props) => <List.Icon {...props} icon="file-code" />}
                    />
                  ))}
                </View>
              ))}
            </Card.Content>
          </Card>
        )}
        
        {/* History */}
        {showHistory && <QueryHistory onSelect={loadQuery} />}
        
        {/* Saved Queries */}
        {savedQueries && savedQueries.length > 0 && (
          <Card style={styles.savedCard}>
            <Card.Title title="Saved Queries" />
            <Card.Content>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                {savedQueries.map((saved: SavedQuery) => (
                  <Chip
                    key={saved.id}
                    onPress={() => loadQuery(saved.query)}
                    style={styles.savedChip}
                  >
                    {saved.name}
                  </Chip>
                ))}
              </ScrollView>
            </Card.Content>
          </Card>
        )}
        
        {/* Error Display */}
        {error && (
          <Card style={[styles.errorCard, { backgroundColor: theme.colors.errorContainer }]}>
            <Card.Content>
              <Text style={{ color: theme.colors.onErrorContainer }}>{error}</Text>
            </Card.Content>
          </Card>
        )}
        
        {/* Results */}
        {results && <QueryResultsView results={results} />}
      </ScrollView>
      
      {/* Save Query Dialog */}
      <Portal>
        <Dialog visible={showSaveDialog} onDismiss={() => setShowSaveDialog(false)}>
          <Dialog.Title>Save Query</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Query Name"
              value={savedQueryName}
              onChangeText={setSavedQueryName}
              mode="outlined"
              style={styles.dialogInput}
            />
            <TextInput
              label="Tags (comma separated)"
              value={savedQueryTags}
              onChangeText={setSavedQueryTags}
              mode="outlined"
              style={styles.dialogInput}
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowSaveDialog(false)}>Cancel</Button>
            <Button onPress={handleSaveQuery}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
      
      <Snackbar
        visible={showSnackbar}
        onDismiss={() => setShowSnackbar(false)}
        duration={3000}
      >
        {snackbarMessage}
      </Snackbar>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  queryCard: {
    margin: 16,
  },
  queryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  queryActions: {
    flexDirection: 'row',
  },
  queryInput: {
    marginBottom: 16,
  },
  queryInputContent: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 14,
  },
  executeButton: {
    marginTop: 8,
  },
  templatesCard: {
    margin: 16,
    marginTop: 0,
  },
  categoryTitle: {
    marginTop: 16,
    marginBottom: 8,
  },
  savedCard: {
    margin: 16,
    marginTop: 0,
  },
  savedChip: {
    marginRight: 8,
  },
  errorCard: {
    margin: 16,
    marginTop: 0,
  },
  dialogInput: {
    marginBottom: 16,
  },
});