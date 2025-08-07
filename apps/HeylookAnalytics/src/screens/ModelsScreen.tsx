// src/screens/ModelsScreen.tsx
// Model management and performance comparison

import React, { useState } from 'react';
import {
  ScrollView,
  View,
  StyleSheet,
  Dimensions,
} from 'react-native';
import {
  Card,
  List,
  Chip,
  Text,
  useTheme,
  DataTable,
  IconButton,
  Surface,
  Badge,
  FAB,
  Portal,
  Dialog,
  Button,
  TextInput,
} from 'react-native-paper';
import { BarChart, RadarChart } from 'react-native-chart-kit';
import { useQuery } from '@tanstack/react-query';

import { useApi } from '../providers/ApiProvider';
import ModelComparisonChart from '../components/ModelComparisonChart';
import ModelTestRunner from '../components/ModelTestRunner';

const { width: screenWidth } = Dimensions.get('window');

interface ModelStats {
  model: string;
  provider: string;
  requests: number;
  avgResponseTime: number;
  p95ResponseTime: number;
  avgTokensPerSecond: number;
  errorRate: number;
  capabilities: {
    vision: boolean;
    streaming: boolean;
    functionCalling: boolean;
  };
  costPer1kTokens: number;
}

export default function ModelsScreen() {
  const theme = useTheme();
  const api = useApi();
  
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [showTestRunner, setShowTestRunner] = useState(false);
  const [comparisonMetric, setComparisonMetric] = useState<
    'speed' | 'quality' | 'cost' | 'reliability'
  >('speed');
  
  // Fetch model statistics
  const { data: modelStats } = useQuery({
    queryKey: ['model-stats'],
    queryFn: async () => {
      const [stats, models] = await Promise.all([
        api.query(`
          SELECT 
            model,
            COUNT(*) as requests,
            AVG(total_time_ms) as avg_response_time,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_response_time,
            AVG(tokens_per_second) as avg_tokens_per_second,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as error_rate
          FROM request_logs
          WHERE timestamp > NOW() - INTERVAL '7 days'
          GROUP BY model
        `),
        api.client.get('/v1/models'),
      ]);
      
      // Merge stats with model info
      return stats.data.map((row: any[]) => ({
        model: row[0],
        provider: getProviderFromModel(row[0]),
        requests: row[1],
        avgResponseTime: row[2],
        p95ResponseTime: row[3],
        avgTokensPerSecond: row[4],
        errorRate: row[5],
        capabilities: getModelCapabilities(row[0], models.data.data),
        costPer1kTokens: getModelCost(row[0]),
      }));
    },
  });
  
  // Performance comparison data
  const comparisonData = {
    labels: selectedModels,
    datasets: [
      {
        label: 'Speed (tokens/s)',
        data: selectedModels.map(
          (model) => modelStats?.find((m) => m.model === model)?.avgTokensPerSecond || 0
        ),
        backgroundColor: theme.colors.primary + '80',
      },
    ],
  };
  
  // Radar chart data for comprehensive comparison
  const radarData = {
    labels: ['Speed', 'Reliability', 'Cost Efficiency', 'Features', 'Consistency'],
    datasets: selectedModels.map((model, index) => {
      const stats = modelStats?.find((m) => m.model === model);
      return {
        label: model,
        data: [
          normalizeMetric(stats?.avgTokensPerSecond || 0, 0, 100), // Speed
          normalizeMetric(1 - (stats?.errorRate || 0), 0, 1) * 100, // Reliability
          normalizeMetric(1 / (stats?.costPer1kTokens || 1), 0, 10) * 100, // Cost efficiency
          Object.values(stats?.capabilities || {}).filter(Boolean).length * 25, // Features
          normalizeMetric(1 / ((stats?.p95ResponseTime || 1) / (stats?.avgResponseTime || 1)), 0, 2) * 100, // Consistency
        ],
        borderColor: CHART_COLORS[index % CHART_COLORS.length],
        backgroundColor: CHART_COLORS[index % CHART_COLORS.length] + '20',
      };
    }),
  };
  
  const toggleModelSelection = (model: string) => {
    setSelectedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
  };
  
  const getModelIcon = (provider: string) => {
    switch (provider) {
      case 'mlx':
        return 'apple';
      case 'llama_cpp':
        return 'language-cpp';
      default:
        return 'robot';
    }
  };
  
  return (
    <ScrollView style={styles.container}>
      {/* Model List */}
      <Card style={styles.card}>
        <Card.Title 
          title="Available Models" 
          subtitle={`${modelStats?.length || 0} models • ${selectedModels.length} selected`}
        />
        <Card.Content>
          {modelStats?.map((model) => (
            <Surface
              key={model.model}
              style={[
                styles.modelItem,
                selectedModels.includes(model.model) && styles.selectedModel,
              ]}
              elevation={selectedModels.includes(model.model) ? 2 : 0}
            >
              <List.Item
                title={model.model}
                description={`${model.requests} requests • ${model.avgTokensPerSecond.toFixed(1)} tok/s`}
                left={(props) => (
                  <List.Icon {...props} icon={getModelIcon(model.provider)} />
                )}
                right={() => (
                  <View style={styles.modelRight}>
                    <View style={styles.capabilities}>
                      {model.capabilities.vision && (
                        <Chip compact icon="eye">Vision</Chip>
                      )}
                      {model.capabilities.streaming && (
                        <Chip compact icon="water">Stream</Chip>
                      )}
                    </View>
                    <IconButton
                      icon={selectedModels.includes(model.model) ? 'check-circle' : 'circle-outline'}
                      onPress={() => toggleModelSelection(model.model)}
                    />
                  </View>
                )}
                onPress={() => toggleModelSelection(model.model)}
              />
              
              {/* Performance Indicators */}
              <View style={styles.performanceRow}>
                <View style={styles.metric}>
                  <Text variant="labelSmall">Avg Response</Text>
                  <Text variant="bodySmall">{model.avgResponseTime.toFixed(0)}ms</Text>
                </View>
                <View style={styles.metric}>
                  <Text variant="labelSmall">P95 Response</Text>
                  <Text variant="bodySmall">{model.p95ResponseTime.toFixed(0)}ms</Text>
                </View>
                <View style={styles.metric}>
                  <Text variant="labelSmall">Error Rate</Text>
                  <Text 
                    variant="bodySmall"
                    style={{ color: model.errorRate > 0.05 ? theme.colors.error : theme.colors.primary }}
                  >
                    {(model.errorRate * 100).toFixed(1)}%
                  </Text>
                </View>
                <View style={styles.metric}>
                  <Text variant="labelSmall">Cost/1k tok</Text>
                  <Text variant="bodySmall">${model.costPer1kTokens.toFixed(3)}</Text>
                </View>
              </View>
            </Surface>
          ))}
        </Card.Content>
      </Card>
      
      {/* Comparison Charts */}
      {selectedModels.length >= 2 && (
        <>
          <Card style={styles.card}>
            <Card.Title title="Speed Comparison" />
            <Card.Content>
              <BarChart
                data={comparisonData}
                width={screenWidth - 64}
                height={200}
                yAxisLabel=""
                yAxisSuffix=" tok/s"
                chartConfig={{
                  backgroundColor: theme.colors.surface,
                  backgroundGradientFrom: theme.colors.surface,
                  backgroundGradientTo: theme.colors.surface,
                  decimalPlaces: 1,
                  color: (opacity = 1) => theme.colors.primary,
                  labelColor: (opacity = 1) => theme.colors.onSurface,
                  style: {
                    borderRadius: 16,
                  },
                }}
                style={styles.chart}
              />
            </Card.Content>
          </Card>
          
          <Card style={styles.card}>
            <Card.Title title="Comprehensive Comparison" />
            <Card.Content>
              <RadarChart
                data={radarData}
                width={screenWidth - 64}
                height={250}
                chartConfig={{
                  backgroundColor: theme.colors.surface,
                  backgroundGradientFrom: theme.colors.surface,
                  backgroundGradientTo: theme.colors.surface,
                  color: (opacity = 1, index = 0) => CHART_COLORS[index % CHART_COLORS.length],
                  labelColor: (opacity = 1) => theme.colors.onSurface,
                  strokeWidth: 2,
                }}
                style={styles.chart}
              />
            </Card.Content>
          </Card>
        </>
      )}
      
      {/* Quick Actions */}
      <View style={styles.actions}>
        <Button
          mode="contained"
          onPress={() => setShowTestRunner(true)}
          disabled={selectedModels.length === 0}
          icon="play"
        >
          Run Benchmark
        </Button>
        
        <Button
          mode="outlined"
          onPress={() => {/* Export comparison */}}
          disabled={selectedModels.length === 0}
          icon="export"
        >
          Export Comparison
        </Button>
      </View>
      
      {/* Test Runner Modal */}
      <Portal>
        <ModelTestRunner
          visible={showTestRunner}
          models={selectedModels}
          onDismiss={() => setShowTestRunner(false)}
        />
      </Portal>
    </ScrollView>
  );
}

// Helper functions
function getProviderFromModel(modelId: string): string {
  if (modelId.includes('mlx')) return 'mlx';
  if (modelId.includes('gguf')) return 'llama_cpp';
  return 'unknown';
}

function getModelCapabilities(modelId: string, models: any[]): any {
  const model = models.find((m) => m.id === modelId);
  return {
    vision: modelId.includes('vision') || modelId.includes('vl'),
    streaming: true, // All models support streaming
    functionCalling: false, // Not implemented yet
  };
}

function getModelCost(modelId: string): number {
  // Simplified cost model - in reality would come from config
  if (modelId.includes('72b')) return 0.001;
  if (modelId.includes('7b')) return 0.0001;
  return 0.0005;
}

function normalizeMetric(value: number, min: number, max: number): number {
  return Math.min(Math.max((value - min) / (max - min), 0), 1);
}

const CHART_COLORS = [
  '#FF6384',
  '#36A2EB',
  '#FFCE56',
  '#4BC0C0',
  '#9966FF',
  '#FF9F40',
];

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  card: {
    margin: 16,
    marginBottom: 0,
  },
  modelItem: {
    marginVertical: 4,
    borderRadius: 8,
    overflow: 'hidden',
  },
  selectedModel: {
    borderWidth: 2,
    borderColor: '#4BC0C0',
  },
  modelRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  capabilities: {
    flexDirection: 'row',
    gap: 4,
    marginRight: 8,
  },
  performanceRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 16,
    paddingBottom: 12,
  },
  metric: {
    alignItems: 'center',
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 16,
    gap: 16,
  },
});