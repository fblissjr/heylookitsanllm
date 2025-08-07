// src/screens/DashboardScreen.tsx
// Main dashboard with real-time metrics and visualizations

import React, { useState, useEffect } from 'react';
import {
  ScrollView,
  View,
  StyleSheet,
  RefreshControl,
  Dimensions,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Surface,
  Text,
  useTheme,
  Chip,
  SegmentedButtons,
  IconButton,
} from 'react-native-paper';
import {
  LineChart,
  BarChart,
  PieChart,
  ProgressChart,
} from 'react-native-chart-kit';
import { useQuery } from '@tanstack/react-query';
import { format, subHours } from 'date-fns';

import { useApi } from '../providers/ApiProvider';
import { useDuckDB } from '../providers/DuckDBProvider';
import MetricCard from '../components/MetricCard';
import ModelPerformanceCard from '../components/ModelPerformanceCard';
import LiveActivityFeed from '../components/LiveActivityFeed';

const { width: screenWidth } = Dimensions.get('window');

export default function DashboardScreen() {
  const theme = useTheme();
  const api = useApi();
  const db = useDuckDB();
  
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState('1h');
  
  // Fetch performance metrics
  const { data: metrics, refetch: refetchMetrics } = useQuery({
    queryKey: ['performance-metrics', timeRange],
    queryFn: async () => {
      const response = await api.query(`
        SELECT 
          model,
          request_type,
          COUNT(*) as requests,
          AVG(total_time_ms) as avg_time,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_time,
          AVG(tokens_per_second) as avg_tps,
          SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
        FROM request_logs
        WHERE timestamp > NOW() - INTERVAL '${timeRange}'
        GROUP BY model, request_type
        ORDER BY requests DESC
      `);
      return response.data;
    },
    refetchInterval: 30000, // Refresh every 30s
  });
  
  // Fetch time series data
  const { data: timeSeries } = useQuery({
    queryKey: ['time-series', timeRange],
    queryFn: async () => {
      const response = await api.query(`
        SELECT 
          time_bucket(INTERVAL '5 minutes', timestamp) as time,
          COUNT(*) as requests,
          AVG(total_time_ms) as avg_response_time
        FROM request_logs
        WHERE timestamp > NOW() - INTERVAL '${timeRange}'
        GROUP BY time
        ORDER BY time
      `);
      return response.data;
    },
  });
  
  // Fetch model distribution
  const { data: modelDistribution } = useQuery({
    queryKey: ['model-distribution', timeRange],
    queryFn: async () => {
      const response = await api.query(`
        SELECT 
          model,
          COUNT(*) as count
        FROM request_logs
        WHERE timestamp > NOW() - INTERVAL '${timeRange}'
        GROUP BY model
      `);
      return response.data;
    },
  });
  
  const onRefresh = async () => {
    setRefreshing(true);
    await refetchMetrics();
    setRefreshing(false);
  };
  
  // Prepare chart data
  const requestsChartData = {
    labels: timeSeries?.map(row => format(new Date(row[0]), 'HH:mm')) || [],
    datasets: [{
      data: timeSeries?.map(row => row[1]) || [],
      color: (opacity = 1) => theme.colors.primary,
      strokeWidth: 2,
    }],
  };
  
  const responseTimeChartData = {
    labels: timeSeries?.map(row => format(new Date(row[0]), 'HH:mm')) || [],
    datasets: [{
      data: timeSeries?.map(row => row[2]) || [],
      color: (opacity = 1) => theme.colors.error,
      strokeWidth: 2,
    }],
  };
  
  const modelPieData = modelDistribution?.map((row, index) => ({
    name: row[0],
    population: row[1],
    color: CHART_COLORS[index % CHART_COLORS.length],
    legendFontColor: theme.colors.onSurface,
    legendFontSize: 12,
  })) || [];
  
  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Time Range Selector */}
      <View style={styles.header}>
        <SegmentedButtons
          value={timeRange}
          onValueChange={setTimeRange}
          buttons={[
            { value: '1h', label: '1H' },
            { value: '6h', label: '6H' },
            { value: '24h', label: '24H' },
            { value: '7d', label: '7D' },
          ]}
          style={styles.segmentedButtons}
        />
      </View>
      
      {/* Summary Metrics */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.metricsRow}
      >
        <MetricCard
          title="Total Requests"
          value={metrics?.reduce((sum, row) => sum + row[2], 0) || 0}
          format="number"
          trend={calculateTrend('requests')}
          icon="chart-line"
        />
        <MetricCard
          title="Avg Response"
          value={calculateAverage(metrics, 3)}
          format="ms"
          trend={calculateTrend('response')}
          icon="speedometer"
        />
        <MetricCard
          title="Tokens/sec"
          value={calculateAverage(metrics, 5)}
          format="number"
          trend={calculateTrend('tps')}
          icon="flash"
        />
        <MetricCard
          title="Error Rate"
          value={calculateErrorRate(metrics)}
          format="percent"
          trend={calculateTrend('errors')}
          icon="alert-circle"
          color={theme.colors.error}
        />
      </ScrollView>
      
      {/* Request Volume Chart */}
      <Card style={styles.card}>
        <Card.Title title="Request Volume" />
        <Card.Content>
          <LineChart
            data={requestsChartData}
            width={screenWidth - 40}
            height={200}
            chartConfig={chartConfig(theme)}
            bezier
            style={styles.chart}
          />
        </Card.Content>
      </Card>
      
      {/* Response Time Chart */}
      <Card style={styles.card}>
        <Card.Title title="Response Time (ms)" />
        <Card.Content>
          <LineChart
            data={responseTimeChartData}
            width={screenWidth - 40}
            height={200}
            chartConfig={{
              ...chartConfig(theme),
              color: (opacity = 1) => theme.colors.error,
            }}
            bezier
            style={styles.chart}
          />
        </Card.Content>
      </Card>
      
      {/* Model Distribution */}
      <Card style={styles.card}>
        <Card.Title title="Model Usage" />
        <Card.Content>
          <PieChart
            data={modelPieData}
            width={screenWidth - 40}
            height={200}
            chartConfig={chartConfig(theme)}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
          />
        </Card.Content>
      </Card>
      
      {/* Model Performance Comparison */}
      <Title style={styles.sectionTitle}>Model Performance</Title>
      {metrics?.map((row, index) => (
        <ModelPerformanceCard
          key={`${row[0]}-${row[1]}`}
          model={row[0]}
          type={row[1]}
          requests={row[2]}
          avgTime={row[3]}
          p95Time={row[4]}
          tokensPerSecond={row[5]}
          errors={row[6]}
        />
      ))}
      
      {/* Live Activity Feed */}
      <Title style={styles.sectionTitle}>Recent Activity</Title>
      <LiveActivityFeed limit={10} />
    </ScrollView>
  );
}

// Helper functions
function calculateAverage(data: any[], columnIndex: number): number {
  if (!data || data.length === 0) return 0;
  const sum = data.reduce((acc, row) => acc + (row[columnIndex] || 0), 0);
  return sum / data.length;
}

function calculateErrorRate(data: any[]): number {
  if (!data || data.length === 0) return 0;
  const totalRequests = data.reduce((sum, row) => sum + row[2], 0);
  const totalErrors = data.reduce((sum, row) => sum + row[6], 0);
  return totalRequests > 0 ? (totalErrors / totalRequests) * 100 : 0;
}

function calculateTrend(metric: string): { direction: 'up' | 'down' | 'neutral', value: number } {
  // TODO: Implement actual trend calculation
  return { direction: 'neutral', value: 0 };
}

const chartConfig = (theme: any) => ({
  backgroundColor: theme.colors.surface,
  backgroundGradientFrom: theme.colors.surface,
  backgroundGradientTo: theme.colors.surface,
  decimalPlaces: 0,
  color: (opacity = 1) => theme.colors.primary,
  labelColor: (opacity = 1) => theme.colors.onSurface,
  style: {
    borderRadius: 16,
  },
  propsForDots: {
    r: '6',
    strokeWidth: '2',
    stroke: theme.colors.primary,
  },
});

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
  header: {
    padding: 16,
  },
  segmentedButtons: {
    marginHorizontal: 16,
  },
  metricsRow: {
    paddingHorizontal: 16,
    marginBottom: 16,
  },
  card: {
    margin: 16,
    marginTop: 0,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  sectionTitle: {
    marginHorizontal: 16,
    marginTop: 24,
    marginBottom: 8,
  },
});