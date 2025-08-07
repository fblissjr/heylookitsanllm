// src/screens/TestsScreen.tsx
// A/B test management and creation

import React, { useState } from 'react';
import {
  ScrollView,
  View,
  StyleSheet,
  Dimensions,
} from 'react-native';
import {
  FAB,
  Card,
  List,
  Chip,
  Text,
  ProgressBar,
  useTheme,
  Portal,
  Modal,
  TextInput,
  Button,
  SegmentedButtons,
  Surface,
  IconButton,
  Badge,
} from 'react-native-paper';
import { useQuery, useMutation } from '@tanstack/react-query';
import { LineChart } from 'react-native-chart-kit';
import { format } from 'date-fns';

import { useApi } from '../providers/ApiProvider';
import CreateTestModal from '../components/CreateTestModal';
import TestDetailsModal from '../components/TestDetailsModal';
import TestResultsChart from '../components/TestResultsChart';

const { width: screenWidth } = Dimensions.get('window');

interface ABTest {
  id: string;
  name: string;
  hypothesis: string;
  status: 'draft' | 'active' | 'completed';
  targetSampleSize: number;
  currentSampleSize: number;
  createdAt: Date;
  variants: TestVariant[];
  winner?: {
    variantId: string;
    confidence: number;
    uplift: number;
  };
}

interface TestVariant {
  id: string;
  name: string;
  isControl: boolean;
  runs: number;
  avgScore: number;
  successRate: number;
}

export default function TestsScreen() {
  const theme = useTheme();
  const api = useApi();
  
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null);
  const [statusFilter, setStatusFilter] = useState('all');
  
  // Fetch tests
  const { data: tests, refetch } = useQuery({
    queryKey: ['ab-tests', statusFilter],
    queryFn: async () => {
      const whereClause = statusFilter === 'all' ? '' : `WHERE t.status = '${statusFilter}'`;
      
      const response = await api.query(`
        WITH test_stats AS (
          SELECT 
            t.id,
            t.name,
            t.hypothesis,
            t.status,
            t.target_sample_size,
            t.created_at,
            COUNT(DISTINCT r.id) as current_sample_size,
            JSON_OBJECT_AGG(
              v.id,
              JSON_BUILD_OBJECT(
                'id', v.id,
                'name', v.name,
                'isControl', v.is_control,
                'runs', COUNT(r.id),
                'avgScore', AVG(r.auto_score),
                'successRate', SUM(CASE WHEN r.error IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(r.id)
              )
            ) as variants
          FROM ab_tests t
          LEFT JOIN test_variants v ON t.id = v.test_id
          LEFT JOIN test_runs r ON v.id = r.variant_id
          ${whereClause}
          GROUP BY t.id, t.name, t.hypothesis, t.status, t.target_sample_size, t.created_at
        )
        SELECT * FROM test_stats
        ORDER BY created_at DESC
      `);
      
      return response.data.map(parseTest);
    },
  });
  
  // Start test mutation
  const startTest = useMutation({
    mutationFn: async (testId: string) => {
      await api.execute(`UPDATE ab_tests SET status = 'active' WHERE id = ?`, [testId]);
    },
    onSuccess: () => refetch(),
  });
  
  // Complete test mutation
  const completeTest = useMutation({
    mutationFn: async (testId: string) => {
      await api.execute(`UPDATE ab_tests SET status = 'completed' WHERE id = ?`, [testId]);
    },
    onSuccess: () => refetch(),
  });
  
  const getTestIcon = (status: string) => {
    switch (status) {
      case 'draft': return 'file-document-outline';
      case 'active': return 'play-circle';
      case 'completed': return 'check-circle';
      default: return 'help-circle';
    }
  };
  
  const getTestColor = (status: string) => {
    switch (status) {
      case 'draft': return theme.colors.outline;
      case 'active': return theme.colors.primary;
      case 'completed': return theme.colors.tertiary;
      default: return theme.colors.outline;
    }
  };
  
  return (
    <View style={styles.container}>
      {/* Filter Buttons */}
      <View style={styles.filterContainer}>
        <SegmentedButtons
          value={statusFilter}
          onValueChange={setStatusFilter}
          buttons={[
            { value: 'all', label: 'All' },
            { value: 'active', label: 'Active' },
            { value: 'completed', label: 'Completed' },
            { value: 'draft', label: 'Drafts' },
          ]}
        />
      </View>
      
      <ScrollView style={styles.scrollView}>
        {tests?.map((test) => (
          <Card
            key={test.id}
            style={styles.testCard}
            onPress={() => setSelectedTest(test)}
          >
            <Card.Content>
              <View style={styles.testHeader}>
                <View style={styles.testInfo}>
                  <View style={styles.testTitleRow}>
                    <Text variant="titleMedium">{test.name}</Text>
                    <Chip
                      icon={getTestIcon(test.status)}
                      style={{ backgroundColor: getTestColor(test.status) + '20' }}
                      textStyle={{ color: getTestColor(test.status) }}
                    >
                      {test.status}
                    </Chip>
                  </View>
                  <Text variant="bodySmall" style={styles.hypothesis}>
                    {test.hypothesis}
                  </Text>
                </View>
              </View>
              
              {/* Progress */}
              {test.status === 'active' && (
                <View style={styles.progressSection}>
                  <View style={styles.progressHeader}>
                    <Text variant="labelSmall">Progress</Text>
                    <Text variant="labelSmall">
                      {test.currentSampleSize} / {test.targetSampleSize}
                    </Text>
                  </View>
                  <ProgressBar
                    progress={test.currentSampleSize / test.targetSampleSize}
                    color={theme.colors.primary}
                    style={styles.progressBar}
                  />
                </View>
              )}
              
              {/* Variants */}
              <View style={styles.variantsSection}>
                {test.variants.map((variant) => (
                  <Surface
                    key={variant.id}
                    style={[
                      styles.variantCard,
                      test.winner?.variantId === variant.id && styles.winningVariant,
                    ]}
                    elevation={1}
                  >
                    <View style={styles.variantHeader}>
                      <Text variant="labelMedium">{variant.name}</Text>
                      {variant.isControl && (
                        <Badge size={16}>Control</Badge>
                      )}
                      {test.winner?.variantId === variant.id && (
                        <Badge size={16} style={{ backgroundColor: theme.colors.primary }}>
                          Winner
                        </Badge>
                      )}
                    </View>
                    <View style={styles.variantStats}>
                      <View style={styles.stat}>
                        <Text variant="labelSmall">Score</Text>
                        <Text variant="bodyLarge">
                          {variant.avgScore?.toFixed(2) || 'N/A'}
                        </Text>
                      </View>
                      <View style={styles.stat}>
                        <Text variant="labelSmall">Success</Text>
                        <Text variant="bodyLarge">
                          {((variant.successRate || 0) * 100).toFixed(0)}%
                        </Text>
                      </View>
                      <View style={styles.stat}>
                        <Text variant="labelSmall">Runs</Text>
                        <Text variant="bodyLarge">{variant.runs}</Text>
                      </View>
                    </View>
                  </Surface>
                ))}
              </View>
              
              {/* Actions */}
              <View style={styles.actions}>
                {test.status === 'draft' && (
                  <Button
                    mode="contained"
                    onPress={() => startTest.mutate(test.id)}
                    compact
                  >
                    Start Test
                  </Button>
                )}
                {test.status === 'active' && (
                  <Button
                    mode="outlined"
                    onPress={() => completeTest.mutate(test.id)}
                    compact
                  >
                    End Test
                  </Button>
                )}
                {test.winner && (
                  <Text variant="bodySmall" style={styles.winnerText}>
                    {test.winner.confidence * 100}% confidence • 
                    {test.winner.uplift > 0 ? '+' : ''}{(test.winner.uplift * 100).toFixed(1)}% uplift
                  </Text>
                )}
              </View>
            </Card.Content>
          </Card>
        ))}
      </ScrollView>
      
      {/* Create FAB */}
      <FAB
        icon="plus"
        style={styles.fab}
        onPress={() => setShowCreateModal(true)}
        label="New Test"
      />
      
      {/* Modals */}
      <Portal>
        <CreateTestModal
          visible={showCreateModal}
          onDismiss={() => setShowCreateModal(false)}
          onCreated={() => {
            setShowCreateModal(false);
            refetch();
          }}
        />
        
        {selectedTest && (
          <TestDetailsModal
            visible={!!selectedTest}
            test={selectedTest}
            onDismiss={() => setSelectedTest(null)}
          />
        )}
      </Portal>
    </View>
  );
}

function parseTest(row: any[]): ABTest {
  return {
    id: row[0],
    name: row[1],
    hypothesis: row[2],
    status: row[3],
    targetSampleSize: row[4],
    createdAt: new Date(row[5]),
    currentSampleSize: row[6],
    variants: Object.values(row[7] || {}),
    winner: row[8] ? JSON.parse(row[8]) : undefined,
  };
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  filterContainer: {
    padding: 16,
  },
  scrollView: {
    flex: 1,
  },
  testCard: {
    margin: 16,
    marginTop: 0,
  },
  testHeader: {
    marginBottom: 12,
  },
  testInfo: {
    flex: 1,
  },
  testTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  hypothesis: {
    opacity: 0.7,
  },
  progressSection: {
    marginVertical: 12,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  variantsSection: {
    marginTop: 12,
    gap: 8,
  },
  variantCard: {
    padding: 12,
    borderRadius: 8,
  },
  winningVariant: {
    borderWidth: 2,
    borderColor: 'rgba(0, 200, 0, 0.3)',
  },
  variantHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  variantStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  stat: {
    alignItems: 'center',
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 16,
  },
  winnerText: {
    color: 'green',
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
});