// examples/ab_test_dashboard.tsx
// React dashboard for prompt A/B testing

import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface ABTestDashboardProps {
  apiUrl: string;
}

export function ABTestDashboard({ apiUrl }: ABTestDashboardProps) {
  const [activeTests, setActiveTests] = useState<ABTest[]>([]);
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null);
  const [results, setResults] = useState<TestResults | null>(null);

  useEffect(() => {
    fetchActiveTests();
  }, []);

  useEffect(() => {
    if (selectedTest) {
      fetchTestResults(selectedTest.id);
    }
  }, [selectedTest]);

  const fetchActiveTests = async () => {
    const response = await fetch(`${apiUrl}/v1/data/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `
          SELECT * FROM ab_tests 
          WHERE status = 'active' 
          ORDER BY created_at DESC
        `
      })
    });
    const data = await response.json();
    setActiveTests(data.data.map(parseABTest));
  };

  const fetchTestResults = async (testId: string) => {
    const response = await fetch(`${apiUrl}/v1/ab-tests/${testId}/results`);
    const data = await response.json();
    setResults(data);
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Prompt A/B Testing Dashboard</h1>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Test List */}
        <div className="col-span-3">
          <Card>
            <CardHeader>
              <CardTitle>Active Tests</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {activeTests.map(test => (
                  <TestListItem 
                    key={test.id}
                    test={test}
                    isSelected={selectedTest?.id === test.id}
                    onClick={() => setSelectedTest(test)}
                  />
                ))}
              </div>
              <Button className="w-full mt-4" variant="outline">
                Create New Test
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Test Details */}
        <div className="col-span-9">
          {selectedTest && results && (
            <TestDetailsView 
              test={selectedTest} 
              results={results}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function TestDetailsView({ test, results }: { test: ABTest; results: TestResults }) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>{test.name}</CardTitle>
              <p className="text-sm text-gray-500 mt-1">{test.hypothesis}</p>
            </div>
            <Badge variant={test.status === 'active' ? 'default' : 'secondary'}>
              {test.status}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex justify-between items-center">
            <Progress 
              value={(results.totalRuns / test.targetSampleSize) * 100} 
              className="flex-1 mr-4"
            />
            <span className="text-sm text-gray-500">
              {results.totalRuns} / {test.targetSampleSize} runs
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Results Tabs */}
      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="prompts">Prompts</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-2 gap-4">
            {results.variants.map(variant => (
              <VariantCard 
                key={variant.id} 
                variant={variant}
                isWinner={variant.id === results.winnerId}
              />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle>Performance Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={results.timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {results.variants.map((variant, index) => (
                    <Line 
                      key={variant.id}
                      type="monotone" 
                      dataKey={`score_${variant.id}`} 
                      stroke={COLORS[index % COLORS.length]}
                      name={variant.name}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="prompts">
          <div className="space-y-4">
            {results.variants.map(variant => (
              <Card key={variant.id}>
                <CardHeader>
                  <CardTitle>{variant.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-100 p-4 rounded overflow-x-auto">
                    <code>{variant.promptTemplate}</code>
                  </pre>
                  {variant.exampleOutputs && (
                    <div className="mt-4">
                      <h4 className="font-semibold mb-2">Example Outputs:</h4>
                      <div className="space-y-2">
                        {variant.exampleOutputs.slice(0, 3).map((output, i) => (
                          <div key={i} className="bg-gray-50 p-3 rounded text-sm">
                            {output}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <StatisticalAnalysis results={results} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

function VariantCard({ variant, isWinner }: { variant: Variant; isWinner: boolean }) {
  return (
    <Card className={isWinner ? 'ring-2 ring-green-500' : ''}>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">{variant.name}</CardTitle>
          {isWinner && (
            <Badge variant="success">Winner</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <Metric 
            label="Avg Score" 
            value={variant.avgScore.toFixed(2)}
            trend={variant.scoreTrend}
          />
          <Metric 
            label="Success Rate" 
            value={`${(variant.successRate * 100).toFixed(0)}%`}
            trend={variant.successTrend}
          />
          <Metric 
            label="Avg Time" 
            value={`${variant.avgTimeMs.toFixed(0)}ms`}
            trend={variant.timeTrend}
            inverse
          />
          <Metric 
            label="Runs" 
            value={variant.runCount.toString()}
          />
        </div>
        
        {variant.confidence && (
          <div className="mt-4 pt-4 border-t">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Statistical Confidence</span>
              <span className="font-semibold">{(variant.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Metric({ 
  label, 
  value, 
  trend, 
  inverse = false 
}: { 
  label: string; 
  value: string; 
  trend?: 'up' | 'down' | 'flat';
  inverse?: boolean;
}) {
  const getTrendColor = () => {
    if (!trend || trend === 'flat') return 'text-gray-500';
    const isPositive = inverse ? trend === 'down' : trend === 'up';
    return isPositive ? 'text-green-500' : 'text-red-500';
  };

  const getTrendIcon = () => {
    if (!trend || trend === 'flat') return '→';
    return trend === 'up' ? '↑' : '↓';
  };

  return (
    <div>
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-xl font-semibold flex items-center gap-1">
        {value}
        {trend && (
          <span className={`text-sm ${getTrendColor()}`}>
            {getTrendIcon()}
          </span>
        )}
      </p>
    </div>
  );
}

function StatisticalAnalysis({ results }: { results: TestResults }) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Statistical Significance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Winner Determination</h4>
              <p className="text-sm text-gray-600">
                {results.winnerId 
                  ? `Variant "${results.variants.find(v => v.id === results.winnerId)?.name}" 
                     is winning with ${(results.winnerConfidence * 100).toFixed(0)}% confidence.`
                  : "No statistically significant winner yet."}
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Sample Size Analysis</h4>
              <p className="text-sm text-gray-600">
                {results.totalRuns >= results.minimumSampleSize
                  ? "Sufficient sample size reached for reliable results."
                  : `Need ${results.minimumSampleSize - results.totalRuns} more runs for statistical power.`}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {results.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-blue-500 mt-0.5">•</span>
                <span className="text-sm">{rec}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}

// Types
interface ABTest {
  id: string;
  name: string;
  hypothesis: string;
  status: 'active' | 'completed' | 'paused';
  targetSampleSize: number;
  createdAt: Date;
}

interface TestResults {
  totalRuns: number;
  variants: Variant[];
  winnerId?: string;
  winnerConfidence: number;
  timeSeriesData: TimeSeriesPoint[];
  minimumSampleSize: number;
  recommendations: string[];
}

interface Variant {
  id: string;
  name: string;
  promptTemplate: string;
  avgScore: number;
  successRate: number;
  avgTimeMs: number;
  runCount: number;
  confidence?: number;
  scoreTrend?: 'up' | 'down' | 'flat';
  successTrend?: 'up' | 'down' | 'flat';
  timeTrend?: 'up' | 'down' | 'flat';
  exampleOutputs?: string[];
}

interface TimeSeriesPoint {
  timestamp: string;
  [key: string]: number | string;
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c'];

const parseABTest = (row: any[]): ABTest => {
  // Parse DuckDB result row
  return {
    id: row[0],
    name: row[1],
    hypothesis: row[2],
    status: row[3],
    targetSampleSize: row[4],
    createdAt: new Date(row[5])
  };
};