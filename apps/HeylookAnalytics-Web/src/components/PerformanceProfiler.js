import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, TrendingUp, TrendingDown, AlertTriangle, Clock } from 'lucide-react';
import './PerformanceProfiler.css';

const API_BASE = 'http://localhost:8080';

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'];

function PerformanceProfiler() {
  const [timeRange, setTimeRange] = useState('1h');
  const [profileData, setProfileData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchProfileData();
  }, [timeRange]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchProfileData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE}/v1/performance/profile/${timeRange}`);
      
      // Calculate percentages for timing breakdown
      const totalTime = response.data.timing_breakdown.reduce((sum, item) => sum + item.avg_time_ms * item.count, 0);
      response.data.timing_breakdown.forEach(item => {
        item.percentage = (item.avg_time_ms * item.count / totalTime) * 100;
      });

      setProfileData(response.data);
    } catch (err) {
      if (err.response?.status === 503) {
        setError('Analytics not enabled. Enable analytics to use the performance profiler.');
      } else {
        setError(err.response?.data?.detail || 'Failed to fetch performance data');
      }
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (ms) => {
    if (!ms) return '0ms';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatMemory = (gb) => {
    if (!gb) return '0 GB';
    return `${gb.toFixed(2)} GB`;
  };

  const formatPercentage = (value) => {
    if (!value) return '0%';
    return `${value.toFixed(1)}%`;
  };

  const getTrendIcon = (change) => {
    if (Math.abs(change) < 1) return null;
    return change > 0 ? <TrendingUp size={16} color="#ff4444" /> : <TrendingDown size={16} color="#44ff44" />;
  };

  const renderTimingBreakdown = () => {
    if (!profileData?.timing_breakdown?.length) return null;

    return (
      <div className="profile-section">
        <h3>Operation Timing Breakdown</h3>
        <div className="timing-breakdown">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={profileData.timing_breakdown}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={entry => `${entry.operation} (${entry.percentage.toFixed(1)}%)`}
                outerRadius={100}
                dataKey="avg_time_ms"
              >
                {profileData.timing_breakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => formatDuration(value)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          
          <div className="timing-details">
            {profileData.timing_breakdown.map((item, index) => (
              <div key={item.operation} className="timing-item">
                <div 
                  className="timing-color"
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <div className="timing-info">
                  <span className="operation-name">{item.operation}</span>
                  <span className="operation-time">{formatDuration(item.avg_time_ms)}</span>
                  <span className="operation-count">({item.count} requests)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderResourceTimeline = () => {
    if (!profileData?.resource_timeline?.length) return null;

    const chartData = profileData.resource_timeline.map(item => ({
      ...item,
      time: new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }));

    return (
      <div className="profile-section">
        <h3>Resource Utilization Over Time</h3>
        
        <div className="resource-charts">
          <div className="chart-container">
            <h4>Memory Usage</h4>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip formatter={(value) => formatMemory(value)} />
                <Area type="monotone" dataKey="memory_gb" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-container">
            <h4>GPU Utilization</h4>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip formatter={(value) => formatPercentage(value)} />
                <Area type="monotone" dataKey="gpu_percent" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-container">
            <h4>Tokens/Second</h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="tokens_per_second" stroke="#ffc658" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  const renderBottleneckAnalysis = () => {
    if (!profileData?.bottlenecks?.length) return null;

    return (
      <div className="profile-section">
        <h3>Model Performance Bottlenecks</h3>
        
        <div className="bottleneck-grid">
          {profileData.bottlenecks.map(bottleneck => {
            const breakdownData = Object.entries(bottleneck.breakdown)
              .filter(([_, value]) => value > 0)
              .map(([key, value]) => ({
                name: key.replace('_', ' '),
                value: value
              }));

            return (
              <div key={bottleneck.model} className="bottleneck-card">
                <h4>{bottleneck.model}</h4>
                <div className="bottleneck-stats">
                  <div className="stat">
                    <span className="stat-label">Avg Response Time:</span>
                    <span className="stat-value">{formatDuration(bottleneck.avg_total_ms)}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Requests:</span>
                    <span className="stat-value">{bottleneck.request_count}</span>
                  </div>
                </div>
                
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={breakdownData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(value) => formatDuration(value)} />
                    <Bar dataKey="value" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderPerformanceTrends = () => {
    if (!profileData?.trends?.length) return null;

    const chartData = profileData.trends.map(item => ({
      ...item,
      time: new Date(item.hour).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }));

    return (
      <div className="profile-section">
        <h3>Performance Trends</h3>
        
        <div className="trends-summary">
          {profileData.trends.length > 0 && (
            <>
              <div className="trend-card">
                <div className="trend-header">
                  <Clock size={20} />
                  <span>Response Time Trend</span>
                </div>
                <div className="trend-value">
                  {formatDuration(profileData.trends[profileData.trends.length - 1].response_time_ms)}
                  <span className={`trend-change ${profileData.trends[profileData.trends.length - 1].response_time_change > 0 ? 'negative' : 'positive'}`}>
                    {getTrendIcon(profileData.trends[profileData.trends.length - 1].response_time_change)}
                    {Math.abs(profileData.trends[profileData.trends.length - 1].response_time_change).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="trend-card">
                <div className="trend-header">
                  <Activity size={20} />
                  <span>Throughput Trend</span>
                </div>
                <div className="trend-value">
                  {profileData.trends[profileData.trends.length - 1].tokens_per_second?.toFixed(1)} TPS
                  <span className={`trend-change ${profileData.trends[profileData.trends.length - 1].tps_change < 0 ? 'negative' : 'positive'}`}>
                    {getTrendIcon(-profileData.trends[profileData.trends.length - 1].tps_change)}
                    {Math.abs(profileData.trends[profileData.trends.length - 1].tps_change).toFixed(1)}%
                  </span>
                </div>
              </div>
            </>
          )}
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis yAxisId="left" orientation="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="response_time_ms" stroke="#8884d8" name="Response Time (ms)" />
            <Line yAxisId="right" type="monotone" dataKey="tokens_per_second" stroke="#82ca9d" name="Tokens/Second" />
            <Line yAxisId="left" type="monotone" dataKey="errors" stroke="#ff7c7c" name="Errors" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="performance-profiler">
      <div className="profiler-header">
        <h2>Performance Profiler</h2>
        <div className="time-selector">
          {['1h', '6h', '24h', '7d'].map(range => (
            <button
              key={range}
              className={timeRange === range ? 'active' : ''}
              onClick={() => setTimeRange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="error-message">
          <AlertTriangle size={20} />
          {error}
        </div>
      )}

      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          Loading performance data...
        </div>
      )}

      {profileData && !loading && (
        <>
          {renderTimingBreakdown()}
          {renderResourceTimeline()}
          {renderBottleneckAnalysis()}
          {renderPerformanceTrends()}
        </>
      )}
    </div>
  );
}

export default PerformanceProfiler;