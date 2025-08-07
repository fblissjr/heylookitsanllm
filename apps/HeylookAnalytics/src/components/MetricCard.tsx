// src/components/MetricCard.tsx
// Reusable metric display card with trends

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Card, Text, useTheme } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

interface MetricCardProps {
  title: string;
  value: number;
  format: 'number' | 'ms' | 'percent' | 'bytes';
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    value: number;
  };
  icon?: string;
  color?: string;
}

export default function MetricCard({
  title,
  value,
  format,
  trend,
  icon,
  color,
}: MetricCardProps) {
  const theme = useTheme();
  
  const formatValue = () => {
    switch (format) {
      case 'number':
        return value >= 1000000
          ? `${(value / 1000000).toFixed(1)}M`
          : value >= 1000
          ? `${(value / 1000).toFixed(1)}K`
          : value.toFixed(0);
      case 'ms':
        return `${value.toFixed(0)}ms`;
      case 'percent':
        return `${value.toFixed(1)}%`;
      case 'bytes':
        return value >= 1073741824
          ? `${(value / 1073741824).toFixed(1)}GB`
          : value >= 1048576
          ? `${(value / 1048576).toFixed(1)}MB`
          : value >= 1024
          ? `${(value / 1024).toFixed(1)}KB`
          : `${value}B`;
      default:
        return value.toString();
    }
  };
  
  const getTrendColor = () => {
    if (!trend) return theme.colors.outline;
    
    switch (trend.direction) {
      case 'up':
        return format === 'percent' && title.includes('Error')
          ? theme.colors.error
          : theme.colors.primary;
      case 'down':
        return format === 'percent' && title.includes('Error')
          ? theme.colors.primary
          : theme.colors.error;
      default:
        return theme.colors.outline;
    }
  };
  
  const getTrendIcon = () => {
    if (!trend) return null;
    
    switch (trend.direction) {
      case 'up':
        return 'trending-up';
      case 'down':
        return 'trending-down';
      default:
        return 'trending-neutral';
    }
  };
  
  return (
    <Card style={styles.card}>
      <Card.Content style={styles.content}>
        <View style={styles.header}>
          {icon && (
            <Icon
              name={icon}
              size={20}
              color={color || theme.colors.primary}
              style={styles.icon}
            />
          )}
          <Text variant="labelMedium" style={styles.title}>
            {title}
          </Text>
        </View>
        
        <Text
          variant="headlineMedium"
          style={[styles.value, { color: color || theme.colors.onSurface }]}
        >
          {formatValue()}
        </Text>
        
        {trend && (
          <View style={styles.trend}>
            <Icon
              name={getTrendIcon()!}
              size={16}
              color={getTrendColor()}
            />
            <Text
              variant="labelSmall"
              style={[styles.trendText, { color: getTrendColor() }]}
            >
              {trend.value > 0 ? '+' : ''}{trend.value.toFixed(1)}%
            </Text>
          </View>
        )}
      </Card.Content>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    marginRight: 12,
    minWidth: 140,
  },
  content: {
    paddingVertical: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  icon: {
    marginRight: 4,
  },
  title: {
    opacity: 0.7,
  },
  value: {
    marginVertical: 4,
  },
  trend: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  trendText: {
    marginLeft: 4,
  },
});