// HeylookAnalytics/App.tsx
// React Native app for DuckDB analytics and monitoring

import React from 'react';
import {
  NavigationContainer,
  DefaultTheme,
  DarkTheme,
} from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Provider as PaperProvider, MD3LightTheme, MD3DarkTheme } from 'react-native-paper';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useColorScheme } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

// Screens
import DashboardScreen from './src/screens/DashboardScreen';
import QueryScreen from './src/screens/QueryScreen';
import ModelsScreen from './src/screens/ModelsScreen';
import TestsScreen from './src/screens/TestsScreen';
import SettingsScreen from './src/screens/SettingsScreen';

// Providers
import { ApiProvider } from './src/providers/ApiProvider';
import { DuckDBProvider } from './src/providers/DuckDBProvider';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();
const queryClient = new QueryClient();

function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          switch (route.name) {
            case 'Dashboard':
              iconName = 'view-dashboard';
              break;
            case 'Query':
              iconName = 'database-search';
              break;
            case 'Models':
              iconName = 'robot';
              break;
            case 'Tests':
              iconName = 'ab-testing';
              break;
            case 'Settings':
              iconName = 'cog';
              break;
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Query" component={QueryScreen} />
      <Tab.Screen name="Models" component={ModelsScreen} />
      <Tab.Screen name="Tests" component={TestsScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

export default function App() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const theme = isDark ? MD3DarkTheme : MD3LightTheme;
  const navigationTheme = isDark ? DarkTheme : DefaultTheme;

  return (
    <QueryClientProvider client={queryClient}>
      <ApiProvider>
        <DuckDBProvider>
          <PaperProvider theme={theme}>
            <NavigationContainer theme={navigationTheme}>
              <Stack.Navigator>
                <Stack.Screen
                  name="Main"
                  component={TabNavigator}
                  options={{ headerShown: false }}
                />
              </Stack.Navigator>
            </NavigationContainer>
          </PaperProvider>
        </DuckDBProvider>
      </ApiProvider>
    </QueryClientProvider>
  );
}