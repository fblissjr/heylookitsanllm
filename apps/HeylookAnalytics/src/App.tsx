import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiProvider } from './providers/ApiProvider';
import { DuckDBProvider } from './providers/DuckDBProvider';
import DashboardScreen from './screens/DashboardScreen';
import ModelsScreen from './screens/ModelsScreen';
import QueryScreen from './screens/QueryScreen';
import TestsScreen from './screens/TestsScreen';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const Tab = createBottomTabNavigator();
const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ApiProvider>
        <DuckDBProvider>
          <PaperProvider>
            <NavigationContainer>
              <Tab.Navigator
                screenOptions={({ route }) => ({
                  tabBarIcon: ({ focused, color, size }) => {
                    let iconName;
                    if (route.name === 'Dashboard') {
                      iconName = focused ? 'view-dashboard' : 'view-dashboard-outline';
                    } else if (route.name === 'Models') {
                      iconName = focused ? 'brain' : 'brain';
                    } else if (route.name === 'Query') {
                      iconName = focused ? 'database-search' : 'database-search-outline';
                    } else if (route.name === 'Tests') {
                      iconName = focused ? 'test-tube' : 'test-tube-empty';
                    }
                    return <MaterialCommunityIcons name={iconName} size={size} color={color} />;
                  },
                  tabBarActiveTintColor: '#007AFF',
                  tabBarInactiveTintColor: 'gray',
                })}
              >
                <Tab.Screen name="Dashboard" component={DashboardScreen} />
                <Tab.Screen name="Models" component={ModelsScreen} />
                <Tab.Screen name="Query" component={QueryScreen} />
                <Tab.Screen name="Tests" component={TestsScreen} />
              </Tab.Navigator>
            </NavigationContainer>
          </PaperProvider>
        </DuckDBProvider>
      </ApiProvider>
    </QueryClientProvider>
  );
}