// src/providers/ApiProvider.tsx
// API client for heylookitsanllm

import React, { createContext, useContext, useState, useEffect } from 'react';
import axios, { AxiosInstance } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface ApiContextType {
  client: AxiosInstance;
  baseURL: string;
  setBaseURL: (url: string) => void;
  isConnected: boolean;
  models: Model[];
  query: (sql: string, limit?: number) => Promise<QueryResult>;
  execute: (sql: string, params?: any[]) => Promise<void>;
  loadDataset: (source: DataSource) => Promise<void>;
  runPrompt: (params: PromptParams) => Promise<ChatCompletion>;
}

interface Model {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  capabilities?: {
    vision: boolean;
    function_calling: boolean;
  };
}

interface QueryResult {
  columns: string[];
  data: any[][];
  row_count: number;
}

interface DataSource {
  type: 'huggingface' | 'parquet' | 'jsonl' | 'csv';
  path: string;
  table_name: string;
  options?: Record<string, any>;
}

interface PromptParams {
  model: string;
  messages: Message[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string | MessageContent[];
}

interface MessageContent {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: {
    url: string;
  };
}

interface ChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export function ApiProvider({ children }: { children: React.ReactNode }) {
  const [baseURL, setBaseURLState] = useState('http://localhost:8080');
  const [isConnected, setIsConnected] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  
  // Create axios instance
  const client = axios.create({
    baseURL,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
  // Load saved URL
  useEffect(() => {
    AsyncStorage.getItem('api_base_url').then((url) => {
      if (url) setBaseURLState(url);
    });
  }, []);
  
  // Update base URL
  const setBaseURL = async (url: string) => {
    setBaseURLState(url);
    await AsyncStorage.setItem('api_base_url', url);
    client.defaults.baseURL = url;
    checkConnection();
  };
  
  // Check connection
  const checkConnection = async () => {
    try {
      await client.get('/');
      setIsConnected(true);
      fetchModels();
    } catch (error) {
      setIsConnected(false);
    }
  };
  
  // Fetch available models
  const fetchModels = async () => {
    try {
      const response = await client.get('/v1/models');
      setModels(response.data.data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };
  
  // Initial connection check
  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, [baseURL]);
  
  // Query DuckDB data
  const query = async (sql: string, limit = 1000): Promise<QueryResult> => {
    const response = await client.post('/v1/data/query', {
      query: sql,
      limit,
    });
    return response.data;
  };
  
  // Execute DuckDB statement
  const execute = async (sql: string, params?: any[]): Promise<void> => {
    await client.post('/v1/data/execute', {
      statement: sql,
      params,
    });
  };
  
  // Load dataset
  const loadDataset = async (source: DataSource): Promise<void> => {
    await client.post('/v1/data/load', {
      source_type: source.type,
      source_path: source.path,
      table_name: source.table_name,
      options: source.options,
    });
  };
  
  // Run prompt
  const runPrompt = async (params: PromptParams): Promise<ChatCompletion> => {
    const response = await client.post('/v1/chat/completions', params);
    return response.data;
  };
  
  const value: ApiContextType = {
    client,
    baseURL,
    setBaseURL,
    isConnected,
    models,
    query,
    execute,
    loadDataset,
    runPrompt,
  };
  
  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

export function useApi() {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within ApiProvider');
  }
  return context;
}