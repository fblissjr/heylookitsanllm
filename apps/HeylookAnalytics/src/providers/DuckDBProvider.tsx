// src/providers/DuckDBProvider.tsx
// Local DuckDB instance for offline analytics

import React, { createContext, useContext, useEffect, useState } from 'react';
import SQLite from 'react-native-sqlite-storage';

interface DuckDBContextType {
  db: SQLite.SQLiteDatabase | null;
  execute: (sql: string, params?: any[]) => Promise<void>;
  query: (sql: string, params?: any[]) => Promise<any[]>;
  importFromServer: (table: string) => Promise<void>;
  exportToServer: (table: string) => Promise<void>;
  syncWithServer: () => Promise<void>;
}

const DuckDBContext = createContext<DuckDBContextType | undefined>(undefined);

// Enable debugging
SQLite.DEBUG(true);
SQLite.enablePromise(true);

export function DuckDBProvider({ children }: { children: React.ReactNode }) {
  const [db, setDb] = useState<SQLite.SQLiteDatabase | null>(null);
  
  useEffect(() => {
    initDatabase();
  }, []);
  
  const initDatabase = async () => {
    try {
      const database = await SQLite.openDatabase({
        name: 'heylook_analytics.db',
        location: 'default',
      });
      
      setDb(database);
      
      // Create local analytics tables
      await database.executeSql(`
        CREATE TABLE IF NOT EXISTS local_metrics (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          metric_type TEXT,
          model TEXT,
          value REAL,
          metadata TEXT
        )
      `);
      
      await database.executeSql(`
        CREATE TABLE IF NOT EXISTS saved_queries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          query TEXT NOT NULL,
          description TEXT,
          tags TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          last_used DATETIME
        )
      `);
      
      await database.executeSql(`
        CREATE TABLE IF NOT EXISTS query_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          query TEXT NOT NULL,
          executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          execution_time_ms INTEGER,
          row_count INTEGER,
          error TEXT
        )
      `);
      
      await database.executeSql(`
        CREATE TABLE IF NOT EXISTS offline_cache (
          table_name TEXT PRIMARY KEY,
          last_sync DATETIME,
          row_count INTEGER,
          schema TEXT
        )
      `);
      
    } catch (error) {
      console.error('Failed to initialize database:', error);
    }
  };
  
  const execute = async (sql: string, params?: any[]): Promise<void> => {
    if (!db) throw new Error('Database not initialized');
    await db.executeSql(sql, params);
  };
  
  const query = async (sql: string, params?: any[]): Promise<any[]> => {
    if (!db) throw new Error('Database not initialized');
    
    const startTime = Date.now();
    try {
      const [result] = await db.executeSql(sql, params);
      const executionTime = Date.now() - startTime;
      
      // Log to history
      await db.executeSql(
        'INSERT INTO query_history (query, execution_time_ms, row_count) VALUES (?, ?, ?)',
        [sql, executionTime, result.rows.length]
      );
      
      // Convert to array
      const rows = [];
      for (let i = 0; i < result.rows.length; i++) {
        rows.push(result.rows.item(i));
      }
      
      return rows;
    } catch (error: any) {
      // Log error
      await db.executeSql(
        'INSERT INTO query_history (query, execution_time_ms, error) VALUES (?, ?, ?)',
        [sql, Date.now() - startTime, error.message]
      );
      throw error;
    }
  };
  
  const importFromServer = async (table: string): Promise<void> => {
    // Import data from server DuckDB to local SQLite
    // This would use the API to fetch data and store locally
    console.log(`Importing ${table} from server...`);
  };
  
  const exportToServer = async (table: string): Promise<void> => {
    // Export local data to server DuckDB
    console.log(`Exporting ${table} to server...`);
  };
  
  const syncWithServer = async (): Promise<void> => {
    // Sync all cached tables with server
    if (!db) return;
    
    const tables = await query('SELECT table_name FROM offline_cache');
    for (const table of tables) {
      await importFromServer(table.table_name);
    }
  };
  
  const value: DuckDBContextType = {
    db,
    execute,
    query,
    importFromServer,
    exportToServer,
    syncWithServer,
  };
  
  return (
    <DuckDBContext.Provider value={value}>
      {children}
    </DuckDBContext.Provider>
  );
}

export function useDuckDB() {
  const context = useContext(DuckDBContext);
  if (!context) {
    throw new Error('useDuckDB must be used within DuckDBProvider');
  }
  return context;
}