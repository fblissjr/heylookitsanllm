import { openDB, type IDBPDatabase } from 'idb'

import type { Conversation } from '../types/chat'

// --- Schema ---

const DB_NAME = 'heylook'
const DB_VERSION = 1

const CONVERSATIONS_STORE = 'conversations'
const NOTEBOOK_STORE = 'notebooks'

export interface PersistedNotebookDocument {
  id: string
  title: string
  content: string
  systemPrompt: string
  images: { id: string; name: string }[]
  createdAt: number
  modifiedAt: number
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type HeylookDB = IDBPDatabase<any>

let dbPromise: Promise<HeylookDB> | null = null

function getDB(): Promise<HeylookDB> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(CONVERSATIONS_STORE)) {
          db.createObjectStore(CONVERSATIONS_STORE, { keyPath: 'id' })
        }
        if (!db.objectStoreNames.contains(NOTEBOOK_STORE)) {
          db.createObjectStore(NOTEBOOK_STORE, { keyPath: 'id' })
        }
      },
    })
  }
  return dbPromise
}

// --- Conversations ---

export async function getAllConversations(): Promise<Conversation[]> {
  const db = await getDB()
  const all: Conversation[] = await db.getAll(CONVERSATIONS_STORE)
  // Return sorted by updatedAt descending (most recent first)
  return all.sort((a, b) => b.updatedAt - a.updatedAt)
}

export async function saveConversation(conversation: Conversation): Promise<void> {
  const db = await getDB()
  await db.put(CONVERSATIONS_STORE, conversation)
}

export async function deleteConversation(id: string): Promise<void> {
  const db = await getDB()
  await db.delete(CONVERSATIONS_STORE, id)
}

export async function exportConversations(): Promise<string> {
  const conversations = await getAllConversations()
  return JSON.stringify(conversations, null, 2)
}

export async function importConversations(json: string): Promise<number> {
  const conversations: Conversation[] = JSON.parse(json)
  if (!Array.isArray(conversations)) {
    throw new Error('Invalid import data: expected an array of conversations')
  }
  const db = await getDB()
  const tx = db.transaction(CONVERSATIONS_STORE, 'readwrite')
  for (const conv of conversations) {
    await tx.store.put(conv)
  }
  await tx.done
  return conversations.length
}

// --- Notebook Documents ---

export async function getAllNotebookDocuments(): Promise<PersistedNotebookDocument[]> {
  const db = await getDB()
  const all: PersistedNotebookDocument[] = await db.getAll(NOTEBOOK_STORE)
  return all.sort((a, b) => b.modifiedAt - a.modifiedAt)
}

export async function saveNotebookDocument(doc: PersistedNotebookDocument): Promise<void> {
  const db = await getDB()
  await db.put(NOTEBOOK_STORE, doc)
}

export async function deleteNotebookDocument(id: string): Promise<void> {
  const db = await getDB()
  await db.delete(NOTEBOOK_STORE, id)
}

export async function saveAllNotebookDocuments(docs: PersistedNotebookDocument[]): Promise<void> {
  const db = await getDB()
  const tx = db.transaction(NOTEBOOK_STORE, 'readwrite')
  for (const doc of docs) {
    await tx.store.put(doc)
  }
  await tx.done
}
