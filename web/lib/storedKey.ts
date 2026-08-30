"use client";

import { useCallback, useSyncExternalStore } from "react";

const STORAGE = "research-copilot-key";

const listeners = new Set<() => void>();

function notify() {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function read(): string {
  try {
    return window.localStorage.getItem(STORAGE) ?? "";
  } catch {
    // A private window, or a browser set to block site data, throws on access.
    return "";
  }
}

/** Nothing is stored during prerender, so the server view of the key is empty. */
const serverSnapshot = () => "";

/**
 * The visitor's API key, read from and written to this browser.
 *
 * Held in browser storage rather than in component state so a reload does not
 * ask for it again, and read through useSyncExternalStore so the prerendered
 * markup and the first client render agree.
 */
export function useStoredKey(): [string, (value: string) => void] {
  const key = useSyncExternalStore(subscribe, read, serverSnapshot);

  const set = useCallback((value: string) => {
    try {
      window.localStorage.setItem(STORAGE, value);
    } catch {
      // Not remembering it is a worse experience, not a broken one.
    }
    notify();
  }, []);

  return [key, set];
}
