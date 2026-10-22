const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('uchitel', {
  pick: k => ipcRenderer.invoke('pick', k), openExternal: u => (ipcRenderer.invoke('open', u), true), reveal: p => ipcRenderer.invoke('reveal', p),
  retry: () => ipcRenderer.invoke('retry'),
  setSecret: (k, v) => ipcRenderer.invoke('secret-set', k, v), hasSecret: k => ipcRenderer.invoke('secret-has', k),
  ivanReleases: () => ipcRenderer.invoke('ivan-releases'), ivanFetch: tag => ipcRenderer.invoke('ivan-fetch', tag), onIvanProgress: f => ipcRenderer.on('ivan-progress', (_, d) => f(d)),
  onBoot: f => ipcRenderer.on('boot', (_, d) => f(d)), onBootLog: f => ipcRenderer.on('boot-log', (_, d) => f(d)), onBootError: f => ipcRenderer.on('boot-error', (_, d) => f(d)) });
