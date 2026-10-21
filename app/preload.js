const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('uchitel', {
  pick: k => ipcRenderer.invoke('pick', k), openExternal: u => (ipcRenderer.invoke('open', u), true), reveal: p => ipcRenderer.invoke('reveal', p),
  retry: () => ipcRenderer.invoke('retry'),
  onBoot: f => ipcRenderer.on('boot', (_, d) => f(d)), onBootLog: f => ipcRenderer.on('boot-log', (_, d) => f(d)), onBootError: f => ipcRenderer.on('boot-error', (_, d) => f(d)) });
