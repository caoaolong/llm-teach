import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import router from './router'

import { env } from 'onnxruntime-web'
env.wasm.wasmPaths = '/onnxruntime/'

createApp(App).use(ElementPlus).use(router).mount('#app')
