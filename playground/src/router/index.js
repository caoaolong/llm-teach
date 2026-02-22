import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('@/views/home/index.vue')
    },
    {
      path: '/dl',
      name: 'dl',
      children: [
        {
          path: '/digital',
          name: 'digital',
          component: () => import('@/views/digital/index.vue')
        },
        {
          path: '/sentiment',
          name: 'sentiment',
          component: () => import('@/views/sentiment/index.vue')
        }
      ]
    }
  ]
})

export default router