<script setup>
import { ref } from 'vue'
const activeIndex = ref('home')
</script>

<template>
  <el-container>
    <el-header>
      <el-menu :default-active="activeIndex" mode="horizontal" router>
        <el-menu-item index="/">首页</el-menu-item>
        <el-sub-menu index="/dl">
          <template #title>深度学习</template>
          <el-menu-item index="/digital">手写数字识别</el-menu-item>
          <el-menu-item index="/sentiment">文本情感识别</el-menu-item>
        </el-sub-menu>
      </el-menu>
    </el-header>
    <el-container>
      <el-container>
        <el-main>
          <router-view />
        </el-main>
        <el-footer class="page-footer">
          CodingSoul - https://github.com/caoaolong
        </el-footer>
      </el-container>
    </el-container>
  </el-container>
</template>

<style scoped>
.page-footer {
  position: absolute;
  bottom: 0;
  display: flex;
  width: 100%;
  max-height: 100px;
  justify-content: center;
  align-items: center;
  border-top: 1px solid #303030;
}
</style>
