# 阿里云百炼 / DashScope 配置（example）

# 使用说明：

# 1) 复制本文件为 `docs/aliyun_api-key.md`

# 2) 填入你自己的 `api_key`

# 3) 确保 `docs/aliyun_api-key.md` 不会被提交到 Git（建议加入根目录 `.gitignore`）

# 百炼控制台（API 页面）：

# - https://bailian.console.aliyun.com/cn-beijing/?spm=a2c4g.11186623.0.0.59d36f3avmN6WB&tab=api#/api

# 说明：

# - 该 API Key 通常可用于多个模型；因此这里不固定 `model` 字段。

# - 具体模型名称/版本在“调用时”由代码或命令行参数指定。

# - 你当前可用模型如下（调用时填入 `model`）：

# - 语言模型：

# - deepseek-v3.2

# - kimi-k2.5

# - kimi-k2-thinking

# - glm-4.7

# - 视觉语言模型 / 多模态理解：

# - qwen-vl-max-2025-08-13

# - qwen3-vl-plus-2025-12-19

# - qwen3-vl-flash-2026-01-22

# - 生图 / 图像编辑 / 视频相关模型：

# - qwen-image-plus-2026-01-09

# - qwen-image-max

# - qwen-image

# - qwen-image-edit

# - qwen-image-edit-plus-2025-12-15

# - z-image-turbo

# - qwen-mt-image

# - wan2.6-t2i

# - wan2.6-image

# - wan2.6-t2v

# - wan2.6-i2v-flash

# - wan2.6-i2v

# DashScope API Base URL（常见默认值）

base_url="https://dashscope.aliyuncs.com"

# 你的 API Key（替换）

api_key="YOUR_ALIYUN_BAILIAN_API_KEY"

# （可选）区域信息（你当前使用 cn-beijing 控制台）

region="cn-beijing"
