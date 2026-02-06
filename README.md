# tieba_story

把“贴吧帖子（楼层/图片）”工程化生成 **约 2 分钟** 的图文叙事短片（静帧 + 旁白 + BGM + 字幕），并支持提示词版本化、风格参数化与可追溯的 trace 映射。

## 功能概览
- 贴吧帖子抓取/导入：按 URL 抓取楼层并落盘为工程目录
- LLM 总结（M2）：楼层 → 结构化摘要（可追溯）
- LLM 分镜（M3）：摘要 → 固定镜头数 storyboard（目标总时长可控）
- 分镜二次润色：旁白更有感染力但克制；生图提示词统一风格
- 资产渲染：每镜头 1 张图 + 1 段旁白音频（TTS）
- BGM 计划：根据故事氛围给出候选与混音建议（ducking/LUFS）
- Remotion 合成：图片 + 旁白 + BGM + 字幕 → mp4

## 目录结构（核心）
- `tools/`：流水线脚本（抓取/LLM/生图/TTS/同步/props）
- `prompts/packs/v1/prompt_pack.json`：提示词模板包（版本化、可参数化）
- `projects/{post_id}/`：单个帖子工程目录（默认不提交 Git）
- `remotion_video/`：Remotion 合成工程

## 环境要求
- Windows / macOS / Linux 均可（本仓库主要在 Windows 下验证）
- Python 3.10+
- Node.js 18+

安装依赖（Remotion）：
```bash
cd remotion_video
npm i
```

## 配置（非常重要：不要提交 API Key）
本项目默认读取本地配置文件 `docs/aliyun_api-key.md`（已在根目录 `.gitignore` 中忽略）。

1) 复制 example：
```bash
cp docs/aliyun_api-key.example.md docs/aliyun_api-key.md
```

2) 编辑 `docs/aliyun_api-key.md`，填入你自己的：
- `base_url`
- `api_key`

3) 自检该文件不会被提交：
```bash
git check-ignore -v docs/aliyun_api-key.md
```

## 快速开始：从帖子到成片（示例以 9461449190）

### 0) 抓取/导入楼层（M1）
抓取帖子并生成 `projects/{post_id}/source/`（具体参数以脚本帮助为准）：
```bash
python tools/tieba_crawl_by_url.py --help
```

### 1) LLM 总结（M2）
生成 `projects/{post_id}/llm/story.summary.json`：
```bash
python tools/tieba_m2_summarize.py --project-dir projects/9461449190 --model glm-4.7 --prompt-pack v1 --style-preset warm --persona 真挚倾诉 --target-min-seconds 110 --target-max-seconds 130 --timeout 240 --force
```

### 2) LLM 分镜（M3）
生成 `projects/{post_id}/llm/storyboard.json`：
```bash
python tools/tieba_m3_storyboard.py --project-dir projects/9461449190 --model glm-4.7 --prompt-pack v1 --style-preset warm --persona 真挚倾诉 --target-seconds 120 --scenes 10 --timeout 240 --force
```

### 3) 分镜二次润色（可选，但推荐）
对旁白与生图提示词做二次打磨（覆盖写回 storyboard.json）：
```bash
python tools/tieba_m3_polish_storyboard.py --in projects/9461449190/llm/storyboard.json --mode voiceover,prompts --prompt-pack v1 --style-preset warm --persona 真挚倾诉 --model glm-4.7 --timeout 240
```

### 4) BGM 计划（M5）与 QA（M7，可选）
```bash
python tools/tieba_m5_bgm_plan.py --project-dir projects/9461449190 --model glm-4.7 --prompt-pack v1 --style-preset warm --persona 真挚倾诉 --timeout 240 --force
python tools/tieba_m7_qa_validate.py --project-dir projects/9461449190 --model glm-4.7 --prompt-pack v1 --scenes 10 --target-seconds 120 --timeout 240
```

### 5) 生成图片与旁白（生图 + TTS）
默认生图模型已切换为 `qwen-image-max`（也可命令行覆盖）。
```bash
python tools/tieba_m3_render_assets.py --project-dir projects/9461449190 --max-scenes 10 --image-model qwen-image-max --consistency-mode edit_prev --timeout 240
```

仅重跑配音（复用已有图片/manifest）：
```bash
python tools/tieba_m3_render_assets.py --project-dir projects/9461449190 --max-scenes 10 --only-tts --timeout 240
```

### 6) 同步到 Remotion public + 生成 props（M6）
```bash
python tools/tieba_m6_sync_public.py --project-dir projects/9461449190 --remotion-dir remotion_video
python tools/tieba_m6_build_props.py --project-dir projects/9461449190 --public-prefix 9461449190 --bgm-rel 9461449190/bgm/bgm.mp3 --max-scenes 10 --out remotion_video/props.9461449190.v1.120s.json
```

### 7) 渲染 mp4（Remotion）
```bash
cd remotion_video
npx remotion render TiebaStory ../output/9461449190.v1.120s.mp4 --props=props.9461449190.v1.120s.json
```

## 提示词体系（v1）
提示词模板统一收敛到：
- `prompts/packs/v1/prompt_pack.json`

核心模板：
- `summarize_v3`：楼层 → story.summary.json
- `storyboard_v3`：summary → storyboard.json（固定 N 镜头，目标时长可配）
- `voiceover_polish_v1`：旁白二次润色（更有感染力但克制）
- `image_prompt_polish_v1`：生图提示词统一画风/禁忌
- `bgm_plan_v2`：BGM 方案与混音建议
- `qa_validate_v1`：产物自检报告

## 隐私与安全
- **不要提交任何 API Key / Token / Cookie**。
- 本仓库默认忽略：
  - `docs/aliyun_api-key.md`
  - `.env*`
  - `projects/`、`output/`（避免大文件与隐私内容进入 Git）
- 如需共享生成产物，建议使用独立的产物仓库或 Git LFS，并先做脱敏检查。

