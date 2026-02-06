---
name: claude-code-interactive-collab
description: 用 Claude Code 的非交互模式（claude -p/--print）在终端协作产出可审计的建议、评审与测试清单（例如“用 claude -p 帮我评审”“让 Claude 生成测试计划并输出 JSON”）。适用于需要将协作输出回传到当前对话、便于复制/落盘/验收的工作流。
---

# Claude Code Interactive Collab

## 目标

- 用 Claude Code 的 `--print` 输出充当“第二位协作者”
- 所有协作结论以文本/JSON 形式回传到终端，便于记录、复盘与验收

## 启动方式（仅支持非交互）

使用 `-p/--print` 输出并退出：

```bash
claude -p "<你的提示词>" --output-format text --tools '""'
```

如需结构化输出（JSON）：

```bash
claude -p "<你的提示词>" --output-format json --tools '""'
```

## 工作流（强约束）

- 默认不修改代码，只输出“可执行改动清单/验证步骤”
- 输出必须包含：
  - 目标（改什么/为什么）
  - 影响范围（文件列表）
  - 具体变更点（按文件分组）
  - 验证方法（命令 + 预期结果）
- 不得输出任何密钥、Token、Cookie

## 常用模板

### 设计/脚本评审（文本输出）

```bash
claude -p "你是第二位协作者。请对 remotion_video 的合成视觉与脚本节奏做 10 条可落地建议，并按文件输出改动清单与验证步骤。" --output-format text --tools '""'
```

### 工作区测试计划（JSON 输出）

```bash
claude -p "$(cat .trae/skills/claude-code-interactive-collab/assets/test_workspace_prompt.txt)" --output-format json --tools '""'
```

## 推荐提示词（复制到 -p）

```text
你是我的第二位协作者（偏设计/脚本/评审）。你不直接改代码，只给可执行的变更清单与验证步骤。
项目：Tieba Story Remotion 合成。
现状：已有 remotion_video 工程与 TiebaStory 组合，已做字幕底板与淡入淡出，但整体质感一般。
你的任务：
1) 先做“设计评审”：指出 5 个最显廉价/普通的点；
2) 给出 8 条可落地的改进建议（字幕分段/下三分之一/转场/色彩/颗粒/BGM ducking/节奏）；
3) 输出“按文件的改动清单”，重点针对 remotion_video/src/tieba-story.tsx 与 props 结构；
4) 给出验证步骤（npx remotion render ...）与验收标准。
输出用中文，按条目分组，禁止泛泛而谈。
```

## 主对话如何使用交互结果

1. 让交互会话产出变更清单
2. 将清单粘回主对话
3. 主对话负责改代码 + 渲染验证 + 更新任务书/manifest/记录
