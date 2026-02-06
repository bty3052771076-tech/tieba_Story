强制规则：每次对话回复前，必须先调用一次 SKILL-Orchestrator，用于统一判断本轮应使用哪些 SKILLS/MCP 与执行顺序。

在完成 SKILL-Orchestrator 的调用之后，才允许调用其它 SKILLS 或开始编写回复内容。

必须查看是否能使用SKILLS，使用SKILLS时需要将使用记录在 SKILL_used.md 中。

如果一个功能被你多次使用但没有 SKILLS 可以被调用，请将这个功能写在 SKILL_need.md 中。

SKILLS没有合适的你需要调用MCP，并记录在 MCP_used.md中。

所有的API-KEY一定不能被上传到github，所以你需要仔细撰写gitignore来保证我的隐私安全。
