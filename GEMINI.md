# 项目指令 (Project Mandates)

## 时间与时区 (Time & Timezone)
- **强制时区**：本项目的所有命理分析、报告生成、日志记录必须使用 **北京时间 (UTC+8)**。
- **日期校准**：如果系统时间与北京时间不一致，必须以北京时间为准进行干支换算。
- **代码规范**：在 Go 代码中使用 `time.Now().In(time.FixedZone("CST", 8*3600))` 或加载 `Asia/Shanghai` 时区。
