# NoGo Mac 棋盘平台

基于 Champion ONNX 引擎和 SAU 文本协议的本地浏览器棋盘。支持人机、双人、AI 自战、合法落点提示、手数、双方独立计时、落子确认、暂停、结束后自动保存 TXT 棋谱和 JSON 棋谱导出。

macOS 13+，Intel / Apple Silicon，Python 3.11–3.13。首次安装需联网，之后可离线运行。

每方共 15 分钟（12 分钟思考＋3 分钟预留），用尽总时间判负；AI 默认单手上限 60 秒的长搜索。人类点击棋盘先选点，点击“确认落子”后才提交，确认前继续计时。

正常对局结束后，棋谱自动写入平台的 `棋谱/` 文件夹，逐手记录黑白方和落子坐标。双方名称默认 `A1`、`A2`，文件名例如 `A1vsA2，A1胜.txt` 或 `A1vsA2，A2胜.txt`，同名追加序号。

详见 [Mac使用教程.md](Mac使用教程.md)，旧版升级见 [更新说明.txt](更新说明.txt)。

```bash
chmod +x *.command NoGoEngine
bash ./安装依赖.command
bash ./启动棋盘.command
```

本项目提供本地训练平台；不包含 Windows SAU GUI 源码，也不代表正式赛事平台认证。Linux 自动化与浏览器测试已运行，macOS 原生实机验证待在目标 Mac 完成。
