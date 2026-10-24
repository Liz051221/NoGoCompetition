# NoGo Mac 棋盘平台

基于 Champion ONNX 引擎和 SAU 文本协议的本地浏览器棋盘。支持人机、双人、AI 自战、合法落点提示、手数、暂停和 JSON 棋谱导出。

macOS 13+，Intel / Apple Silicon，Python 3.11–3.13。首次安装需联网，之后可离线运行。

详见 [Mac使用教程.md](Mac使用教程.md)。

```bash
chmod +x *.command NoGoEngine
bash ./安装依赖.command
bash ./启动棋盘.command
```

本项目提供本地训练平台；不包含 Windows SAU GUI 源码，也不代表正式赛事平台认证。Linux 自动化与浏览器测试已运行，macOS 原生实机验证待在目标 Mac 完成。
