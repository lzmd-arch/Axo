# 开屏形象资源

`ui/splash.py` 按这个顺序取形象：

1. `character.gif` / `character.webp`（动图，原速播放）
2. `character.png`（静态立绘，做呼吸与浮动）
3. 上一级目录的 `axo-icon.png`（兜底）

## 两个文件

- **`character.svg`**：手工矢量概念稿。六束羽状外鳃（三束/侧，鳃尖荧光青）、
  轴突尾（髓鞘节段 + 突触扣 + 琥珀火花）、胸口暖光核、虹膜突触纹理；白化粉皮肤、
  深炭灰简洁服装与荧光脉络。可编辑，比例与色值都按官方设定文档 v1.0。
- **`character.png`**：运行时使用的 667×1000 透明底立绘。由 ComfyUI
  （Counterfeit-V3.0，SD1.5 anime）以矢量稿的 512×768 品红底渲染图为输入做
  img2img（denoise 0.45，seed 202），再经：品红抠底与去溢色、肤色/鳃底粉调、
  胸口核改琥珀、鳃尖补荧光、外缘冷色微光。

换成立绘约稿的图时，只需替换 `character.png`（竖版、透明底最佳），版面与动画
不用改。若要动图，放 `character.webp` 或 `character.gif` 即可，静态图会被让位。
