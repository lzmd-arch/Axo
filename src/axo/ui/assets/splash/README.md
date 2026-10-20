# 开屏形象资源

`ui/splash.py` 按这个顺序取形象：

1. `character.gif` / `character.webp`（动图，原速播放）
2. `character.png`（静态立绘，做呼吸与浮动）
3. 上一级目录的 `axo-icon.png`（兜底）

## 两个文件

- **`character.svg`**：手工矢量概念稿。六束羽状外鳃（三束/侧，鳃尖荧光青；第 5 版
  整体缩到 84%）、轴突尾（髓鞘节段 + 突触扣 + 琥珀火花）、胸口暖光核、虹膜突触纹理；
  蓝粉渐变发色、白化粉皮肤、深炭灰简洁服装与荧光脉络。可编辑。
- **`character.png`**：运行时使用的 832×1216 透明底半身立绘。由 ComfyUI +
  Illustrious-XL v2.0（SDXL anime 底模）文生图（短提示词，外接 SDXL VAE），
  再用 BiRefNet 抠底。第 5 版按要求重绘：蓝粉渐变短发、六束外鳃收小；角色名
  千岁（昵称阿螈）。

换成立绘约稿的图时，只需替换 `character.png`（竖版、透明底最佳），版面与动画
不用改。若要动图，放 `character.webp` 或 `character.gif` 即可，静态图会被让位。
