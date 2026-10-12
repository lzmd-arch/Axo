/* Live2D 接口预留。
 *
 * 这个文件现在的全部作用是**把接口的形状定下来**：四个方法、一个容器、
 * 一套"没模型时静默降级"的语义。等真的接 Cubism Web SDK 时，只需要把
 * load() 里的注释换成 SDK 调用，其余代码（app.js、Python 侧、CSS）一行都不用动。
 *
 * 为什么值得先写：Live2D 是**唯一一个会反向影响对话区渲染的第三方组件**——
 * 它要求页面暴露口型同步与表情入口。若等到接入时才发现这些入口没留，就得回头
 * 改动对话流的每一处。现在留好，接入就只是往这个文件里填实现。
 *
 * 通过 window.BASE 解析模型路径（Python 侧在初始化时注入 web 根目录的 file:// URL），
 * 因为任意位置的模型路径都必须先拼成绝对路径才能被 fetch 到。
 */
(function (global) {
  "use strict";

  var host = null; // 现在的 DOM 容器
  var runtime = null; // 将来的 Cubism 运行时句柄
  var mouthOpen = 0;

  function resolve(path) {
    if (!path) return "";
    if (/^(https?:|file:|data:)/i.test(path)) return path;
    return (global.BASE || "") + "/" + String(path).replace(/^\/+/, "");
  }

  global.Live2DHost = {
    /** 模型是否已经加载。没加载时其余方法全是空操作。 */
    get loaded() {
      return runtime !== null;
    },

    /**
     * 加载一个 Live2D 模型。
     * @param {string} modelPath 相对 web 根目录的路径，或绝对 URL。
     * @returns {Promise<boolean>} 是否加载成功。**失败不抛异常**——一个装饰性的
     *   看板娘加载不出来，不该让整个对话窗口崩掉。
     */
    load: function (modelPath) {
      host = host || document.getElementById("live2d-host");
      if (!host) return Promise.resolve(false);
      if (!modelPath) {
        host.classList.remove("active");
        return Promise.resolve(false);
      }
      host.classList.add("active");
      // ── 接入点 ──────────────────────────────────────────────────────────
      // const { Application } = await import(resolve("vendor/live2d/cubism4.min.js"));
      // runtime = await Application.from({ url: resolve(modelPath) }).run(host);
      // ────────────────────────────────────────────────────────────────────
      console.info("[Live2D] 尚未接入渲染器，已记录待加载模型：", resolve(modelPath));
      return Promise.resolve(false);
    },

    /** 切换表情。名字由模型定义。 */
    setExpression: function (name) {
      if (!runtime || !name) return;
      // runtime.expression(name);
    },

    /**
     * 设置张嘴程度。**这是口型同步的接入口**：0 闭嘴，1 全开。
     * @param {number} value
     */
    setMouthOpen: function (value) {
      mouthOpen = Math.max(0, Math.min(1, Number(value) || 0));
      if (!runtime) return;
      // runtime.setParameterValueById("ParamMouthOpenY", mouthOpen);
    },

    /** 让模型看向某个方向（-1..1 归一化坐标），可接鼠标跟随。 */
    setLookAt: function (x, y) {
      if (!runtime) return;
      // runtime.focusController.focus(x, y);
    },
  };
})(window);
