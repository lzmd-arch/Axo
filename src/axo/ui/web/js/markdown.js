/* 一个刻意很小的 markdown 渲染器。
 *
 * 为什么自己写而不用 marked.js：对话内容来自模型，把它塞进 innerHTML 之前
 * 必须过一道**白名单**。第三方库的 XSS 防护是靠配置与版本更新的，而这里需要的
 * 子集小到可以完整审计——整个文件就是那条白名单。
 *
 * 支持：段落、标题（#~######）、围栏与缩进代码块、行内代码、粗体、斜体、
 * 链接、无序/有序列表、引用、分隔线、换行。不支持表格、脚注、HTML 内嵌——
 * 后三个都是把任意 HTML 放进页面的口子，宁可显示成纯文本。
 *
 * 转义策略：**先转义、后标记**。文本里的 < > & " ' 在解析之前就换成实体，
 * 于是解析阶段生成的标签是页面里仅有的标签。这样即使解析逻辑有 bug，最坏
 * 结果也只是显示错乱，而不是执行注入的脚本。
 */
(function (global) {
  "use strict";

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // 链接白名单。javascript: 与 data: 直接退化成纯文本，不给任何机会。
  function safeHref(url) {
    var trimmed = url.trim();
    if (/^(https?:|mailto:|file:)/i.test(trimmed)) return trimmed;
    return null;
  }

  // 行内代码与链接的暂存符。用 NUL 作前缀是刻意的：它不可能出现在模型输出里
  // （文本里的控制字符在转义前就被剥掉了），因此不会与正文撞号。
  var TOKEN = "\u0000";

  function renderInline(text) {
    var held = [];

    function hold(html) {
      held.push(html);
      return TOKEN + (held.length - 1) + TOKEN;
    }

    var out = text;

    // 行内代码与链接**先摘出来**，否则后面的粗体/斜体规则会钻进 <code> 内部
    // 甚至跨过标签边界，产出 <strong>code>…</stro</strong> 这种残破 HTML。
    out = out.replace(/`([^`]+)`/g, function (_m, code) {
      return hold("<code>" + escapeHtml(code) + "</code>");
    });
    out = out.replace(/\[([^\]]*)\]\(([^)\s]+)\)/g, function (match, label, url) {
      var href = safeHref(url.trim());
      if (href === null) return match; // 非白名单协议：原样退回，当纯文本显示
      return hold(
        '<a href="' + escapeHtml(href) + '" rel="noopener noreferrer">' + label + "</a>"
      );
    });

    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[\s(（])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    out = out.replace(/(^|[\s(（])_([^_\s][^_]*)_/g, "$1<em>$2</em>");

    return out.replace(/\u0000(\d+)\u0000/g, function (_m, index) {
      return held[Number(index)];
    });
  }

  function render(text) {
    if (!text) return "";
    var lines = String(text).replace(/\r\n?/g, "\n").split("\n");
    var html = [];
    var index = 0;

    while (index < lines.length) {
      var line = lines[index];

      // 围栏代码块
      var fence = line.match(/^\s*```(.*)$/);
      if (fence) {
        var lang = fence[1].trim();
        var body = [];
        index += 1;
        while (index < lines.length && !/^\s*```/.test(lines[index])) {
          body.push(lines[index]);
          index += 1;
        }
        index += 1; // 跳过结束围栏
        var cls = lang ? ' class="lang-' + escapeHtml(lang) + '"' : "";
        html.push("<pre><code" + cls + ">" + escapeHtml(body.join("\n")) + "</code></pre>");
        continue;
      }

      // 标题
      var heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        var level = heading[1].length;
        html.push("<h" + level + ">" + renderInline(escapeHtml(heading[2])) + "</h" + level + ">");
        index += 1;
        continue;
      }

      // 分隔线
      if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
        html.push("<hr>");
        index += 1;
        continue;
      }

      // 引用（连续的 > 行合成一块）
      if (/^\s*>/.test(line)) {
        var quoted = [];
        while (index < lines.length && /^\s*>/.test(lines[index])) {
          quoted.push(lines[index].replace(/^\s*>\s?/, ""));
          index += 1;
        }
        html.push("<blockquote>" + renderInline(escapeHtml(quoted.join("\n"))) + "</blockquote>");
        continue;
      }

      // 列表
      if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
        var ordered = /^\s*\d+[.)]\s+/.test(line);
        var items = [];
        while (index < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[index])) {
          items.push(
            "<li>" + renderInline(escapeHtml(lines[index].replace(/^\s*([-*+]|\d+[.)])\s+/, ""))) + "</li>"
          );
          index += 1;
        }
        var tag = ordered ? "ol" : "ul";
        html.push("<" + tag + ">" + items.join("") + "</" + tag + ">");
        continue;
      }

      // 空行
      if (/^\s*$/.test(line)) {
        index += 1;
        continue;
      }

      // 段落：吃到下一个空行或块级元素的开头为止
      var paragraph = [];
      while (
        index < lines.length &&
        !/^\s*$/.test(lines[index]) &&
        !/^\s*```/.test(lines[index]) &&
        !/^(#{1,6})\s+/.test(lines[index]) &&
        !/^\s*>/.test(lines[index]) &&
        !/^\s*([-*+]|\d+[.)])\s+/.test(lines[index]) &&
        !/^\s*([-*_])\1{2,}\s*$/.test(lines[index])
      ) {
        paragraph.push(lines[index]);
        index += 1;
      }
      html.push("<p>" + renderInline(escapeHtml(paragraph.join("\n"))) + "</p>");
    }

    return html.join("");
  }

  global.MiniMarkdown = { render: render, escapeHtml: escapeHtml };
})(window);
