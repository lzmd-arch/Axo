/* 对话页。
 *
 * 与 Python 的契约见 chat_view.py 的模块文档：网页提供 window.ChatHost 那一组方法，
 * Python 通过 runJavaScript 调用它们；网页通过 window.bridge.send(text) 把用户输入
 * 送回去、通过 window.bridge.answer_persona(text) 把人设卡片的答案送回去
 * （空串 = 这一次先不写）、通过 window.bridge.voice_toggle() 说"麦克风按钮被按了"。
 * **方法个数不在这里数**（数过两次，两次都数错了）——要看清单去 chat_view.py 那张
 * 契约表，它是唯一一处两边都认的账。两边都**只增不改**：改签名会让旧的一方静默失联。
 *
 * 两条性能纪律：
 *
 * 1. **增量渲染走 requestAnimationFrame 合帧**。模型每秒能吐几十片，每片都
 *    重排一次 DOM 会把界面拖到卡顿。合帧后每屏最多重渲染一次。
 * 2. **自动滚动只在用户本来就在底部时生效**。否则用户往上翻看历史时，流式的
 *    新内容会不停把他拽回底部——这是聊天界面最让人恼火的毛病之一。
 */
(function () {
  "use strict";

  var stream = document.getElementById("stream");
  var input = document.getElementById("input");
  var sendButton = document.getElementById("send");
  var micButton = document.getElementById("mic");
  var callButton = document.getElementById("call");
  var callBar = document.getElementById("call-bar");
  var callStateLabel = document.getElementById("call-state");
  var callPartial = document.getElementById("call-partial");
  var callLevel = document.getElementById("call-level");
  var callInterrupt = document.getElementById("call-interrupt");
  var callMute = document.getElementById("call-mute");
  var callHangup = document.getElementById("call-hangup");
  var statusLabel = document.getElementById("status");
  var emptyState = document.getElementById("empty");
  var welcome = document.getElementById("welcome");
  var welcomeBuiltin = document.getElementById("welcome-builtin");
  var personaInput = document.getElementById("persona-input");
  var personaSave = document.getElementById("persona-save");
  var personaSkip = document.getElementById("persona-skip");

  var bridge = null;
  var busy = false;
  //: 录音那一路的状态："" / "listening" / "recognizing"。**它由终端推过来**
  //: （`setVoiceState`），因为录音与识别都在 Python 侧——网页只知道"按了哪一下"。
  var voiceState = "";
  //: 通话状态："" = 没在通话；其余由终端推（setCallState）。**麦克风的开关不归
  //: 网页判断**——半双工由 Python 的状态机决定，这里只显示。
  var callState = "";
  var callMuted = false;
  var active = null; // 正在流式写入的消息体
  var pending = null; // 待渲染的增量文本缓冲

  /* 人设卡片正等着回答。**初值 false，由 Python 推上来的那次打开它。**

     方向是这么定的：反过来的写法（初值 true，"没人说一句之前不让打字"）在正常
     路径上只差几百毫秒——Python 一定会在页面加载完的那一刻推一次（见 chat_view.py
     的 set_persona_state）。可接线一旦断了，代价是**输入框永远点不动、而这句原因
     只出现在控制台里**，正是"设置按钮点不动"那个 bug 的形状。这一版宁可退化成
     "首次那张卡片没弹出来"，也不要再制造一个看不出原因的死按钮。 */
  var personaPending = false;

  /* 与 Python 的通道已经确认建立不起来了。这是个**终态**：之后不管谁推什么，
     都不该把输入框重新解开——那会让用户在"通道没建立"的页面上打字，打完了没人接。 */
  var channelDead = false;

  // ── 渲染 ──────────────────────────────────────────────────────────────────

  function nearBottom() {
    return stream.scrollHeight - stream.scrollTop - stream.clientHeight < 120;
  }

  function scrollToBottom() {
    stream.scrollTop = stream.scrollHeight;
  }

  function makeBubble(role, text, meta) {
    var wrap = document.createElement("div");
    wrap.className = "msg msg-" + role;

    if (meta && meta.title) {
      var caption = document.createElement("div");
      caption.className = "msg-caption";
      caption.textContent = meta.title;
      wrap.appendChild(caption);
    }

    var body = document.createElement("div");
    body.className = "msg-body";
    body.innerHTML = window.MiniMarkdown.render(text || "");
    wrap.appendChild(body);

    /* 她发的一张图（表情包）。**只认 meta.image 这个键**，不给正文里的 Markdown
       开图片语法：图的来源是本地目录（Python 那边扫出来的），而不是模型写的一串
       地址——从文本里认图片等于让模型能指定「去加载哪个 URL」，而那是一次可以对
       外发起的请求。 */
    if (meta && meta.image) {
      var picture = document.createElement("img");
      picture.className = "msg-image";
      picture.src = meta.image;
      picture.alt = "";
      wrap.appendChild(picture);
    }

    if (meta && meta.footer) {
      var footer = document.createElement("div");
      footer.className = "msg-footer";
      footer.textContent = meta.footer;
      wrap.appendChild(footer);
    }

    wrap._body = body;
    wrap._text = text || "";
    return wrap;
  }

  /* 输入区三个控件（输入框 / 麦克风 / 发送）的启用状态与措辞**只有这一个出口**。
     此前 setBusy 自己写 disabled、通道失败时又各写各的，三处各有一套说法；加上人设
     卡片之后"能不能打字"由两个条件共同决定（正忙 / 正等人设），再分开写就一定会漏掉
     一种组合。

     **麦克风也归这里**，理由多一层：生成中按它会让识别结果发不出去（`_on_submit` 那
     一支直接 return），而那句话会人间蒸发——所以它跟着发送键一起禁。 */
  function updateComposer() {
    if (channelDead) {
      input.disabled = true;
      sendButton.disabled = true;
      micButton.disabled = true;
      callButton.disabled = true;
      return;
    }
    var inCall = callState !== "" && callState !== "off";
    sendButton.disabled = busy || personaPending;
    input.disabled = personaPending;
    // 麦克风跟着发送键一起禁：生成中她说不了话，而"识别完一句却发不出去"（`_on_submit`
    // 那一支直接 return）在界面上是一句人间蒸发的话。**不静默丢一次点击。**
    micButton.disabled = busy || personaPending || inCall;
    callButton.disabled = personaPending;
    callButton.textContent = inCall ? "挂断" : "通话";
    callButton.title = inCall ? "结束通话" : "开始通话";
    input.placeholder = personaPending
      ? "先回答上面的问题，或者选「这一次先不写」。"
      : voiceState === "listening"
        ? "正在听…（再按一下结束）"
        : busy
          ? "正在生成…"
          : "说点什么…（Enter 发送，Shift+Enter 换行）";
    micButton.classList.toggle("listening", voiceState === "listening");
    micButton.classList.toggle("recognizing", voiceState === "recognizing");
    micButton.title =
      voiceState === "listening"
        ? "正在听…（再按一下结束）"
        : voiceState === "recognizing"
          ? "正在识别…"
          : "说一句（按一下开始，再按一下结束）";
    micButton.textContent = voiceState === "recognizing" ? "识别中…" : "说话";
    callBar.hidden = !inCall;
    document.body.classList.toggle("call-active", inCall);
    callMute.textContent = callMuted ? "取消静音" : "静音";
  }

  /* 空状态显隐。#empty 自己住在 #stream 里，所以判据就是"stream 里有没有 .msg"
     ——再加上"卡片占着这块地方吗"（两者是同一件事的两种形态，不能同时出现）。
     **每个会改变这两件事的地方都要调它**（append / resetStream / loadHistory /
     setPersonaState），漏一处就会留下"一条消息都没有、却什么都不显示"的界面。 */
  function updateEmpty() {
    if (emptyState) emptyState.hidden = stream.querySelector(".msg") !== null || personaPending;
  }

  /* 清空对话流。**不用 `stream.innerHTML = ""`**——那会把常驻的 #empty 节点一起
     抹掉，于是清空之后连"说点什么就开始"都没有了，页面看起来像是坏了。 */
  function resetStream() {
    var messages = stream.querySelectorAll(".msg");
    for (var i = 0; i < messages.length; i++) messages[i].remove();
    active = null;
    pending = null;
    updateEmpty();
  }

  function append(role, text, meta) {
    var stick = nearBottom();
    var bubble = makeBubble(role, text, meta);
    stream.appendChild(bubble);
    updateEmpty();
    if (stick) scrollToBottom();
    return bubble;
  }

  // ── 确认卡片 ──────────────────────────────────────────────────────────────
  //
  // 一次改状态的动作要用户点头。卡片是**对话流里的一个 .msg 节点**，不是弹窗：
  // 用户是看着助手刚说的那句话决定要不要点头的，模态框会把那句话挡住。
  //
  // 卡片记得自己的 callId，因为裁决回传时要带上它——闸门靠它比对"这一下算不算
  // 现在这次问询"，而比对不上就会被丢弃（见 agent/gate.py）。

  var cards = []; // 活着的卡片：[{node, at, resolved}]

  function findCard(callId) {
    for (var i = 0; i < cards.length; i++) {
      if (cards[i].node.getAttribute("data-call-id") === callId) return cards[i];
    }
    return null;
  }

  function makeToolCard(callId, name, summary, meta) {
    var wrap = document.createElement("div");
    wrap.className = "msg msg-tool-confirm";
    // call_id 进 DOM 属性：用户点了「允许」之后，回传的那个 id 必须与这里显示的是
    // 同一个。肉眼核对不了 32 个字符，所以排查时只能靠这个属性。
    wrap.setAttribute("data-call-id", callId || "");

    var caption = document.createElement("div");
    caption.className = "msg-caption";
    caption.textContent = "需要你确认 · " + (name || "工具");
    wrap.appendChild(caption);

    var body = document.createElement("div");
    body.className = "msg-body";

    var line = document.createElement("div");
    line.className = "tool-summary";
    // textContent 而不是 innerHTML：摘要是从**模型给的参数**拼出来的，里面出现
    // 尖括号完全正常。用 innerHTML 就等于让模型的参数决定这段 DOM 的结构。
    line.textContent = summary || "（没有参数）";
    body.appendChild(line);

    var actions = document.createElement("div");
    actions.className = "tool-actions";
    var allow = document.createElement("button");
    allow.type = "button";
    allow.className = "tool-allow";
    allow.textContent = "允许";
    var deny = document.createElement("button");
    deny.type = "button";
    deny.className = "tool-deny";
    deny.textContent = "拒绝";
    allow.addEventListener("click", function () {
      answerToolConfirm(callId, true);
    });
    deny.addEventListener("click", function () {
      answerToolConfirm(callId, false);
    });
    actions.appendChild(allow);
    actions.appendChild(deny);
    body.appendChild(actions);

    var countdown = document.createElement("div");
    countdown.className = "tool-countdown";
    body.appendChild(countdown);

    wrap.appendChild(body);
    wrap._allow = allow;
    wrap._deny = deny;
    wrap._countdown = countdown;
    return wrap;
  }

  function appendToolCard(callId, name, summary, meta) {
    var stick = nearBottom();
    var wasEmpty = stream.querySelector(".msg") === null;
    var node = makeToolCard(callId, name, summary, meta);
    stream.appendChild(node);
    var timeoutMs = Number(meta && meta.timeoutMs);
    cards.push({
      node: node,
      at: Date.now() + (isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 0),
      resolved: false,
    });
    if (wasEmpty) updateEmpty();
    if (stick) scrollToBottom();
    return node;
  }

  /* 点击的**唯一出口**。按钮在这里就禁掉，不等 Python 回话：一来一回要走四层，
     而这段时间里用户完全可能再点一下。Python 侧同样拒绝第二次（第一次点击定案），
     这里是让界面先说清楚"已经收到了"。 */
  function answerToolConfirm(callId, granted) {
    var entry = findCard(callId);
    if (entry) {
      entry.node._allow.disabled = true;
      entry.node._deny.disabled = true;
    }
    if (channelDead || !bridge) return;
    bridge.tool_confirm(callId, granted);
  }

  /* 倒计时。**一个全局定时器扫所有活着的卡片**，不是每张卡一个 setInterval：
     卡片会被 resetStream/loadHistory 连同 .msg 一起从 DOM 里删掉，而 per-card
     的定时器不知道这件事，于是它永远跑下去（切几次会话就攒一堆）。这里靠
     isConnected 自清，删掉的卡片下一次扫描就出列。 */
  function tickCards() {
    var now = Date.now();
    for (var i = cards.length - 1; i >= 0; i--) {
      var entry = cards[i];
      if (!entry.node.isConnected) {
        cards.splice(i, 1);
        continue;
      }
      if (entry.resolved || !entry.at) continue;
      var left = Math.max(0, Math.ceil((entry.at - now) / 1000));
      entry.node._countdown.textContent =
        left > 0 ? "还有 " + left + " 秒自动拒绝" : "已超时：视为拒绝";
      if (left <= 0) {
        entry.node._allow.disabled = true;
        entry.node._deny.disabled = true;
      }
    }
  }
  window.setInterval(tickCards, 1000);

  // ── 合帧 ──────────────────────────────────────────────────────────────────

  var frame = null;

  function flush() {
    frame = null;
    if (active && pending !== null) {
      var stick = nearBottom();
      active._text = pending;
      active._body.innerHTML = window.MiniMarkdown.render(pending);
      pending = null;
      if (stick) scrollToBottom();
    }
  }

  function schedule() {
    if (frame === null) frame = window.requestAnimationFrame(flush);
  }

  // ── 暴露给 Python 的接口 ──────────────────────────────────────────────────

  window.ChatHost = {
    appendMessage: function (role, text, meta) {
      append(role || "assistant", text, meta || {});
    },

    pushToken: function (text) {
      if (!active) active = append("assistant", "");
      pending = (pending === null ? active._text : pending) + (text || "");
      schedule();
      // 口型同步入口。现在 Live2D 未接入，这两行是空操作；
      // 接入后这里就是"模型说话时嘴在动"的唯一来源。
      if (window.Live2DHost) {
        window.Live2DHost.setMouthOpen(0.85);
        window.setTimeout(function () {
          window.Live2DHost.setMouthOpen(0.1);
        }, 90);
      }
    },

    endMessage: function () {
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
        frame = null;
      }
      flush();
      active = null;
      pending = null;
    },

    setBusy: function (value) {
      busy = !!value;
      document.body.classList.toggle("busy", busy);
      updateComposer();
    },

    setStatus: function (text) {
      statusLabel.textContent = text || "";
    },

    setVoiceState: function (state) {
      voiceState = state || "";
      updateComposer();
    },

    setCallState: function (state, meta) {
      callState = state || "";
      callMuted = !!(meta && meta.muted);
      var labels = {
        listening: "正在听…",
        capturing: "在听你说…",
        thinking: "她正在想…",
        speaking: "她正在说…（点「插话」打断）",
        error: "通话出错了",
        off: "通话已结束",
      };
      callStateLabel.textContent = labels[callState] || "";
      if (meta && meta.detail) callStateLabel.textContent += "  " + meta.detail;
      updateComposer();
    },

    setCallLevel: function (level) {
      var value = Math.max(0, Math.min(100, Number(level) || 0));
      callLevel.style.setProperty("--call-level", String(value));
    },

    setCallPartial: function (text) {
      callPartial.textContent = text || "";
    },

    clear: function () {
      resetStream();
    },

    loadHistory: function (messages) {
      resetStream();
      (messages || []).forEach(function (message) {
        append(message.role || "assistant", message.text || "", message.meta || {});
      });
      scrollToBottom();
    },

    /* 主题。**明暗由 Python 说了算**，见 css/app.css 顶部那段说明：两侧各检测
       一次系统偏好，迟早会在某台机器上分成两半。主色只发 --accent——网页侧
       用得到用户主色的地方只有链接、输入框焦点边和发送键，其余颜色由
       [data-theme] 那一块按明暗给（与 Qt 侧同一套 token 值）。 */
    setTheme: function (theme) {
      if (!theme) return;
      var root = document.documentElement;
      if (theme.mode) root.setAttribute("data-theme", String(theme.mode));
      if (theme.accent) root.style.setProperty("--accent", String(theme.accent));
    },

    /* 壁纸可见度，单位是百分比（与设置窗口的滑杆同一个数）。 */
    setWallpaperOpacity: function (percent) {
      var value = Number(percent);
      if (!isFinite(value)) return;
      var visible = Math.max(0, Math.min(100, value));
      document.documentElement.style.setProperty("--wallpaper-dim", String(1 - visible / 100));
    },

    setWallpaper: function (path) {
      var layer = document.getElementById("wallpaper");
      if (!layer) return;
      if (!path) {
        layer.style.backgroundImage = "";
        document.body.classList.remove("has-wallpaper");
        return;
      }
      var url = /^(https?:|file:|data:)/i.test(path)
        ? path
        : (window.BASE || "") + "/" + String(path).replace(/^\/+/, "");
      layer.style.backgroundImage = 'url("' + url.replace(/"/g, "%22") + '")';
      document.body.classList.add("has-wallpaper");
    },

    /* 人设卡片：要不要问、内置的那段是什么。**Python 每次启动都会推一次**，包括
       "不用问"（asking=false）——那一次同时也是把发送键解开的信号，所以它不能省。
       传进来的 builtin 可能带换行，写进 <p> 靠 CSS 的 white-space: pre-line 保住。 */
    setPersonaState: function (state) {
      if (channelDead || !welcome) return;
      var asking = !!(state && state.asking);
      if (welcomeBuiltin) welcomeBuiltin.textContent = (state && state.builtin) || "";
      personaPending = asking;
      welcome.hidden = !asking;
      // 收起时清空：答案已经收下了，留着它下次打开卡片会以为用户写了没提交。
      if (!asking) personaInput.value = "";
      personaSave.disabled = true; // 空框不可提交；打开后由输入内容决定
      updateEmpty();
      updateComposer();
      if (asking) personaInput.focus();
    },

    /* 确认卡片。meta.timeoutMs 决定倒计时；没有它就只显示按钮，不显示倒计时
       （闸门那一侧仍然会超时，只是界面上不预告——这比显示一个编出来的秒数诚实）。 */
    appendToolConfirm: function (callId, name, summary, meta) {
      if (!channelDead) appendToolCard(callId, name, summary || {}, meta || {});
    },

    /* 卡片的结果。**accepted 与 granted 是两件事**：granted 是用户点了什么，
       accepted 是闸门有没有采纳。accepted 为假时必须说清"这次点击没生效"——
       不区分的话，用户会以为他同意了，而闸门记的是没人同意过。 */
    resolveToolConfirm: function (callId, granted, accepted) {
      var entry = findCard(callId);
      if (!entry) return;
      var node = entry.node;
      entry.resolved = true;
      node._allow.disabled = true;
      node._deny.disabled = true;
      if (!accepted) {
        node._countdown.textContent = "这次点击没有生效（这一轮已经结束了）";
        node.classList.add("tool-stale");
        return;
      }
      node._countdown.textContent = granted ? "已允许" : "已拒绝";
      node.classList.toggle("tool-denied", !granted);
    },
  };

  // ── 输入 ──────────────────────────────────────────────────────────────────

  function submit() {
    var text = input.value.trim();
    if (!text || busy || personaPending || channelDead || !bridge) return;
    input.value = "";
    autosize();
    bridge.send(text);
  }

  function autosize() {
    input.style.height = "auto";
    var height = Math.min(input.scrollHeight, 220);
    input.style.height = height + "px";
  }

  input.addEventListener("input", autosize);
  input.addEventListener("keydown", function (event) {
    // Enter 发送、Shift+Enter 换行。不拦 Ctrl/Alt 组合键，留给系统快捷键。
    if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      submit();
    }
  });
  sendButton.addEventListener("click", submit);

  // 麦克风：**只发一次"切换"**，说什么由终端那边的识别决定（见 `app.py` 那张契约表）。
  // 守卫与 `submit` 同形：通道死了、正忙、正等人设时都不该开始录。
  micButton.addEventListener("click", function () {
    if (busy || personaPending || channelDead || !bridge || callState) return;
    bridge.voice_toggle();
  });

  // 通话：开始/挂断、插话、静音。**挂断永远可用**（正忙也要能挂）。
  callButton.addEventListener("click", function () {
    if (personaPending || channelDead || !bridge) return;
    bridge.call_toggle();
  });
  callHangup.addEventListener("click", function () {
    if (!bridge) return;
    bridge.call_toggle();
  });
  callInterrupt.addEventListener("click", function () {
    if (!bridge) return;
    bridge.call_interrupt();
  });
  callMute.addEventListener("click", function () {
    if (!bridge) return;
    bridge.call_mute(!callMuted);
  });
  // 空格插话：输入框里有焦点时不抢（那是打字用的）。
  document.addEventListener("keydown", function (event) {
    if (event.key !== " " || callState !== "speaking") return;
    var tag = (document.activeElement && document.activeElement.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    event.preventDefault();
    if (bridge) bridge.call_interrupt();
  });

  // ── 人设卡片 ──────────────────────────────────────────────────────────────
  //
  // 两个按钮**不重叠**：「就这样」要求框里真的有字（空框时它是灰的），「先不写」
  // 专门用来放行。所以不存在"空框点保存"这种要额外解释的情况——空着不写正是
  // 用户表达"不写"的方式，而它有自己的按钮。

  function answerPersona(text) {
    if (channelDead || !bridge) return;
    bridge.answer_persona(text);
  }

  personaInput.addEventListener("input", function () {
    personaSave.disabled = personaInput.value.trim() === "";
  });
  personaSave.addEventListener("click", function () {
    var text = personaInput.value.trim();
    if (text) answerPersona(text);
  });
  personaSkip.addEventListener("click", function () {
    answerPersona(""); // 空串 = 这一次先不写。为什么空串的含义不含糊，见 chat_view.py
  });

  // ── 与 Python 建立通道 ────────────────────────────────────────────────────

  /* 通道建立不起来时的**终态**。三处失败分支原本各写一遍"禁用输入框"，现在都走
     这里：输入框禁用、卡片收起（它的按钮点了也没人接），状态栏说清是哪一步断的。 */
  function killChannel(message) {
    channelDead = true;
    personaPending = false;
    welcome.hidden = true;
    updateComposer();
    window.ChatHost.setStatus(message);
  }

  function connect() {
    if (typeof QWebChannel === "undefined" || !window.qt || !window.qt.webChannelTransport) {
      killChannel("网页通道未建立：qwebchannel.js 没加载成功。");
      return;
    }
    new QWebChannel(window.qt.webChannelTransport, function (channel) {
      bridge = channel.objects.bridge;
      // 挂到 window 上。**这是刻意的**：QWebChannel 的连接是异步的，出问题时
      // （Python 侧忘记 registerObject、页面被 CSP 拦了、对象名拼错）表现是
      // "输入了没反应"——从界面上完全看不出是哪一步断的。挂出来之后，
      // 排查只需要一句 window.bridge 就能定位。
      window.bridge = bridge;
      if (!bridge) {
        killChannel("网页通道已建立，但找不到 bridge 对象。");
        return;
      }
      bridge.submitted.connect(function () {
        // Python 侧已经把消息显示出来了；这里只负责复位输入框焦点。
        input.focus();
      });
      window.ChatHost.setStatus("已连接终端。");
      input.focus();
    });
  }

  /* 拖进来的文件。**只认 file: 开头的 URI**——拖一段网页上的图片进来时
     dataTransfer 里是 http(s) 地址，而那条路是一条我们不该去取的网络请求。

     **两处 preventDefault 缺一不可**：dragover 不拦就没有 drop 事件；drop 不拦的话
     浏览器会直接**导航到那个文件**（整个对话区变成一张图或者一段纯文本，而用户以为
     程序坏了）。 */
  function hasFiles(event) {
    var transfer = event.dataTransfer;
    if (!transfer || !transfer.types) { return false; }
    for (var i = 0; i < transfer.types.length; i++) {
      // **`"Files"` 之外还要认 `text/uri-list`**：从资源管理器拖文件时，
      // QtWebEngine 并不总是给出前者；只认它会让整个拖放静默失效（或者让浏览器
      // 导航到那个文件）。非文件的 uri-list（拖一个链接）在 drop 里再区分。
      if (transfer.types[i] === "Files" || transfer.types[i] === "text/uri-list") {
        return true;
      }
    }
    return false;
  }

  function installDrop() {
    var zone = document.body;
    zone.addEventListener("dragover", function (event) {
      if (!hasFiles(event)) { return; }
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    });
    zone.addEventListener("drop", function (event) {
      // 拖一段**文字**进来照旧走默认行为（插进输入框），只有文件才归我们管。
      if (!hasFiles(event)) { return; }
      event.preventDefault();
      var list = event.dataTransfer.getData("text/uri-list") || "";
      // 不用正则：这个文件里一条反斜杠都不想要（它要经过好几层拼装才落到盘上）。
      // 换行与回车都用字符码取，跨平台也对。
      var lines = list.split(String.fromCharCode(10));
      var uris = [];
      for (var i = 0; i < lines.length; i++) {
        var one = lines[i].replace(String.fromCharCode(13), "").trim();
        if (!one || one.charAt(0) === "#") { continue; }
        if (one.slice(0, 5).toLowerCase() !== "file:") { continue; }
        uris.push(one);
      }
      if (uris.length) {
        window.bridge.attach_files(uris);
        return;
      }
      // 拖进来的是网页链接（不是本地文件）：按普通文本插进输入框，别让这次拖放
      // 因为我们拦过默认行为而消失。
      var text = event.dataTransfer.getData("text/plain");
      if (text && input) {
        var at = input.selectionStart === null ? input.value.length : input.selectionStart;
        input.value = input.value.slice(0, at) + text + input.value.slice(at);
        input.selectionStart = input.selectionEnd = at + text.length;
        autosize();
      }
    });
  }
  document.addEventListener("DOMContentLoaded", function () {
    autosize();
    connect();
    installDrop();
  });
  if (document.readyState !== "loading") {
    autosize();
    connect();
    installDrop();
  }
})();
