/*!
 * Webflow: Front-end site library
 * @license MIT
 * Inline scripts may access the api using an async handler:
 *   var Webflow = Webflow || [];
 *   Webflow.push(readyFunction);
 */ !(function (t) {
  var e = {};
  function n(r) {
    if (e[r]) return e[r].exports;
    var i = (e[r] = { i: r, l: !1, exports: {} });
    return t[r].call(i.exports, i, i.exports, n), (i.l = !0), i.exports;
  }
  (n.m = t),
    (n.c = e),
    (n.d = function (t, e, r) {
      n.o(t, e) || Object.defineProperty(t, e, { enumerable: !0, get: r });
    }),
    (n.r = function (t) {
      "undefined" != typeof Symbol &&
        Symbol.toStringTag &&
        Object.defineProperty(t, Symbol.toStringTag, { value: "Module" }),
        Object.defineProperty(t, "__esModule", { value: !0 });
    }),
    (n.t = function (t, e) {
      if ((1 & e && (t = n(t)), 8 & e)) return t;
      if (4 & e && "object" == typeof t && t && t.__esModule) return t;
      var r = Object.create(null);
      if (
        (n.r(r),
        Object.defineProperty(r, "default", { enumerable: !0, value: t }),
        2 & e && "string" != typeof t)
      )
        for (var i in t)
          n.d(
            r,
            i,
            function (e) {
              return t[e];
            }.bind(null, i)
          );
      return r;
    }),
    (n.n = function (t) {
      var e =
        t && t.__esModule
          ? function () {
              return t.default;
            }
          : function () {
              return t;
            };
      return n.d(e, "a", e), e;
    }),
    (n.o = function (t, e) {
      return Object.prototype.hasOwnProperty.call(t, e);
    }),
    (n.p = ""),
    n((n.s = 120));
})([
  function (t, e) {
    t.exports = function (t) {
      return t && t.__esModule ? t : { default: t };
    };
  },
  function (t, e) {
    var n = Array.isArray;
    t.exports = n;
  },
  function (t, e, n) {
    "use strict";
    var r = n(14);
    Object.defineProperty(e, "__esModule", { value: !0 });
    var i = { IX2EngineActionTypes: !0, IX2EngineConstants: !0 };
    e.IX2EngineConstants = e.IX2EngineActionTypes = void 0;
    var o = n(170);
    Object.keys(o).forEach(function (t) {
      "default" !== t &&
        "__esModule" !== t &&
        (Object.prototype.hasOwnProperty.call(i, t) ||
          Object.defineProperty(e, t, {
            enumerable: !0,
            get: function () {
              return o[t];
            },
          }));
    });
    var a = n(171);
    Object.keys(a).forEach(function (t) {
      "default" !== t &&
        "__esModule" !== t &&
        (Object.prototype.hasOwnProperty.call(i, t) ||
          Object.defineProperty(e, t, {
            enumerable: !0,
            get: function () {
              return a[t];
            },
          }));
    });
    var u = n(172);
    Object.keys(u).forEach(function (t) {
      "default" !== t &&
        "__esModule" !== t &&
        (Object.prototype.hasOwnProperty.call(i, t) ||
          Object.defineProperty(e, t, {
            enumerable: !0,
            get: function () {
              return u[t];
            },
          }));
    });
    var c = r(n(173));
    e.IX2EngineActionTypes = c;
    var s = r(n(174));
    e.IX2EngineConstants = s;
  },
  function (t, e, n) {
    "use strict";
    var r = {},
      i = {},
      o = [],
      a = window.Webflow || [],
      u = window.jQuery,
      c = u(window),
      s = u(document),
      f = u.isFunction,
      l = (r._ = n(122)),
      d = (r.tram = n(65) && u.tram),
      p = !1,
      v = !1;
    function h(t) {
      r.env() &&
        (f(t.design) && c.on("__wf_design", t.design),
        f(t.preview) && c.on("__wf_preview", t.preview)),
        f(t.destroy) && c.on("__wf_destroy", t.destroy),
        t.ready &&
          f(t.ready) &&
          (function (t) {
            if (p) return void t.ready();
            if (l.contains(o, t.ready)) return;
            o.push(t.ready);
          })(t);
    }
    function E(t) {
      f(t.design) && c.off("__wf_design", t.design),
        f(t.preview) && c.off("__wf_preview", t.preview),
        f(t.destroy) && c.off("__wf_destroy", t.destroy),
        t.ready &&
          f(t.ready) &&
          (function (t) {
            o = l.filter(o, function (e) {
              return e !== t.ready;
            });
          })(t);
    }
    (d.config.hideBackface = !1),
      (d.config.keepInherited = !0),
      (r.define = function (t, e, n) {
        i[t] && E(i[t]);
        var r = (i[t] = e(u, l, n) || {});
        return h(r), r;
      }),
      (r.require = function (t) {
        return i[t];
      }),
      (r.push = function (t) {
        p ? f(t) && t() : a.push(t);
      }),
      (r.env = function (t) {
        var e = window.__wf_design,
          n = void 0 !== e;
        return t
          ? "design" === t
            ? n && e
            : "preview" === t
            ? n && !e
            : "slug" === t
            ? n && window.__wf_slug
            : "editor" === t
            ? window.WebflowEditor
            : "test" === t
            ? window.__wf_test
            : "frame" === t
            ? window !== window.top
            : void 0
          : n;
      });
    var g,
      _ = navigator.userAgent.toLowerCase(),
      y = (r.env.touch =
        "ontouchstart" in window ||
        (window.DocumentTouch && document instanceof window.DocumentTouch)),
      m = (r.env.chrome =
        /chrome/.test(_) &&
        /Google/.test(navigator.vendor) &&
        parseInt(_.match(/chrome\/(\d+)\./)[1], 10)),
      I = (r.env.ios = /(ipod|iphone|ipad)/.test(_));
    (r.env.safari = /safari/.test(_) && !m && !I),
      y &&
        s.on("touchstart mousedown", function (t) {
          g = t.target;
        }),
      (r.validClick = y
        ? function (t) {
            return t === g || u.contains(t, g);
          }
        : function () {
            return !0;
          });
    var b,
      T = "resize.webflow orientationchange.webflow load.webflow";
    function O(t, e) {
      var n = [],
        r = {};
      return (
        (r.up = l.throttle(function (t) {
          l.each(n, function (e) {
            e(t);
          });
        })),
        t && e && t.on(e, r.up),
        (r.on = function (t) {
          "function" == typeof t && (l.contains(n, t) || n.push(t));
        }),
        (r.off = function (t) {
          n = arguments.length
            ? l.filter(n, function (e) {
                return e !== t;
              })
            : [];
        }),
        r
      );
    }
    function w(t) {
      f(t) && t();
    }
    function A() {
      b && (b.reject(), c.off("load", b.resolve)),
        (b = new u.Deferred()),
        c.on("load", b.resolve);
    }
    (r.resize = O(c, T)),
      (r.scroll = O(
        c,
        "scroll.webflow resize.webflow orientationchange.webflow load.webflow"
      )),
      (r.redraw = O()),
      (r.location = function (t) {
        window.location = t;
      }),
      r.env() && (r.location = function () {}),
      (r.ready = function () {
        (p = !0),
          v ? ((v = !1), l.each(i, h)) : l.each(o, w),
          l.each(a, w),
          r.resize.up();
      }),
      (r.load = function (t) {
        b.then(t);
      }),
      (r.destroy = function (t) {
        (t = t || {}),
          (v = !0),
          c.triggerHandler("__wf_destroy"),
          null != t.domready && (p = t.domready),
          l.each(i, E),
          r.resize.off(),
          r.scroll.off(),
          r.redraw.off(),
          (o = []),
          (a = []),
          "pending" === b.state() && A();
      }),
      u(r.ready),
      A(),
      (t.exports = window.Webflow = r);
  },
  function (t, e, n) {
    (function (e) {
      var n = "object",
        r = function (t) {
          return t && t.Math == Math && t;
        };
      t.exports =
        r(typeof globalThis == n && globalThis) ||
        r(typeof window == n && window) ||
        r(typeof self == n && self) ||
        r(typeof e == n && e) ||
        Function("return this")();
    }).call(this, n(23));
  },
  function (t, e, n) {
    var r = n(89),
      i = "object" == typeof self && self && self.Object === Object && self,
      o = r || i || Function("return this")();
    t.exports = o;
  },
  function (t, e) {
    t.exports = function (t) {
      var e = typeof t;
      return null != t && ("object" == e || "function" == e);
    };
  },
  function (t, e, n) {
    var r = n(177),
      i = n(231),
      o = n(59),
      a = n(1),
      u = n(240);
    t.exports = function (t) {
      return "function" == typeof t
        ? t
        : null == t
        ? o
        : "object" == typeof t
        ? a(t)
          ? i(t[0], t[1])
          : r(t)
        : u(t);
    };
  },
  function (t, e, n) {
    var r = n(189),
      i = n(194);
    t.exports = function (t, e) {
      var n = i(t, e);
      return r(n) ? n : void 0;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return null != t && "object" == typeof t;
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(14);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.IX2VanillaUtils =
        e.IX2VanillaPlugins =
        e.IX2ElementsReducer =
        e.IX2EasingUtils =
        e.IX2Easings =
        e.IX2BrowserSupport =
          void 0);
    var i = r(n(44));
    e.IX2BrowserSupport = i;
    var o = r(n(106));
    e.IX2Easings = o;
    var a = r(n(108));
    e.IX2EasingUtils = a;
    var u = r(n(247));
    e.IX2ElementsReducer = u;
    var c = r(n(110));
    e.IX2VanillaPlugins = c;
    var s = r(n(249));
    e.IX2VanillaUtils = s;
  },
  function (t, e, n) {
    var r = n(20),
      i = n(190),
      o = n(191),
      a = "[object Null]",
      u = "[object Undefined]",
      c = r ? r.toStringTag : void 0;
    t.exports = function (t) {
      return null == t
        ? void 0 === t
          ? u
          : a
        : c && c in Object(t)
        ? i(t)
        : o(t);
    };
  },
  function (t, e, n) {
    var r = n(88),
      i = n(52);
    t.exports = function (t) {
      return null != t && i(t.length) && !r(t);
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(124);
    function i(t, e) {
      var n = document.createEvent("CustomEvent");
      n.initCustomEvent(e, !0, !0, null), t.dispatchEvent(n);
    }
    var o = window.jQuery,
      a = {},
      u = {
        reset: function (t, e) {
          r.triggers.reset(t, e);
        },
        intro: function (t, e) {
          r.triggers.intro(t, e), i(e, "COMPONENT_ACTIVE");
        },
        outro: function (t, e) {
          r.triggers.outro(t, e), i(e, "COMPONENT_INACTIVE");
        },
      };
    (a.triggers = {}),
      (a.types = { INTRO: "w-ix-intro.w-ix", OUTRO: "w-ix-outro.w-ix" }),
      o.extend(a.triggers, u),
      (t.exports = a);
  },
  function (t, e) {
    t.exports = function (t) {
      if (t && t.__esModule) return t;
      var e = {};
      if (null != t)
        for (var n in t)
          if (Object.prototype.hasOwnProperty.call(t, n)) {
            var r =
              Object.defineProperty && Object.getOwnPropertyDescriptor
                ? Object.getOwnPropertyDescriptor(t, n)
                : {};
            r.get || r.set ? Object.defineProperty(e, n, r) : (e[n] = t[n]);
          }
      return (e.default = t), e;
    };
  },
  function (t, e, n) {
    var r = n(16);
    t.exports = !r(function () {
      return (
        7 !=
        Object.defineProperty({}, "a", {
          get: function () {
            return 7;
          },
        }).a
      );
    });
  },
  function (t, e) {
    t.exports = function (t) {
      try {
        return !!t();
      } catch (t) {
        return !0;
      }
    };
  },
  function (t, e) {
    var n = {}.hasOwnProperty;
    t.exports = function (t, e) {
      return n.call(t, e);
    };
  },
  function (t, e, n) {
    var r = n(15),
      i = n(38),
      o = n(67);
    t.exports = r
      ? function (t, e, n) {
          return i.f(t, e, o(1, n));
        }
      : function (t, e, n) {
          return (t[e] = n), t;
        };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 });
    var r =
      "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
        ? function (t) {
            return typeof t;
          }
        : function (t) {
            return t &&
              "function" == typeof Symbol &&
              t.constructor === Symbol &&
              t !== Symbol.prototype
              ? "symbol"
              : typeof t;
          };
    (e.clone = c),
      (e.addLast = l),
      (e.addFirst = d),
      (e.removeLast = p),
      (e.removeFirst = v),
      (e.insert = h),
      (e.removeAt = E),
      (e.replaceAt = g),
      (e.getIn = _),
      (e.set = y),
      (e.setIn = m),
      (e.update = I),
      (e.updateIn = b),
      (e.merge = T),
      (e.mergeDeep = O),
      (e.mergeIn = w),
      (e.omit = A),
      (e.addDefaults = S);
    /*!
     * Timm
     *
     * Immutability helpers with fast reads and acceptable writes.
     *
     * @copyright Guillermo Grau Panea 2016
     * @license MIT
     */
    var i = "INVALID_ARGS";
    function o(t) {
      throw new Error(t);
    }
    function a(t) {
      var e = Object.keys(t);
      return Object.getOwnPropertySymbols
        ? e.concat(Object.getOwnPropertySymbols(t))
        : e;
    }
    var u = {}.hasOwnProperty;
    function c(t) {
      if (Array.isArray(t)) return t.slice();
      for (var e = a(t), n = {}, r = 0; r < e.length; r++) {
        var i = e[r];
        n[i] = t[i];
      }
      return n;
    }
    function s(t, e, n) {
      var r = n;
      null == r && o(i);
      for (
        var u = !1, l = arguments.length, d = Array(l > 3 ? l - 3 : 0), p = 3;
        p < l;
        p++
      )
        d[p - 3] = arguments[p];
      for (var v = 0; v < d.length; v++) {
        var h = d[v];
        if (null != h) {
          var E = a(h);
          if (E.length)
            for (var g = 0; g <= E.length; g++) {
              var _ = E[g];
              if (!t || void 0 === r[_]) {
                var y = h[_];
                e && f(r[_]) && f(y) && (y = s(t, e, r[_], y)),
                  void 0 !== y &&
                    y !== r[_] &&
                    (u || ((u = !0), (r = c(r))), (r[_] = y));
              }
            }
        }
      }
      return r;
    }
    function f(t) {
      var e = void 0 === t ? "undefined" : r(t);
      return null != t && ("object" === e || "function" === e);
    }
    function l(t, e) {
      return Array.isArray(e) ? t.concat(e) : t.concat([e]);
    }
    function d(t, e) {
      return Array.isArray(e) ? e.concat(t) : [e].concat(t);
    }
    function p(t) {
      return t.length ? t.slice(0, t.length - 1) : t;
    }
    function v(t) {
      return t.length ? t.slice(1) : t;
    }
    function h(t, e, n) {
      return t
        .slice(0, e)
        .concat(Array.isArray(n) ? n : [n])
        .concat(t.slice(e));
    }
    function E(t, e) {
      return e >= t.length || e < 0 ? t : t.slice(0, e).concat(t.slice(e + 1));
    }
    function g(t, e, n) {
      if (t[e] === n) return t;
      for (var r = t.length, i = Array(r), o = 0; o < r; o++) i[o] = t[o];
      return (i[e] = n), i;
    }
    function _(t, e) {
      if ((!Array.isArray(e) && o(i), null != t)) {
        for (var n = t, r = 0; r < e.length; r++) {
          var a = e[r];
          if (void 0 === (n = null != n ? n[a] : void 0)) return n;
        }
        return n;
      }
    }
    function y(t, e, n) {
      var r = null == t ? ("number" == typeof e ? [] : {}) : t;
      if (r[e] === n) return r;
      var i = c(r);
      return (i[e] = n), i;
    }
    function m(t, e, n) {
      return e.length
        ? (function t(e, n, r, i) {
            var o = void 0,
              a = n[i];
            o =
              i === n.length - 1
                ? r
                : t(
                    f(e) && f(e[a])
                      ? e[a]
                      : "number" == typeof n[i + 1]
                      ? []
                      : {},
                    n,
                    r,
                    i + 1
                  );
            return y(e, a, o);
          })(t, e, n, 0)
        : n;
    }
    function I(t, e, n) {
      return y(t, e, n(null == t ? void 0 : t[e]));
    }
    function b(t, e, n) {
      return m(t, e, n(_(t, e)));
    }
    function T(t, e, n, r, i, o) {
      for (
        var a = arguments.length, u = Array(a > 6 ? a - 6 : 0), c = 6;
        c < a;
        c++
      )
        u[c - 6] = arguments[c];
      return u.length
        ? s.call.apply(s, [null, !1, !1, t, e, n, r, i, o].concat(u))
        : s(!1, !1, t, e, n, r, i, o);
    }
    function O(t, e, n, r, i, o) {
      for (
        var a = arguments.length, u = Array(a > 6 ? a - 6 : 0), c = 6;
        c < a;
        c++
      )
        u[c - 6] = arguments[c];
      return u.length
        ? s.call.apply(s, [null, !1, !0, t, e, n, r, i, o].concat(u))
        : s(!1, !0, t, e, n, r, i, o);
    }
    function w(t, e, n, r, i, o, a) {
      var u = _(t, e);
      null == u && (u = {});
      for (
        var c = arguments.length, f = Array(c > 7 ? c - 7 : 0), l = 7;
        l < c;
        l++
      )
        f[l - 7] = arguments[l];
      return m(
        t,
        e,
        f.length
          ? s.call.apply(s, [null, !1, !1, u, n, r, i, o, a].concat(f))
          : s(!1, !1, u, n, r, i, o, a)
      );
    }
    function A(t, e) {
      for (var n = Array.isArray(e) ? e : [e], r = !1, i = 0; i < n.length; i++)
        if (u.call(t, n[i])) {
          r = !0;
          break;
        }
      if (!r) return t;
      for (var o = {}, c = a(t), s = 0; s < c.length; s++) {
        var f = c[s];
        n.indexOf(f) >= 0 || (o[f] = t[f]);
      }
      return o;
    }
    function S(t, e, n, r, i, o) {
      for (
        var a = arguments.length, u = Array(a > 6 ? a - 6 : 0), c = 6;
        c < a;
        c++
      )
        u[c - 6] = arguments[c];
      return u.length
        ? s.call.apply(s, [null, !0, !1, t, e, n, r, i, o].concat(u))
        : s(!0, !1, t, e, n, r, i, o);
    }
    var x = {
      clone: c,
      addLast: l,
      addFirst: d,
      removeLast: p,
      removeFirst: v,
      insert: h,
      removeAt: E,
      replaceAt: g,
      getIn: _,
      set: y,
      setIn: m,
      update: I,
      updateIn: b,
      merge: T,
      mergeDeep: O,
      mergeIn: w,
      omit: A,
      addDefaults: S,
    };
    e.default = x;
  },
  function (t, e, n) {
    var r = n(5).Symbol;
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(36),
      i = 1 / 0;
    t.exports = function (t) {
      if ("string" == typeof t || r(t)) return t;
      var e = t + "";
      return "0" == e && 1 / t == -i ? "-0" : e;
    };
  },
  function (t, e) {
    function n(t) {
      return (n =
        "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
          ? function (t) {
              return typeof t;
            }
          : function (t) {
              return t &&
                "function" == typeof Symbol &&
                t.constructor === Symbol &&
                t !== Symbol.prototype
                ? "symbol"
                : typeof t;
            })(t);
    }
    function r(e) {
      return (
        "function" == typeof Symbol && "symbol" === n(Symbol.iterator)
          ? (t.exports = r =
              function (t) {
                return n(t);
              })
          : (t.exports = r =
              function (t) {
                return t &&
                  "function" == typeof Symbol &&
                  t.constructor === Symbol &&
                  t !== Symbol.prototype
                  ? "symbol"
                  : n(t);
              }),
        r(e)
      );
    }
    t.exports = r;
  },
  function (t, e) {
    var n;
    n = (function () {
      return this;
    })();
    try {
      n = n || new Function("return this")();
    } catch (t) {
      "object" == typeof window && (n = window);
    }
    t.exports = n;
  },
  function (t, e) {
    t.exports = function (t) {
      return "object" == typeof t ? null !== t : "function" == typeof t;
    };
  },
  function (t, e, n) {
    var r = n(24);
    t.exports = function (t) {
      if (!r(t)) throw TypeError(String(t) + " is not an object");
      return t;
    };
  },
  function (t, e, n) {
    var r = n(4),
      i = n(39),
      o = n(136),
      a = r["__core-js_shared__"] || i("__core-js_shared__", {});
    (t.exports = function (t, e) {
      return a[t] || (a[t] = void 0 !== e ? e : {});
    })("versions", []).push({
      version: "3.1.3",
      mode: o ? "pure" : "global",
      copyright: "Â© 2019 Denis Pushkarev (zloirock.ru)",
    });
  },
  function (t, e) {
    t.exports = function (t, e, n) {
      return (
        e in t
          ? Object.defineProperty(t, e, {
              value: n,
              enumerable: !0,
              configurable: !0,
              writable: !0,
            })
          : (t[e] = n),
        t
      );
    };
  },
  function (t, e) {
    function n() {
      return (
        (t.exports = n =
          Object.assign ||
          function (t) {
            for (var e = 1; e < arguments.length; e++) {
              var n = arguments[e];
              for (var r in n)
                Object.prototype.hasOwnProperty.call(n, r) && (t[r] = n[r]);
            }
            return t;
          }),
        n.apply(this, arguments)
      );
    }
    t.exports = n;
  },
  function (t, e, n) {
    var r = n(179),
      i = n(180),
      o = n(181),
      a = n(182),
      u = n(183);
    function c(t) {
      var e = -1,
        n = null == t ? 0 : t.length;
      for (this.clear(); ++e < n; ) {
        var r = t[e];
        this.set(r[0], r[1]);
      }
    }
    (c.prototype.clear = r),
      (c.prototype.delete = i),
      (c.prototype.get = o),
      (c.prototype.has = a),
      (c.prototype.set = u),
      (t.exports = c);
  },
  function (t, e, n) {
    var r = n(45);
    t.exports = function (t, e) {
      for (var n = t.length; n--; ) if (r(t[n][0], e)) return n;
      return -1;
    };
  },
  function (t, e, n) {
    var r = n(8)(Object, "create");
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(203);
    t.exports = function (t, e) {
      var n = t.__data__;
      return r(e) ? n["string" == typeof e ? "string" : "hash"] : n.map;
    };
  },
  function (t, e, n) {
    var r = n(96),
      i = n(53),
      o = n(12);
    t.exports = function (t) {
      return o(t) ? r(t) : i(t);
    };
  },
  function (t, e, n) {
    var r = n(221),
      i = n(9),
      o = Object.prototype,
      a = o.hasOwnProperty,
      u = o.propertyIsEnumerable,
      c = r(
        (function () {
          return arguments;
        })()
      )
        ? r
        : function (t) {
            return i(t) && a.call(t, "callee") && !u.call(t, "callee");
          };
    t.exports = c;
  },
  function (t, e, n) {
    var r = n(1),
      i = n(58),
      o = n(232),
      a = n(235);
    t.exports = function (t, e) {
      return r(t) ? t : i(t, e) ? [t] : o(a(t));
    };
  },
  function (t, e, n) {
    var r = n(11),
      i = n(9),
      o = "[object Symbol]";
    t.exports = function (t) {
      return "symbol" == typeof t || (i(t) && r(t) == o);
    };
  },
  function (t, e, n) {
    var r = n(132),
      i = n(134);
    t.exports = function (t) {
      return r(i(t));
    };
  },
  function (t, e, n) {
    var r = n(15),
      i = n(69),
      o = n(25),
      a = n(68),
      u = Object.defineProperty;
    e.f = r
      ? u
      : function (t, e, n) {
          if ((o(t), (e = a(e, !0)), o(n), i))
            try {
              return u(t, e, n);
            } catch (t) {}
          if ("get" in n || "set" in n)
            throw TypeError("Accessors not supported");
          return "value" in n && (t[e] = n.value), t;
        };
  },
  function (t, e, n) {
    var r = n(4),
      i = n(18);
    t.exports = function (t, e) {
      try {
        i(r, t, e);
      } catch (n) {
        r[t] = e;
      }
      return e;
    };
  },
  function (t, e) {
    t.exports = {};
  },
  function (t, e) {
    t.exports = [
      "constructor",
      "hasOwnProperty",
      "isPrototypeOf",
      "propertyIsEnumerable",
      "toLocaleString",
      "toString",
      "valueOf",
    ];
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      n.d(e, "ActionTypes", function () {
        return o;
      }),
      n.d(e, "default", function () {
        return a;
      });
    var r = n(79),
      i = n(165),
      o = { INIT: "@@redux/INIT" };
    function a(t, e, n) {
      var u;
      if (
        ("function" == typeof e && void 0 === n && ((n = e), (e = void 0)),
        void 0 !== n)
      ) {
        if ("function" != typeof n)
          throw new Error("Expected the enhancer to be a function.");
        return n(a)(t, e);
      }
      if ("function" != typeof t)
        throw new Error("Expected the reducer to be a function.");
      var c = t,
        s = e,
        f = [],
        l = f,
        d = !1;
      function p() {
        l === f && (l = f.slice());
      }
      function v() {
        return s;
      }
      function h(t) {
        if ("function" != typeof t)
          throw new Error("Expected listener to be a function.");
        var e = !0;
        return (
          p(),
          l.push(t),
          function () {
            if (e) {
              (e = !1), p();
              var n = l.indexOf(t);
              l.splice(n, 1);
            }
          }
        );
      }
      function E(t) {
        if (!Object(r.default)(t))
          throw new Error(
            "Actions must be plain objects. Use custom middleware for async actions."
          );
        if (void 0 === t.type)
          throw new Error(
            'Actions may not have an undefined "type" property. Have you misspelled a constant?'
          );
        if (d) throw new Error("Reducers may not dispatch actions.");
        try {
          (d = !0), (s = c(s, t));
        } finally {
          d = !1;
        }
        for (var e = (f = l), n = 0; n < e.length; n++) e[n]();
        return t;
      }
      return (
        E({ type: o.INIT }),
        ((u = {
          dispatch: E,
          subscribe: h,
          getState: v,
          replaceReducer: function (t) {
            if ("function" != typeof t)
              throw new Error("Expected the nextReducer to be a function.");
            (c = t), E({ type: o.INIT });
          },
        })[i.default] = function () {
          var t,
            e = h;
          return (
            ((t = {
              subscribe: function (t) {
                if ("object" != typeof t)
                  throw new TypeError("Expected the observer to be an object.");
                function n() {
                  t.next && t.next(v());
                }
                return n(), { unsubscribe: e(n) };
              },
            })[i.default] = function () {
              return this;
            }),
            t
          );
        }),
        u
      );
    }
  },
  function (t, e, n) {
    "use strict";
    function r() {
      for (var t = arguments.length, e = Array(t), n = 0; n < t; n++)
        e[n] = arguments[n];
      if (0 === e.length)
        return function (t) {
          return t;
        };
      if (1 === e.length) return e[0];
      var r = e[e.length - 1],
        i = e.slice(0, -1);
      return function () {
        return i.reduceRight(function (t, e) {
          return e(t);
        }, r.apply(void 0, arguments));
      };
    }
    n.r(e),
      n.d(e, "default", function () {
        return r;
      });
  },
  function (t, e, n) {
    "use strict";
    var r = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.TRANSFORM_STYLE_PREFIXED =
        e.TRANSFORM_PREFIXED =
        e.FLEX_PREFIXED =
        e.ELEMENT_MATCHES =
        e.withBrowser =
        e.IS_BROWSER_ENV =
          void 0);
    var i = r(n(85)),
      o = "undefined" != typeof window;
    e.IS_BROWSER_ENV = o;
    var a = function (t, e) {
      return o ? t() : e;
    };
    e.withBrowser = a;
    var u = a(function () {
      return (0,
      i.default)(["matches", "matchesSelector", "mozMatchesSelector", "msMatchesSelector", "oMatchesSelector", "webkitMatchesSelector"], function (t) {
        return t in Element.prototype;
      });
    });
    e.ELEMENT_MATCHES = u;
    var c = a(function () {
      var t = document.createElement("i"),
        e = ["flex", "-webkit-flex", "-ms-flexbox", "-moz-box", "-webkit-box"];
      try {
        for (var n = e.length, r = 0; r < n; r++) {
          var i = e[r];
          if (((t.style.display = i), t.style.display === i)) return i;
        }
        return "";
      } catch (t) {
        return "";
      }
    }, "flex");
    e.FLEX_PREFIXED = c;
    var s = a(function () {
      var t = document.createElement("i");
      if (null == t.style.transform)
        for (var e = ["Webkit", "Moz", "ms"], n = e.length, r = 0; r < n; r++) {
          var i = e[r] + "Transform";
          if (void 0 !== t.style[i]) return i;
        }
      return "transform";
    }, "transform");
    e.TRANSFORM_PREFIXED = s;
    var f = s.split("transform")[0],
      l = f ? f + "TransformStyle" : "transformStyle";
    e.TRANSFORM_STYLE_PREFIXED = l;
  },
  function (t, e) {
    t.exports = function (t, e) {
      return t === e || (t != t && e != e);
    };
  },
  function (t, e, n) {
    var r = n(8)(n(5), "Map");
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(195),
      i = n(202),
      o = n(204),
      a = n(205),
      u = n(206);
    function c(t) {
      var e = -1,
        n = null == t ? 0 : t.length;
      for (this.clear(); ++e < n; ) {
        var r = t[e];
        this.set(r[0], r[1]);
      }
    }
    (c.prototype.clear = r),
      (c.prototype.delete = i),
      (c.prototype.get = o),
      (c.prototype.has = a),
      (c.prototype.set = u),
      (t.exports = c);
  },
  function (t, e) {
    t.exports = function (t, e) {
      for (var n = -1, r = e.length, i = t.length; ++n < r; ) t[i + n] = e[n];
      return t;
    };
  },
  function (t, e, n) {
    (function (t) {
      var r = n(5),
        i = n(222),
        o = e && !e.nodeType && e,
        a = o && "object" == typeof t && t && !t.nodeType && t,
        u = a && a.exports === o ? r.Buffer : void 0,
        c = (u ? u.isBuffer : void 0) || i;
      t.exports = c;
    }).call(this, n(97)(t));
  },
  function (t, e) {
    var n = 9007199254740991,
      r = /^(?:0|[1-9]\d*)$/;
    t.exports = function (t, e) {
      var i = typeof t;
      return (
        !!(e = null == e ? n : e) &&
        ("number" == i || ("symbol" != i && r.test(t))) &&
        t > -1 &&
        t % 1 == 0 &&
        t < e
      );
    };
  },
  function (t, e, n) {
    var r = n(223),
      i = n(224),
      o = n(225),
      a = o && o.isTypedArray,
      u = a ? i(a) : r;
    t.exports = u;
  },
  function (t, e) {
    var n = 9007199254740991;
    t.exports = function (t) {
      return "number" == typeof t && t > -1 && t % 1 == 0 && t <= n;
    };
  },
  function (t, e, n) {
    var r = n(54),
      i = n(226),
      o = Object.prototype.hasOwnProperty;
    t.exports = function (t) {
      if (!r(t)) return i(t);
      var e = [];
      for (var n in Object(t)) o.call(t, n) && "constructor" != n && e.push(n);
      return e;
    };
  },
  function (t, e) {
    var n = Object.prototype;
    t.exports = function (t) {
      var e = t && t.constructor;
      return t === (("function" == typeof e && e.prototype) || n);
    };
  },
  function (t, e, n) {
    var r = n(227),
      i = n(46),
      o = n(228),
      a = n(229),
      u = n(99),
      c = n(11),
      s = n(90),
      f = s(r),
      l = s(i),
      d = s(o),
      p = s(a),
      v = s(u),
      h = c;
    ((r && "[object DataView]" != h(new r(new ArrayBuffer(1)))) ||
      (i && "[object Map]" != h(new i())) ||
      (o && "[object Promise]" != h(o.resolve())) ||
      (a && "[object Set]" != h(new a())) ||
      (u && "[object WeakMap]" != h(new u()))) &&
      (h = function (t) {
        var e = c(t),
          n = "[object Object]" == e ? t.constructor : void 0,
          r = n ? s(n) : "";
        if (r)
          switch (r) {
            case f:
              return "[object DataView]";
            case l:
              return "[object Map]";
            case d:
              return "[object Promise]";
            case p:
              return "[object Set]";
            case v:
              return "[object WeakMap]";
          }
        return e;
      }),
      (t.exports = h);
  },
  function (t, e, n) {
    var r = n(57);
    t.exports = function (t, e, n) {
      var i = null == t ? void 0 : r(t, e);
      return void 0 === i ? n : i;
    };
  },
  function (t, e, n) {
    var r = n(35),
      i = n(21);
    t.exports = function (t, e) {
      for (var n = 0, o = (e = r(e, t)).length; null != t && n < o; )
        t = t[i(e[n++])];
      return n && n == o ? t : void 0;
    };
  },
  function (t, e, n) {
    var r = n(1),
      i = n(36),
      o = /\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/,
      a = /^\w*$/;
    t.exports = function (t, e) {
      if (r(t)) return !1;
      var n = typeof t;
      return (
        !(
          "number" != n &&
          "symbol" != n &&
          "boolean" != n &&
          null != t &&
          !i(t)
        ) ||
        a.test(t) ||
        !o.test(t) ||
        (null != e && t in Object(e))
      );
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return t;
    };
  },
  function (t, e, n) {
    var r = n(6),
      i = n(36),
      o = NaN,
      a = /^\s+|\s+$/g,
      u = /^[-+]0x[0-9a-f]+$/i,
      c = /^0b[01]+$/i,
      s = /^0o[0-7]+$/i,
      f = parseInt;
    t.exports = function (t) {
      if ("number" == typeof t) return t;
      if (i(t)) return o;
      if (r(t)) {
        var e = "function" == typeof t.valueOf ? t.valueOf() : t;
        t = r(e) ? e + "" : e;
      }
      if ("string" != typeof t) return 0 === t ? t : +t;
      t = t.replace(a, "");
      var n = c.test(t);
      return n || s.test(t) ? f(t.slice(2), n ? 2 : 8) : u.test(t) ? o : +t;
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.mediaQueriesDefined =
        e.viewportWidthChanged =
        e.actionListPlaybackChanged =
        e.elementStateChanged =
        e.instanceRemoved =
        e.instanceStarted =
        e.instanceAdded =
        e.parameterChanged =
        e.animationFrameChanged =
        e.eventStateChanged =
        e.testFrameRendered =
        e.eventListenerAdded =
        e.clearRequested =
        e.stopRequested =
        e.playbackRequested =
        e.previewRequested =
        e.sessionStopped =
        e.sessionStarted =
        e.sessionInitialized =
        e.rawDataImported =
          void 0);
    var i = r(n(28)),
      o = n(2),
      a = n(10),
      u = o.IX2EngineActionTypes,
      c = u.IX2_RAW_DATA_IMPORTED,
      s = u.IX2_SESSION_INITIALIZED,
      f = u.IX2_SESSION_STARTED,
      l = u.IX2_SESSION_STOPPED,
      d = u.IX2_PREVIEW_REQUESTED,
      p = u.IX2_PLAYBACK_REQUESTED,
      v = u.IX2_STOP_REQUESTED,
      h = u.IX2_CLEAR_REQUESTED,
      E = u.IX2_EVENT_LISTENER_ADDED,
      g = u.IX2_TEST_FRAME_RENDERED,
      _ = u.IX2_EVENT_STATE_CHANGED,
      y = u.IX2_ANIMATION_FRAME_CHANGED,
      m = u.IX2_PARAMETER_CHANGED,
      I = u.IX2_INSTANCE_ADDED,
      b = u.IX2_INSTANCE_STARTED,
      T = u.IX2_INSTANCE_REMOVED,
      O = u.IX2_ELEMENT_STATE_CHANGED,
      w = u.IX2_ACTION_LIST_PLAYBACK_CHANGED,
      A = u.IX2_VIEWPORT_WIDTH_CHANGED,
      S = u.IX2_MEDIA_QUERIES_DEFINED,
      x = a.IX2VanillaUtils.reifyState;
    e.rawDataImported = function (t) {
      return { type: c, payload: (0, i.default)({}, x(t)) };
    };
    e.sessionInitialized = function (t) {
      var e = t.hasBoundaryNodes;
      return { type: s, payload: { hasBoundaryNodes: e } };
    };
    e.sessionStarted = function () {
      return { type: f };
    };
    e.sessionStopped = function () {
      return { type: l };
    };
    e.previewRequested = function (t) {
      var e = t.rawData,
        n = t.defer;
      return { type: d, payload: { defer: n, rawData: e } };
    };
    e.playbackRequested = function (t) {
      var e = t.actionTypeId,
        n = void 0 === e ? o.ActionTypeConsts.GENERAL_START_ACTION : e,
        r = t.actionListId,
        i = t.actionItemId,
        a = t.eventId,
        u = t.allowEvents,
        c = t.immediate,
        s = t.testManual,
        f = t.verbose,
        l = t.rawData;
      return {
        type: p,
        payload: {
          actionTypeId: n,
          actionListId: r,
          actionItemId: i,
          testManual: s,
          eventId: a,
          allowEvents: u,
          immediate: c,
          verbose: f,
          rawData: l,
        },
      };
    };
    e.stopRequested = function (t) {
      return { type: v, payload: { actionListId: t } };
    };
    e.clearRequested = function () {
      return { type: h };
    };
    e.eventListenerAdded = function (t, e) {
      return { type: E, payload: { target: t, listenerParams: e } };
    };
    e.testFrameRendered = function () {
      var t =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : 1;
      return { type: g, payload: { step: t } };
    };
    e.eventStateChanged = function (t, e) {
      return { type: _, payload: { stateKey: t, newState: e } };
    };
    e.animationFrameChanged = function (t, e) {
      return { type: y, payload: { now: t, parameters: e } };
    };
    e.parameterChanged = function (t, e) {
      return { type: m, payload: { key: t, value: e } };
    };
    e.instanceAdded = function (t) {
      return { type: I, payload: (0, i.default)({}, t) };
    };
    e.instanceStarted = function (t, e) {
      return { type: b, payload: { instanceId: t, time: e } };
    };
    e.instanceRemoved = function (t) {
      return { type: T, payload: { instanceId: t } };
    };
    e.elementStateChanged = function (t, e, n, r) {
      return {
        type: O,
        payload: { elementId: t, actionTypeId: e, current: n, actionItem: r },
      };
    };
    e.actionListPlaybackChanged = function (t) {
      var e = t.actionListId,
        n = t.isPlaying;
      return { type: w, payload: { actionListId: e, isPlaying: n } };
    };
    e.viewportWidthChanged = function (t) {
      var e = t.width,
        n = t.mediaQueries;
      return { type: A, payload: { width: e, mediaQueries: n } };
    };
    e.mediaQueriesDefined = function () {
      return { type: S };
    };
  },
  function (t, e, n) {
    var r = n(117),
      i = n(63);
    function o(t, e) {
      (this.__wrapped__ = t),
        (this.__actions__ = []),
        (this.__chain__ = !!e),
        (this.__index__ = 0),
        (this.__values__ = void 0);
    }
    (o.prototype = r(i.prototype)),
      (o.prototype.constructor = o),
      (t.exports = o);
  },
  function (t, e) {
    t.exports = function () {};
  },
  function (t, e, n) {
    var r = n(117),
      i = n(63),
      o = 4294967295;
    function a(t) {
      (this.__wrapped__ = t),
        (this.__actions__ = []),
        (this.__dir__ = 1),
        (this.__filtered__ = !1),
        (this.__iteratees__ = []),
        (this.__takeCount__ = o),
        (this.__views__ = []);
    }
    (a.prototype = r(i.prototype)),
      (a.prototype.constructor = a),
      (t.exports = a);
  },
  function (t, e, n) {
    "use strict";
    var r = n(0)(n(22));
    window.tram = (function (t) {
      function e(t, e) {
        return new F.Bare().init(t, e);
      }
      function n(t) {
        return t.replace(/[A-Z]/g, function (t) {
          return "-" + t.toLowerCase();
        });
      }
      function i(t) {
        var e = parseInt(t.slice(1), 16);
        return [(e >> 16) & 255, (e >> 8) & 255, 255 & e];
      }
      function o(t, e, n) {
        return (
          "#" + ((1 << 24) | (t << 16) | (e << 8) | n).toString(16).slice(1)
        );
      }
      function a() {}
      function u(t, e, n) {
        s("Units do not match [" + t + "]: " + e + ", " + n);
      }
      function c(t, e, n) {
        if ((void 0 !== e && (n = e), void 0 === t)) return n;
        var r = n;
        return (
          $.test(t) || !Z.test(t)
            ? (r = parseInt(t, 10))
            : Z.test(t) && (r = 1e3 * parseFloat(t)),
          0 > r && (r = 0),
          r == r ? r : n
        );
      }
      function s(t) {
        B.debug && window && window.console.warn(t);
      }
      var f = (function (t, e, n) {
          function i(t) {
            return "object" == (0, r.default)(t);
          }
          function o(t) {
            return "function" == typeof t;
          }
          function a() {}
          return function r(u, c) {
            function s() {
              var t = new f();
              return o(t.init) && t.init.apply(t, arguments), t;
            }
            function f() {}
            c === n && ((c = u), (u = Object)), (s.Bare = f);
            var l,
              d = (a[t] = u[t]),
              p = (f[t] = s[t] = new a());
            return (
              (p.constructor = s),
              (s.mixin = function (e) {
                return (f[t] = s[t] = r(s, e)[t]), s;
              }),
              (s.open = function (t) {
                if (
                  ((l = {}),
                  o(t) ? (l = t.call(s, p, d, s, u)) : i(t) && (l = t),
                  i(l))
                )
                  for (var n in l) e.call(l, n) && (p[n] = l[n]);
                return o(p.init) || (p.init = u), s;
              }),
              s.open(c)
            );
          };
        })("prototype", {}.hasOwnProperty),
        l = {
          ease: [
            "ease",
            function (t, e, n, r) {
              var i = (t /= r) * t,
                o = i * t;
              return (
                e +
                n * (-2.75 * o * i + 11 * i * i + -15.5 * o + 8 * i + 0.25 * t)
              );
            },
          ],
          "ease-in": [
            "ease-in",
            function (t, e, n, r) {
              var i = (t /= r) * t,
                o = i * t;
              return e + n * (-1 * o * i + 3 * i * i + -3 * o + 2 * i);
            },
          ],
          "ease-out": [
            "ease-out",
            function (t, e, n, r) {
              var i = (t /= r) * t,
                o = i * t;
              return (
                e +
                n * (0.3 * o * i + -1.6 * i * i + 2.2 * o + -1.8 * i + 1.9 * t)
              );
            },
          ],
          "ease-in-out": [
            "ease-in-out",
            function (t, e, n, r) {
              var i = (t /= r) * t,
                o = i * t;
              return e + n * (2 * o * i + -5 * i * i + 2 * o + 2 * i);
            },
          ],
          linear: [
            "linear",
            function (t, e, n, r) {
              return (n * t) / r + e;
            },
          ],
          "ease-in-quad": [
            "cubic-bezier(0.550, 0.085, 0.680, 0.530)",
            function (t, e, n, r) {
              return n * (t /= r) * t + e;
            },
          ],
          "ease-out-quad": [
            "cubic-bezier(0.250, 0.460, 0.450, 0.940)",
            function (t, e, n, r) {
              return -n * (t /= r) * (t - 2) + e;
            },
          ],
          "ease-in-out-quad": [
            "cubic-bezier(0.455, 0.030, 0.515, 0.955)",
            function (t, e, n, r) {
              return (t /= r / 2) < 1
                ? (n / 2) * t * t + e
                : (-n / 2) * (--t * (t - 2) - 1) + e;
            },
          ],
          "ease-in-cubic": [
            "cubic-bezier(0.550, 0.055, 0.675, 0.190)",
            function (t, e, n, r) {
              return n * (t /= r) * t * t + e;
            },
          ],
          "ease-out-cubic": [
            "cubic-bezier(0.215, 0.610, 0.355, 1)",
            function (t, e, n, r) {
              return n * ((t = t / r - 1) * t * t + 1) + e;
            },
          ],
          "ease-in-out-cubic": [
            "cubic-bezier(0.645, 0.045, 0.355, 1)",
            function (t, e, n, r) {
              return (t /= r / 2) < 1
                ? (n / 2) * t * t * t + e
                : (n / 2) * ((t -= 2) * t * t + 2) + e;
            },
          ],
          "ease-in-quart": [
            "cubic-bezier(0.895, 0.030, 0.685, 0.220)",
            function (t, e, n, r) {
              return n * (t /= r) * t * t * t + e;
            },
          ],
          "ease-out-quart": [
            "cubic-bezier(0.165, 0.840, 0.440, 1)",
            function (t, e, n, r) {
              return -n * ((t = t / r - 1) * t * t * t - 1) + e;
            },
          ],
          "ease-in-out-quart": [
            "cubic-bezier(0.770, 0, 0.175, 1)",
            function (t, e, n, r) {
              return (t /= r / 2) < 1
                ? (n / 2) * t * t * t * t + e
                : (-n / 2) * ((t -= 2) * t * t * t - 2) + e;
            },
          ],
          "ease-in-quint": [
            "cubic-bezier(0.755, 0.050, 0.855, 0.060)",
            function (t, e, n, r) {
              return n * (t /= r) * t * t * t * t + e;
            },
          ],
          "ease-out-quint": [
            "cubic-bezier(0.230, 1, 0.320, 1)",
            function (t, e, n, r) {
              return n * ((t = t / r - 1) * t * t * t * t + 1) + e;
            },
          ],
          "ease-in-out-quint": [
            "cubic-bezier(0.860, 0, 0.070, 1)",
            function (t, e, n, r) {
              return (t /= r / 2) < 1
                ? (n / 2) * t * t * t * t * t + e
                : (n / 2) * ((t -= 2) * t * t * t * t + 2) + e;
            },
          ],
          "ease-in-sine": [
            "cubic-bezier(0.470, 0, 0.745, 0.715)",
            function (t, e, n, r) {
              return -n * Math.cos((t / r) * (Math.PI / 2)) + n + e;
            },
          ],
          "ease-out-sine": [
            "cubic-bezier(0.390, 0.575, 0.565, 1)",
            function (t, e, n, r) {
              return n * Math.sin((t / r) * (Math.PI / 2)) + e;
            },
          ],
          "ease-in-out-sine": [
            "cubic-bezier(0.445, 0.050, 0.550, 0.950)",
            function (t, e, n, r) {
              return (-n / 2) * (Math.cos((Math.PI * t) / r) - 1) + e;
            },
          ],
          "ease-in-expo": [
            "cubic-bezier(0.950, 0.050, 0.795, 0.035)",
            function (t, e, n, r) {
              return 0 === t ? e : n * Math.pow(2, 10 * (t / r - 1)) + e;
            },
          ],
          "ease-out-expo": [
            "cubic-bezier(0.190, 1, 0.220, 1)",
            function (t, e, n, r) {
              return t === r ? e + n : n * (1 - Math.pow(2, (-10 * t) / r)) + e;
            },
          ],
          "ease-in-out-expo": [
            "cubic-bezier(1, 0, 0, 1)",
            function (t, e, n, r) {
              return 0 === t
                ? e
                : t === r
                ? e + n
                : (t /= r / 2) < 1
                ? (n / 2) * Math.pow(2, 10 * (t - 1)) + e
                : (n / 2) * (2 - Math.pow(2, -10 * --t)) + e;
            },
          ],
          "ease-in-circ": [
            "cubic-bezier(0.600, 0.040, 0.980, 0.335)",
            function (t, e, n, r) {
              return -n * (Math.sqrt(1 - (t /= r) * t) - 1) + e;
            },
          ],
          "ease-out-circ": [
            "cubic-bezier(0.075, 0.820, 0.165, 1)",
            function (t, e, n, r) {
              return n * Math.sqrt(1 - (t = t / r - 1) * t) + e;
            },
          ],
          "ease-in-out-circ": [
            "cubic-bezier(0.785, 0.135, 0.150, 0.860)",
            function (t, e, n, r) {
              return (t /= r / 2) < 1
                ? (-n / 2) * (Math.sqrt(1 - t * t) - 1) + e
                : (n / 2) * (Math.sqrt(1 - (t -= 2) * t) + 1) + e;
            },
          ],
          "ease-in-back": [
            "cubic-bezier(0.600, -0.280, 0.735, 0.045)",
            function (t, e, n, r, i) {
              return (
                void 0 === i && (i = 1.70158),
                n * (t /= r) * t * ((i + 1) * t - i) + e
              );
            },
          ],
          "ease-out-back": [
            "cubic-bezier(0.175, 0.885, 0.320, 1.275)",
            function (t, e, n, r, i) {
              return (
                void 0 === i && (i = 1.70158),
                n * ((t = t / r - 1) * t * ((i + 1) * t + i) + 1) + e
              );
            },
          ],
          "ease-in-out-back": [
            "cubic-bezier(0.680, -0.550, 0.265, 1.550)",
            function (t, e, n, r, i) {
              return (
                void 0 === i && (i = 1.70158),
                (t /= r / 2) < 1
                  ? (n / 2) * t * t * ((1 + (i *= 1.525)) * t - i) + e
                  : (n / 2) *
                      ((t -= 2) * t * ((1 + (i *= 1.525)) * t + i) + 2) +
                    e
              );
            },
          ],
        },
        d = {
          "ease-in-back": "cubic-bezier(0.600, 0, 0.735, 0.045)",
          "ease-out-back": "cubic-bezier(0.175, 0.885, 0.320, 1)",
          "ease-in-out-back": "cubic-bezier(0.680, 0, 0.265, 1)",
        },
        p = document,
        v = window,
        h = "bkwld-tram",
        E = /[\-\.0-9]/g,
        g = /[A-Z]/,
        _ = "number",
        y = /^(rgb|#)/,
        m = /(em|cm|mm|in|pt|pc|px)$/,
        I = /(em|cm|mm|in|pt|pc|px|%)$/,
        b = /(deg|rad|turn)$/,
        T = "unitless",
        O = /(all|none) 0s ease 0s/,
        w = /^(width|height)$/,
        A = " ",
        S = p.createElement("a"),
        x = ["Webkit", "Moz", "O", "ms"],
        R = ["-webkit-", "-moz-", "-o-", "-ms-"],
        C = function (t) {
          if (t in S.style) return { dom: t, css: t };
          var e,
            n,
            r = "",
            i = t.split("-");
          for (e = 0; e < i.length; e++)
            r += i[e].charAt(0).toUpperCase() + i[e].slice(1);
          for (e = 0; e < x.length; e++)
            if ((n = x[e] + r) in S.style) return { dom: n, css: R[e] + t };
        },
        N = (e.support = {
          bind: Function.prototype.bind,
          transform: C("transform"),
          transition: C("transition"),
          backface: C("backface-visibility"),
          timing: C("transition-timing-function"),
        });
      if (N.transition) {
        var L = N.timing.dom;
        if (((S.style[L] = l["ease-in-back"][0]), !S.style[L]))
          for (var D in d) l[D][0] = d[D];
      }
      var P = (e.frame = (function () {
          var t =
            v.requestAnimationFrame ||
            v.webkitRequestAnimationFrame ||
            v.mozRequestAnimationFrame ||
            v.oRequestAnimationFrame ||
            v.msRequestAnimationFrame;
          return t && N.bind
            ? t.bind(v)
            : function (t) {
                v.setTimeout(t, 16);
              };
        })()),
        M = (e.now = (function () {
          var t = v.performance,
            e = t && (t.now || t.webkitNow || t.msNow || t.mozNow);
          return e && N.bind
            ? e.bind(t)
            : Date.now ||
                function () {
                  return +new Date();
                };
        })()),
        j = f(function (e) {
          function i(t, e) {
            var n = (function (t) {
                for (var e = -1, n = t ? t.length : 0, r = []; ++e < n; ) {
                  var i = t[e];
                  i && r.push(i);
                }
                return r;
              })(("" + t).split(A)),
              r = n[0];
            e = e || {};
            var i = Q[r];
            if (!i) return s("Unsupported property: " + r);
            if (!e.weak || !this.props[r]) {
              var o = i[0],
                a = this.props[r];
              return (
                a || (a = this.props[r] = new o.Bare()),
                a.init(this.$el, n, i, e),
                a
              );
            }
          }
          function o(t, e, n) {
            if (t) {
              var o = (0, r.default)(t);
              if (
                (e ||
                  (this.timer && this.timer.destroy(),
                  (this.queue = []),
                  (this.active = !1)),
                "number" == o && e)
              )
                return (
                  (this.timer = new W({
                    duration: t,
                    context: this,
                    complete: a,
                  })),
                  void (this.active = !0)
                );
              if ("string" == o && e) {
                switch (t) {
                  case "hide":
                    f.call(this);
                    break;
                  case "stop":
                    u.call(this);
                    break;
                  case "redraw":
                    l.call(this);
                    break;
                  default:
                    i.call(this, t, n && n[1]);
                }
                return a.call(this);
              }
              if ("function" == o) return void t.call(this, this);
              if ("object" == o) {
                var s = 0;
                p.call(
                  this,
                  t,
                  function (t, e) {
                    t.span > s && (s = t.span), t.stop(), t.animate(e);
                  },
                  function (t) {
                    "wait" in t && (s = c(t.wait, 0));
                  }
                ),
                  d.call(this),
                  s > 0 &&
                    ((this.timer = new W({ duration: s, context: this })),
                    (this.active = !0),
                    e && (this.timer.complete = a));
                var v = this,
                  h = !1,
                  E = {};
                P(function () {
                  p.call(v, t, function (t) {
                    t.active && ((h = !0), (E[t.name] = t.nextStyle));
                  }),
                    h && v.$el.css(E);
                });
              }
            }
          }
          function a() {
            if (
              (this.timer && this.timer.destroy(),
              (this.active = !1),
              this.queue.length)
            ) {
              var t = this.queue.shift();
              o.call(this, t.options, !0, t.args);
            }
          }
          function u(t) {
            var e;
            this.timer && this.timer.destroy(),
              (this.queue = []),
              (this.active = !1),
              "string" == typeof t
                ? ((e = {})[t] = 1)
                : (e =
                    "object" == (0, r.default)(t) && null != t
                      ? t
                      : this.props),
              p.call(this, e, v),
              d.call(this);
          }
          function f() {
            u.call(this), (this.el.style.display = "none");
          }
          function l() {
            this.el.offsetHeight;
          }
          function d() {
            var t,
              e,
              n = [];
            for (t in (this.upstream && n.push(this.upstream), this.props))
              (e = this.props[t]).active && n.push(e.string);
            (n = n.join(",")),
              this.style !== n &&
                ((this.style = n), (this.el.style[N.transition.dom] = n));
          }
          function p(t, e, r) {
            var o,
              a,
              u,
              c,
              s = e !== v,
              f = {};
            for (o in t)
              (u = t[o]),
                o in q
                  ? (f.transform || (f.transform = {}), (f.transform[o] = u))
                  : (g.test(o) && (o = n(o)),
                    o in Q ? (f[o] = u) : (c || (c = {}), (c[o] = u)));
            for (o in f) {
              if (((u = f[o]), !(a = this.props[o]))) {
                if (!s) continue;
                a = i.call(this, o);
              }
              e.call(this, a, u);
            }
            r && c && r.call(this, c);
          }
          function v(t) {
            t.stop();
          }
          function E(t, e) {
            t.set(e);
          }
          function _(t) {
            this.$el.css(t);
          }
          function y(t, n) {
            e[t] = function () {
              return this.children
                ? function (t, e) {
                    var n,
                      r = this.children.length;
                    for (n = 0; r > n; n++) t.apply(this.children[n], e);
                    return this;
                  }.call(this, n, arguments)
                : (this.el && n.apply(this, arguments), this);
            };
          }
          (e.init = function (e) {
            if (
              ((this.$el = t(e)),
              (this.el = this.$el[0]),
              (this.props = {}),
              (this.queue = []),
              (this.style = ""),
              (this.active = !1),
              B.keepInherited && !B.fallback)
            ) {
              var n = Y(this.el, "transition");
              n && !O.test(n) && (this.upstream = n);
            }
            N.backface &&
              B.hideBackface &&
              z(this.el, N.backface.css, "hidden");
          }),
            y("add", i),
            y("start", o),
            y("wait", function (t) {
              (t = c(t, 0)),
                this.active
                  ? this.queue.push({ options: t })
                  : ((this.timer = new W({
                      duration: t,
                      context: this,
                      complete: a,
                    })),
                    (this.active = !0));
            }),
            y("then", function (t) {
              return this.active
                ? (this.queue.push({ options: t, args: arguments }),
                  void (this.timer.complete = a))
                : s(
                    "No active transition timer. Use start() or wait() before then()."
                  );
            }),
            y("next", a),
            y("stop", u),
            y("set", function (t) {
              u.call(this, t), p.call(this, t, E, _);
            }),
            y("show", function (t) {
              "string" != typeof t && (t = "block"),
                (this.el.style.display = t);
            }),
            y("hide", f),
            y("redraw", l),
            y("destroy", function () {
              u.call(this),
                t.removeData(this.el, h),
                (this.$el = this.el = null);
            });
        }),
        F = f(j, function (e) {
          function n(e, n) {
            var r = t.data(e, h) || t.data(e, h, new j.Bare());
            return r.el || r.init(e), n ? r.start(n) : r;
          }
          e.init = function (e, r) {
            var i = t(e);
            if (!i.length) return this;
            if (1 === i.length) return n(i[0], r);
            var o = [];
            return (
              i.each(function (t, e) {
                o.push(n(e, r));
              }),
              (this.children = o),
              this
            );
          };
        }),
        k = f(function (t) {
          function e() {
            var t = this.get();
            this.update("auto");
            var e = this.get();
            return this.update(t), e;
          }
          function n(t) {
            var e = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(t);
            return (e ? o(e[1], e[2], e[3]) : t).replace(
              /#(\w)(\w)(\w)$/,
              "#$1$1$2$2$3$3"
            );
          }
          var i = 500,
            a = "ease",
            u = 0;
          (t.init = function (t, e, n, r) {
            (this.$el = t), (this.el = t[0]);
            var o = e[0];
            n[2] && (o = n[2]),
              K[o] && (o = K[o]),
              (this.name = o),
              (this.type = n[1]),
              (this.duration = c(e[1], this.duration, i)),
              (this.ease = (function (t, e, n) {
                return void 0 !== e && (n = e), t in l ? t : n;
              })(e[2], this.ease, a)),
              (this.delay = c(e[3], this.delay, u)),
              (this.span = this.duration + this.delay),
              (this.active = !1),
              (this.nextStyle = null),
              (this.auto = w.test(this.name)),
              (this.unit = r.unit || this.unit || B.defaultUnit),
              (this.angle = r.angle || this.angle || B.defaultAngle),
              B.fallback || r.fallback
                ? (this.animate = this.fallback)
                : ((this.animate = this.transition),
                  (this.string =
                    this.name +
                    A +
                    this.duration +
                    "ms" +
                    ("ease" != this.ease ? A + l[this.ease][0] : "") +
                    (this.delay ? A + this.delay + "ms" : "")));
          }),
            (t.set = function (t) {
              (t = this.convert(t, this.type)), this.update(t), this.redraw();
            }),
            (t.transition = function (t) {
              (this.active = !0),
                (t = this.convert(t, this.type)),
                this.auto &&
                  ("auto" == this.el.style[this.name] &&
                    (this.update(this.get()), this.redraw()),
                  "auto" == t && (t = e.call(this))),
                (this.nextStyle = t);
            }),
            (t.fallback = function (t) {
              var n =
                this.el.style[this.name] || this.convert(this.get(), this.type);
              (t = this.convert(t, this.type)),
                this.auto &&
                  ("auto" == n && (n = this.convert(this.get(), this.type)),
                  "auto" == t && (t = e.call(this))),
                (this.tween = new V({
                  from: n,
                  to: t,
                  duration: this.duration,
                  delay: this.delay,
                  ease: this.ease,
                  update: this.update,
                  context: this,
                }));
            }),
            (t.get = function () {
              return Y(this.el, this.name);
            }),
            (t.update = function (t) {
              z(this.el, this.name, t);
            }),
            (t.stop = function () {
              (this.active || this.nextStyle) &&
                ((this.active = !1),
                (this.nextStyle = null),
                z(this.el, this.name, this.get()));
              var t = this.tween;
              t && t.context && t.destroy();
            }),
            (t.convert = function (t, e) {
              if ("auto" == t && this.auto) return t;
              var i,
                o = "number" == typeof t,
                a = "string" == typeof t;
              switch (e) {
                case _:
                  if (o) return t;
                  if (a && "" === t.replace(E, "")) return +t;
                  i = "number(unitless)";
                  break;
                case y:
                  if (a) {
                    if ("" === t && this.original) return this.original;
                    if (e.test(t))
                      return "#" == t.charAt(0) && 7 == t.length ? t : n(t);
                  }
                  i = "hex or rgb string";
                  break;
                case m:
                  if (o) return t + this.unit;
                  if (a && e.test(t)) return t;
                  i = "number(px) or string(unit)";
                  break;
                case I:
                  if (o) return t + this.unit;
                  if (a && e.test(t)) return t;
                  i = "number(px) or string(unit or %)";
                  break;
                case b:
                  if (o) return t + this.angle;
                  if (a && e.test(t)) return t;
                  i = "number(deg) or string(angle)";
                  break;
                case T:
                  if (o) return t;
                  if (a && I.test(t)) return t;
                  i = "number(unitless) or string(unit or %)";
              }
              return (
                (function (t, e) {
                  s(
                    "Type warning: Expected: [" +
                      t +
                      "] Got: [" +
                      (0, r.default)(e) +
                      "] " +
                      e
                  );
                })(i, t),
                t
              );
            }),
            (t.redraw = function () {
              this.el.offsetHeight;
            });
        }),
        G = f(k, function (t, e) {
          t.init = function () {
            e.init.apply(this, arguments),
              this.original || (this.original = this.convert(this.get(), y));
          };
        }),
        X = f(k, function (t, e) {
          (t.init = function () {
            e.init.apply(this, arguments), (this.animate = this.fallback);
          }),
            (t.get = function () {
              return this.$el[this.name]();
            }),
            (t.update = function (t) {
              this.$el[this.name](t);
            });
        }),
        U = f(k, function (t, e) {
          function n(t, e) {
            var n, r, i, o, a;
            for (n in t)
              (i = (o = q[n])[0]),
                (r = o[1] || n),
                (a = this.convert(t[n], i)),
                e.call(this, r, a, i);
          }
          (t.init = function () {
            e.init.apply(this, arguments),
              this.current ||
                ((this.current = {}),
                q.perspective &&
                  B.perspective &&
                  ((this.current.perspective = B.perspective),
                  z(this.el, this.name, this.style(this.current)),
                  this.redraw()));
          }),
            (t.set = function (t) {
              n.call(this, t, function (t, e) {
                this.current[t] = e;
              }),
                z(this.el, this.name, this.style(this.current)),
                this.redraw();
            }),
            (t.transition = function (t) {
              var e = this.values(t);
              this.tween = new H({
                current: this.current,
                values: e,
                duration: this.duration,
                delay: this.delay,
                ease: this.ease,
              });
              var n,
                r = {};
              for (n in this.current) r[n] = n in e ? e[n] : this.current[n];
              (this.active = !0), (this.nextStyle = this.style(r));
            }),
            (t.fallback = function (t) {
              var e = this.values(t);
              this.tween = new H({
                current: this.current,
                values: e,
                duration: this.duration,
                delay: this.delay,
                ease: this.ease,
                update: this.update,
                context: this,
              });
            }),
            (t.update = function () {
              z(this.el, this.name, this.style(this.current));
            }),
            (t.style = function (t) {
              var e,
                n = "";
              for (e in t) n += e + "(" + t[e] + ") ";
              return n;
            }),
            (t.values = function (t) {
              var e,
                r = {};
              return (
                n.call(this, t, function (t, n, i) {
                  (r[t] = n),
                    void 0 === this.current[t] &&
                      ((e = 0),
                      ~t.indexOf("scale") && (e = 1),
                      (this.current[t] = this.convert(e, i)));
                }),
                r
              );
            });
        }),
        V = f(function (e) {
          function n() {
            var t,
              e,
              r,
              i = c.length;
            if (i) for (P(n), e = M(), t = i; t--; ) (r = c[t]) && r.render(e);
          }
          var r = { ease: l.ease[1], from: 0, to: 1 };
          (e.init = function (t) {
            (this.duration = t.duration || 0), (this.delay = t.delay || 0);
            var e = t.ease || r.ease;
            l[e] && (e = l[e][1]),
              "function" != typeof e && (e = r.ease),
              (this.ease = e),
              (this.update = t.update || a),
              (this.complete = t.complete || a),
              (this.context = t.context || this),
              (this.name = t.name);
            var n = t.from,
              i = t.to;
            void 0 === n && (n = r.from),
              void 0 === i && (i = r.to),
              (this.unit = t.unit || ""),
              "number" == typeof n && "number" == typeof i
                ? ((this.begin = n), (this.change = i - n))
                : this.format(i, n),
              (this.value = this.begin + this.unit),
              (this.start = M()),
              !1 !== t.autoplay && this.play();
          }),
            (e.play = function () {
              var t;
              this.active ||
                (this.start || (this.start = M()),
                (this.active = !0),
                (t = this),
                1 === c.push(t) && P(n));
            }),
            (e.stop = function () {
              var e, n, r;
              this.active &&
                ((this.active = !1),
                (e = this),
                (r = t.inArray(e, c)) >= 0 &&
                  ((n = c.slice(r + 1)),
                  (c.length = r),
                  n.length && (c = c.concat(n))));
            }),
            (e.render = function (t) {
              var e,
                n = t - this.start;
              if (this.delay) {
                if (n <= this.delay) return;
                n -= this.delay;
              }
              if (n < this.duration) {
                var r = this.ease(n, 0, 1, this.duration);
                return (
                  (e = this.startRGB
                    ? (function (t, e, n) {
                        return o(
                          t[0] + n * (e[0] - t[0]),
                          t[1] + n * (e[1] - t[1]),
                          t[2] + n * (e[2] - t[2])
                        );
                      })(this.startRGB, this.endRGB, r)
                    : (function (t) {
                        return Math.round(t * s) / s;
                      })(this.begin + r * this.change)),
                  (this.value = e + this.unit),
                  void this.update.call(this.context, this.value)
                );
              }
              (e = this.endHex || this.begin + this.change),
                (this.value = e + this.unit),
                this.update.call(this.context, this.value),
                this.complete.call(this.context),
                this.destroy();
            }),
            (e.format = function (t, e) {
              if (((e += ""), "#" == (t += "").charAt(0)))
                return (
                  (this.startRGB = i(e)),
                  (this.endRGB = i(t)),
                  (this.endHex = t),
                  (this.begin = 0),
                  void (this.change = 1)
                );
              if (!this.unit) {
                var n = e.replace(E, "");
                n !== t.replace(E, "") && u("tween", e, t), (this.unit = n);
              }
              (e = parseFloat(e)),
                (t = parseFloat(t)),
                (this.begin = this.value = e),
                (this.change = t - e);
            }),
            (e.destroy = function () {
              this.stop(),
                (this.context = null),
                (this.ease = this.update = this.complete = a);
            });
          var c = [],
            s = 1e3;
        }),
        W = f(V, function (t) {
          (t.init = function (t) {
            (this.duration = t.duration || 0),
              (this.complete = t.complete || a),
              (this.context = t.context),
              this.play();
          }),
            (t.render = function (t) {
              t - this.start < this.duration ||
                (this.complete.call(this.context), this.destroy());
            });
        }),
        H = f(V, function (t, e) {
          (t.init = function (t) {
            var e, n;
            for (e in ((this.context = t.context),
            (this.update = t.update),
            (this.tweens = []),
            (this.current = t.current),
            t.values))
              (n = t.values[e]),
                this.current[e] !== n &&
                  this.tweens.push(
                    new V({
                      name: e,
                      from: this.current[e],
                      to: n,
                      duration: t.duration,
                      delay: t.delay,
                      ease: t.ease,
                      autoplay: !1,
                    })
                  );
            this.play();
          }),
            (t.render = function (t) {
              var e,
                n,
                r = !1;
              for (e = this.tweens.length; e--; )
                (n = this.tweens[e]).context &&
                  (n.render(t), (this.current[n.name] = n.value), (r = !0));
              return r
                ? void (this.update && this.update.call(this.context))
                : this.destroy();
            }),
            (t.destroy = function () {
              if ((e.destroy.call(this), this.tweens)) {
                var t;
                for (t = this.tweens.length; t--; ) this.tweens[t].destroy();
                (this.tweens = null), (this.current = null);
              }
            });
        }),
        B = (e.config = {
          debug: !1,
          defaultUnit: "px",
          defaultAngle: "deg",
          keepInherited: !1,
          hideBackface: !1,
          perspective: "",
          fallback: !N.transition,
          agentTests: [],
        });
      (e.fallback = function (t) {
        if (!N.transition) return (B.fallback = !0);
        B.agentTests.push("(" + t + ")");
        var e = new RegExp(B.agentTests.join("|"), "i");
        B.fallback = e.test(navigator.userAgent);
      }),
        e.fallback("6.0.[2-5] Safari"),
        (e.tween = function (t) {
          return new V(t);
        }),
        (e.delay = function (t, e, n) {
          return new W({ complete: e, duration: t, context: n });
        }),
        (t.fn.tram = function (t) {
          return e.call(null, this, t);
        });
      var z = t.style,
        Y = t.css,
        K = { transform: N.transform && N.transform.css },
        Q = {
          color: [G, y],
          background: [G, y, "background-color"],
          "outline-color": [G, y],
          "border-color": [G, y],
          "border-top-color": [G, y],
          "border-right-color": [G, y],
          "border-bottom-color": [G, y],
          "border-left-color": [G, y],
          "border-width": [k, m],
          "border-top-width": [k, m],
          "border-right-width": [k, m],
          "border-bottom-width": [k, m],
          "border-left-width": [k, m],
          "border-spacing": [k, m],
          "letter-spacing": [k, m],
          margin: [k, m],
          "margin-top": [k, m],
          "margin-right": [k, m],
          "margin-bottom": [k, m],
          "margin-left": [k, m],
          padding: [k, m],
          "padding-top": [k, m],
          "padding-right": [k, m],
          "padding-bottom": [k, m],
          "padding-left": [k, m],
          "outline-width": [k, m],
          opacity: [k, _],
          top: [k, I],
          right: [k, I],
          bottom: [k, I],
          left: [k, I],
          "font-size": [k, I],
          "text-indent": [k, I],
          "word-spacing": [k, I],
          width: [k, I],
          "min-width": [k, I],
          "max-width": [k, I],
          height: [k, I],
          "min-height": [k, I],
          "max-height": [k, I],
          "line-height": [k, T],
          "scroll-top": [X, _, "scrollTop"],
          "scroll-left": [X, _, "scrollLeft"],
        },
        q = {};
      N.transform &&
        ((Q.transform = [U]),
        (q = {
          x: [I, "translateX"],
          y: [I, "translateY"],
          rotate: [b],
          rotateX: [b],
          rotateY: [b],
          scale: [_],
          scaleX: [_],
          scaleY: [_],
          skew: [b],
          skewX: [b],
          skewY: [b],
        })),
        N.transform &&
          N.backface &&
          ((q.z = [I, "translateZ"]),
          (q.rotateZ = [b]),
          (q.scaleZ = [_]),
          (q.perspective = [m]));
      var $ = /ms/,
        Z = /s|\./;
      return (t.tram = e);
    })(window.jQuery);
  },
  function (t, e, n) {
    var r = n(15),
      i = n(131),
      o = n(67),
      a = n(37),
      u = n(68),
      c = n(17),
      s = n(69),
      f = Object.getOwnPropertyDescriptor;
    e.f = r
      ? f
      : function (t, e) {
          if (((t = a(t)), (e = u(e, !0)), s))
            try {
              return f(t, e);
            } catch (t) {}
          if (c(t, e)) return o(!i.f.call(t, e), t[e]);
        };
  },
  function (t, e) {
    t.exports = function (t, e) {
      return {
        enumerable: !(1 & t),
        configurable: !(2 & t),
        writable: !(4 & t),
        value: e,
      };
    };
  },
  function (t, e, n) {
    var r = n(24);
    t.exports = function (t, e) {
      if (!r(t)) return t;
      var n, i;
      if (e && "function" == typeof (n = t.toString) && !r((i = n.call(t))))
        return i;
      if ("function" == typeof (n = t.valueOf) && !r((i = n.call(t)))) return i;
      if (!e && "function" == typeof (n = t.toString) && !r((i = n.call(t))))
        return i;
      throw TypeError("Can't convert object to primitive value");
    };
  },
  function (t, e, n) {
    var r = n(15),
      i = n(16),
      o = n(70);
    t.exports =
      !r &&
      !i(function () {
        return (
          7 !=
          Object.defineProperty(o("div"), "a", {
            get: function () {
              return 7;
            },
          }).a
        );
      });
  },
  function (t, e, n) {
    var r = n(4),
      i = n(24),
      o = r.document,
      a = i(o) && i(o.createElement);
    t.exports = function (t) {
      return a ? o.createElement(t) : {};
    };
  },
  function (t, e, n) {
    var r = n(26);
    t.exports = r("native-function-to-string", Function.toString);
  },
  function (t, e, n) {
    var r = n(26),
      i = n(73),
      o = r("keys");
    t.exports = function (t) {
      return o[t] || (o[t] = i(t));
    };
  },
  function (t, e) {
    var n = 0,
      r = Math.random();
    t.exports = function (t) {
      return (
        "Symbol(" +
        String(void 0 === t ? "" : t) +
        ")_" +
        (++n + r).toString(36)
      );
    };
  },
  function (t, e, n) {
    var r = n(141),
      i = n(4),
      o = function (t) {
        return "function" == typeof t ? t : void 0;
      };
    t.exports = function (t, e) {
      return arguments.length < 2
        ? o(r[t]) || o(i[t])
        : (r[t] && r[t][e]) || (i[t] && i[t][e]);
    };
  },
  function (t, e, n) {
    var r = n(17),
      i = n(37),
      o = n(76).indexOf,
      a = n(40);
    t.exports = function (t, e) {
      var n,
        u = i(t),
        c = 0,
        s = [];
      for (n in u) !r(a, n) && r(u, n) && s.push(n);
      for (; e.length > c; ) r(u, (n = e[c++])) && (~o(s, n) || s.push(n));
      return s;
    };
  },
  function (t, e, n) {
    var r = n(37),
      i = n(143),
      o = n(144),
      a = function (t) {
        return function (e, n, a) {
          var u,
            c = r(e),
            s = i(c.length),
            f = o(a, s);
          if (t && n != n) {
            for (; s > f; ) if ((u = c[f++]) != u) return !0;
          } else
            for (; s > f; f++)
              if ((t || f in c) && c[f] === n) return t || f || 0;
          return !t && -1;
        };
      };
    t.exports = { includes: a(!0), indexOf: a(!1) };
  },
  function (t, e) {
    var n = Math.ceil,
      r = Math.floor;
    t.exports = function (t) {
      return isNaN((t = +t)) ? 0 : (t > 0 ? r : n)(t);
    };
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(42);
    n.d(e, "createStore", function () {
      return r.default;
    });
    var i = n(81);
    n.d(e, "combineReducers", function () {
      return i.default;
    });
    var o = n(83);
    n.d(e, "bindActionCreators", function () {
      return o.default;
    });
    var a = n(84);
    n.d(e, "applyMiddleware", function () {
      return a.default;
    });
    var u = n(43);
    n.d(e, "compose", function () {
      return u.default;
    });
    n(82);
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(157),
      i = n(162),
      o = n(164),
      a = "[object Object]",
      u = Function.prototype,
      c = Object.prototype,
      s = u.toString,
      f = c.hasOwnProperty,
      l = s.call(Object);
    e.default = function (t) {
      if (!Object(o.default)(t) || Object(r.default)(t) != a) return !1;
      var e = Object(i.default)(t);
      if (null === e) return !0;
      var n = f.call(e, "constructor") && e.constructor;
      return "function" == typeof n && n instanceof n && s.call(n) == l;
    };
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(158).default.Symbol;
    e.default = r;
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      n.d(e, "default", function () {
        return o;
      });
    var r = n(42);
    n(79), n(82);
    function i(t, e) {
      var n = e && e.type;
      return (
        "Given action " +
        ((n && '"' + n.toString() + '"') || "an action") +
        ', reducer "' +
        t +
        '" returned undefined. To ignore an action, you must explicitly return the previous state.'
      );
    }
    function o(t) {
      for (var e = Object.keys(t), n = {}, o = 0; o < e.length; o++) {
        var a = e[o];
        0, "function" == typeof t[a] && (n[a] = t[a]);
      }
      var u,
        c = Object.keys(n);
      try {
        !(function (t) {
          Object.keys(t).forEach(function (e) {
            var n = t[e];
            if (void 0 === n(void 0, { type: r.ActionTypes.INIT }))
              throw new Error(
                'Reducer "' +
                  e +
                  '" returned undefined during initialization. If the state passed to the reducer is undefined, you must explicitly return the initial state. The initial state may not be undefined.'
              );
            if (
              void 0 ===
              n(void 0, {
                type:
                  "@@redux/PROBE_UNKNOWN_ACTION_" +
                  Math.random().toString(36).substring(7).split("").join("."),
              })
            )
              throw new Error(
                'Reducer "' +
                  e +
                  "\" returned undefined when probed with a random type. Don't try to handle " +
                  r.ActionTypes.INIT +
                  ' or other actions in "redux/*" namespace. They are considered private. Instead, you must return the current state for any unknown actions, unless it is undefined, in which case you must return the initial state, regardless of the action type. The initial state may not be undefined.'
              );
          });
        })(n);
      } catch (t) {
        u = t;
      }
      return function () {
        var t =
            arguments.length <= 0 || void 0 === arguments[0]
              ? {}
              : arguments[0],
          e = arguments[1];
        if (u) throw u;
        for (var r = !1, o = {}, a = 0; a < c.length; a++) {
          var s = c[a],
            f = n[s],
            l = t[s],
            d = f(l, e);
          if (void 0 === d) {
            var p = i(s, e);
            throw new Error(p);
          }
          (o[s] = d), (r = r || d !== l);
        }
        return r ? o : t;
      };
    }
  },
  function (t, e, n) {
    "use strict";
    function r(t) {
      "undefined" != typeof console &&
        "function" == typeof console.error &&
        console.error(t);
      try {
        throw new Error(t);
      } catch (t) {}
    }
    n.r(e),
      n.d(e, "default", function () {
        return r;
      });
  },
  function (t, e, n) {
    "use strict";
    function r(t, e) {
      return function () {
        return e(t.apply(void 0, arguments));
      };
    }
    function i(t, e) {
      if ("function" == typeof t) return r(t, e);
      if ("object" != typeof t || null === t)
        throw new Error(
          "bindActionCreators expected an object or a function, instead received " +
            (null === t ? "null" : typeof t) +
            '. Did you write "import ActionCreators from" instead of "import * as ActionCreators from"?'
        );
      for (var n = Object.keys(t), i = {}, o = 0; o < n.length; o++) {
        var a = n[o],
          u = t[a];
        "function" == typeof u && (i[a] = r(u, e));
      }
      return i;
    }
    n.r(e),
      n.d(e, "default", function () {
        return i;
      });
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      n.d(e, "default", function () {
        return o;
      });
    var r = n(43),
      i =
        Object.assign ||
        function (t) {
          for (var e = 1; e < arguments.length; e++) {
            var n = arguments[e];
            for (var r in n)
              Object.prototype.hasOwnProperty.call(n, r) && (t[r] = n[r]);
          }
          return t;
        };
    function o() {
      for (var t = arguments.length, e = Array(t), n = 0; n < t; n++)
        e[n] = arguments[n];
      return function (t) {
        return function (n, o, a) {
          var u,
            c = t(n, o, a),
            s = c.dispatch,
            f = {
              getState: c.getState,
              dispatch: function (t) {
                return s(t);
              },
            };
          return (
            (u = e.map(function (t) {
              return t(f);
            })),
            (s = r.default.apply(void 0, u)(c.dispatch)),
            i({}, c, { dispatch: s })
          );
        };
      };
    }
  },
  function (t, e, n) {
    var r = n(86)(n(242));
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(7),
      i = n(12),
      o = n(33);
    t.exports = function (t) {
      return function (e, n, a) {
        var u = Object(e);
        if (!i(e)) {
          var c = r(n, 3);
          (e = o(e)),
            (n = function (t) {
              return c(u[t], t, u);
            });
        }
        var s = t(e, n, a);
        return s > -1 ? u[c ? e[s] : s] : void 0;
      };
    };
  },
  function (t, e, n) {
    var r = n(29),
      i = n(184),
      o = n(185),
      a = n(186),
      u = n(187),
      c = n(188);
    function s(t) {
      var e = (this.__data__ = new r(t));
      this.size = e.size;
    }
    (s.prototype.clear = i),
      (s.prototype.delete = o),
      (s.prototype.get = a),
      (s.prototype.has = u),
      (s.prototype.set = c),
      (t.exports = s);
  },
  function (t, e, n) {
    var r = n(11),
      i = n(6),
      o = "[object AsyncFunction]",
      a = "[object Function]",
      u = "[object GeneratorFunction]",
      c = "[object Proxy]";
    t.exports = function (t) {
      if (!i(t)) return !1;
      var e = r(t);
      return e == a || e == u || e == o || e == c;
    };
  },
  function (t, e, n) {
    (function (e) {
      var n = "object" == typeof e && e && e.Object === Object && e;
      t.exports = n;
    }).call(this, n(23));
  },
  function (t, e) {
    var n = Function.prototype.toString;
    t.exports = function (t) {
      if (null != t) {
        try {
          return n.call(t);
        } catch (t) {}
        try {
          return t + "";
        } catch (t) {}
      }
      return "";
    };
  },
  function (t, e, n) {
    var r = n(207),
      i = n(9);
    t.exports = function t(e, n, o, a, u) {
      return (
        e === n ||
        (null == e || null == n || (!i(e) && !i(n))
          ? e != e && n != n
          : r(e, n, o, a, t, u))
      );
    };
  },
  function (t, e, n) {
    var r = n(208),
      i = n(211),
      o = n(212),
      a = 1,
      u = 2;
    t.exports = function (t, e, n, c, s, f) {
      var l = n & a,
        d = t.length,
        p = e.length;
      if (d != p && !(l && p > d)) return !1;
      var v = f.get(t),
        h = f.get(e);
      if (v && h) return v == e && h == t;
      var E = -1,
        g = !0,
        _ = n & u ? new r() : void 0;
      for (f.set(t, e), f.set(e, t); ++E < d; ) {
        var y = t[E],
          m = e[E];
        if (c) var I = l ? c(m, y, E, e, t, f) : c(y, m, E, t, e, f);
        if (void 0 !== I) {
          if (I) continue;
          g = !1;
          break;
        }
        if (_) {
          if (
            !i(e, function (t, e) {
              if (!o(_, e) && (y === t || s(y, t, n, c, f))) return _.push(e);
            })
          ) {
            g = !1;
            break;
          }
        } else if (y !== m && !s(y, m, n, c, f)) {
          g = !1;
          break;
        }
      }
      return f.delete(t), f.delete(e), g;
    };
  },
  function (t, e, n) {
    var r = n(48),
      i = n(1);
    t.exports = function (t, e, n) {
      var o = e(t);
      return i(t) ? o : r(o, n(t));
    };
  },
  function (t, e, n) {
    var r = n(219),
      i = n(95),
      o = Object.prototype.propertyIsEnumerable,
      a = Object.getOwnPropertySymbols,
      u = a
        ? function (t) {
            return null == t
              ? []
              : ((t = Object(t)),
                r(a(t), function (e) {
                  return o.call(t, e);
                }));
          }
        : i;
    t.exports = u;
  },
  function (t, e) {
    t.exports = function () {
      return [];
    };
  },
  function (t, e, n) {
    var r = n(220),
      i = n(34),
      o = n(1),
      a = n(49),
      u = n(50),
      c = n(51),
      s = Object.prototype.hasOwnProperty;
    t.exports = function (t, e) {
      var n = o(t),
        f = !n && i(t),
        l = !n && !f && a(t),
        d = !n && !f && !l && c(t),
        p = n || f || l || d,
        v = p ? r(t.length, String) : [],
        h = v.length;
      for (var E in t)
        (!e && !s.call(t, E)) ||
          (p &&
            ("length" == E ||
              (l && ("offset" == E || "parent" == E)) ||
              (d &&
                ("buffer" == E || "byteLength" == E || "byteOffset" == E)) ||
              u(E, h))) ||
          v.push(E);
      return v;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return (
        t.webpackPolyfill ||
          ((t.deprecate = function () {}),
          (t.paths = []),
          t.children || (t.children = []),
          Object.defineProperty(t, "loaded", {
            enumerable: !0,
            get: function () {
              return t.l;
            },
          }),
          Object.defineProperty(t, "id", {
            enumerable: !0,
            get: function () {
              return t.i;
            },
          }),
          (t.webpackPolyfill = 1)),
        t
      );
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      return function (n) {
        return t(e(n));
      };
    };
  },
  function (t, e, n) {
    var r = n(8)(n(5), "WeakMap");
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(6);
    t.exports = function (t) {
      return t == t && !r(t);
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      return function (n) {
        return null != n && n[t] === e && (void 0 !== e || t in Object(n));
      };
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      for (var n = -1, r = null == t ? 0 : t.length, i = Array(r); ++n < r; )
        i[n] = e(t[n], n, t);
      return i;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return function (e) {
        return null == e ? void 0 : e[t];
      };
    };
  },
  function (t, e) {
    t.exports = function (t, e, n, r) {
      for (var i = t.length, o = n + (r ? 1 : -1); r ? o-- : ++o < i; )
        if (e(t[o], o, t)) return o;
      return -1;
    };
  },
  function (t, e, n) {
    var r = n(243);
    t.exports = function (t) {
      var e = r(t),
        n = e % 1;
      return e == e ? (n ? e - n : e) : 0;
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.inQuad = function (t) {
        return Math.pow(t, 2);
      }),
      (e.outQuad = function (t) {
        return -(Math.pow(t - 1, 2) - 1);
      }),
      (e.inOutQuad = function (t) {
        if ((t /= 0.5) < 1) return 0.5 * Math.pow(t, 2);
        return -0.5 * ((t -= 2) * t - 2);
      }),
      (e.inCubic = function (t) {
        return Math.pow(t, 3);
      }),
      (e.outCubic = function (t) {
        return Math.pow(t - 1, 3) + 1;
      }),
      (e.inOutCubic = function (t) {
        if ((t /= 0.5) < 1) return 0.5 * Math.pow(t, 3);
        return 0.5 * (Math.pow(t - 2, 3) + 2);
      }),
      (e.inQuart = function (t) {
        return Math.pow(t, 4);
      }),
      (e.outQuart = function (t) {
        return -(Math.pow(t - 1, 4) - 1);
      }),
      (e.inOutQuart = function (t) {
        if ((t /= 0.5) < 1) return 0.5 * Math.pow(t, 4);
        return -0.5 * ((t -= 2) * Math.pow(t, 3) - 2);
      }),
      (e.inQuint = function (t) {
        return Math.pow(t, 5);
      }),
      (e.outQuint = function (t) {
        return Math.pow(t - 1, 5) + 1;
      }),
      (e.inOutQuint = function (t) {
        if ((t /= 0.5) < 1) return 0.5 * Math.pow(t, 5);
        return 0.5 * (Math.pow(t - 2, 5) + 2);
      }),
      (e.inSine = function (t) {
        return 1 - Math.cos(t * (Math.PI / 2));
      }),
      (e.outSine = function (t) {
        return Math.sin(t * (Math.PI / 2));
      }),
      (e.inOutSine = function (t) {
        return -0.5 * (Math.cos(Math.PI * t) - 1);
      }),
      (e.inExpo = function (t) {
        return 0 === t ? 0 : Math.pow(2, 10 * (t - 1));
      }),
      (e.outExpo = function (t) {
        return 1 === t ? 1 : 1 - Math.pow(2, -10 * t);
      }),
      (e.inOutExpo = function (t) {
        if (0 === t) return 0;
        if (1 === t) return 1;
        if ((t /= 0.5) < 1) return 0.5 * Math.pow(2, 10 * (t - 1));
        return 0.5 * (2 - Math.pow(2, -10 * --t));
      }),
      (e.inCirc = function (t) {
        return -(Math.sqrt(1 - t * t) - 1);
      }),
      (e.outCirc = function (t) {
        return Math.sqrt(1 - Math.pow(t - 1, 2));
      }),
      (e.inOutCirc = function (t) {
        if ((t /= 0.5) < 1) return -0.5 * (Math.sqrt(1 - t * t) - 1);
        return 0.5 * (Math.sqrt(1 - (t -= 2) * t) + 1);
      }),
      (e.outBounce = function (t) {
        return t < 1 / 2.75
          ? 7.5625 * t * t
          : t < 2 / 2.75
          ? 7.5625 * (t -= 1.5 / 2.75) * t + 0.75
          : t < 2.5 / 2.75
          ? 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375
          : 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375;
      }),
      (e.inBack = function (t) {
        return t * t * ((o + 1) * t - o);
      }),
      (e.outBack = function (t) {
        return (t -= 1) * t * ((o + 1) * t + o) + 1;
      }),
      (e.inOutBack = function (t) {
        var e = o;
        if ((t /= 0.5) < 1) return t * t * ((1 + (e *= 1.525)) * t - e) * 0.5;
        return 0.5 * ((t -= 2) * t * ((1 + (e *= 1.525)) * t + e) + 2);
      }),
      (e.inElastic = function (t) {
        var e = o,
          n = 0,
          r = 1;
        if (0 === t) return 0;
        if (1 === t) return 1;
        n || (n = 0.3);
        r < 1
          ? ((r = 1), (e = n / 4))
          : (e = (n / (2 * Math.PI)) * Math.asin(1 / r));
        return (
          -r *
          Math.pow(2, 10 * (t -= 1)) *
          Math.sin(((t - e) * (2 * Math.PI)) / n)
        );
      }),
      (e.outElastic = function (t) {
        var e = o,
          n = 0,
          r = 1;
        if (0 === t) return 0;
        if (1 === t) return 1;
        n || (n = 0.3);
        r < 1
          ? ((r = 1), (e = n / 4))
          : (e = (n / (2 * Math.PI)) * Math.asin(1 / r));
        return (
          r * Math.pow(2, -10 * t) * Math.sin(((t - e) * (2 * Math.PI)) / n) + 1
        );
      }),
      (e.inOutElastic = function (t) {
        var e = o,
          n = 0,
          r = 1;
        if (0 === t) return 0;
        if (2 == (t /= 0.5)) return 1;
        n || (n = 0.3 * 1.5);
        r < 1
          ? ((r = 1), (e = n / 4))
          : (e = (n / (2 * Math.PI)) * Math.asin(1 / r));
        if (t < 1)
          return (
            r *
            Math.pow(2, 10 * (t -= 1)) *
            Math.sin(((t - e) * (2 * Math.PI)) / n) *
            -0.5
          );
        return (
          r *
            Math.pow(2, -10 * (t -= 1)) *
            Math.sin(((t - e) * (2 * Math.PI)) / n) *
            0.5 +
          1
        );
      }),
      (e.swingFromTo = function (t) {
        var e = o;
        return (t /= 0.5) < 1
          ? t * t * ((1 + (e *= 1.525)) * t - e) * 0.5
          : 0.5 * ((t -= 2) * t * ((1 + (e *= 1.525)) * t + e) + 2);
      }),
      (e.swingFrom = function (t) {
        return t * t * ((o + 1) * t - o);
      }),
      (e.swingTo = function (t) {
        return (t -= 1) * t * ((o + 1) * t + o) + 1;
      }),
      (e.bounce = function (t) {
        return t < 1 / 2.75
          ? 7.5625 * t * t
          : t < 2 / 2.75
          ? 7.5625 * (t -= 1.5 / 2.75) * t + 0.75
          : t < 2.5 / 2.75
          ? 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375
          : 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375;
      }),
      (e.bouncePast = function (t) {
        return t < 1 / 2.75
          ? 7.5625 * t * t
          : t < 2 / 2.75
          ? 2 - (7.5625 * (t -= 1.5 / 2.75) * t + 0.75)
          : t < 2.5 / 2.75
          ? 2 - (7.5625 * (t -= 2.25 / 2.75) * t + 0.9375)
          : 2 - (7.5625 * (t -= 2.625 / 2.75) * t + 0.984375);
      }),
      (e.easeInOut = e.easeOut = e.easeIn = e.ease = void 0);
    var i = r(n(107)),
      o = 1.70158,
      a = (0, i.default)(0.25, 0.1, 0.25, 1);
    e.ease = a;
    var u = (0, i.default)(0.42, 0, 1, 1);
    e.easeIn = u;
    var c = (0, i.default)(0, 0, 0.58, 1);
    e.easeOut = c;
    var s = (0, i.default)(0.42, 0, 0.58, 1);
    e.easeInOut = s;
  },
  function (t, e) {
    var n = 4,
      r = 0.001,
      i = 1e-7,
      o = 10,
      a = 11,
      u = 1 / (a - 1),
      c = "function" == typeof Float32Array;
    function s(t, e) {
      return 1 - 3 * e + 3 * t;
    }
    function f(t, e) {
      return 3 * e - 6 * t;
    }
    function l(t) {
      return 3 * t;
    }
    function d(t, e, n) {
      return ((s(e, n) * t + f(e, n)) * t + l(e)) * t;
    }
    function p(t, e, n) {
      return 3 * s(e, n) * t * t + 2 * f(e, n) * t + l(e);
    }
    t.exports = function (t, e, s, f) {
      if (!(0 <= t && t <= 1 && 0 <= s && s <= 1))
        throw new Error("bezier x values must be in [0, 1] range");
      var l = c ? new Float32Array(a) : new Array(a);
      if (t !== e || s !== f) for (var v = 0; v < a; ++v) l[v] = d(v * u, t, s);
      function h(e) {
        for (var c = 0, f = 1, v = a - 1; f !== v && l[f] <= e; ++f) c += u;
        var h = c + ((e - l[--f]) / (l[f + 1] - l[f])) * u,
          E = p(h, t, s);
        return E >= r
          ? (function (t, e, r, i) {
              for (var o = 0; o < n; ++o) {
                var a = p(e, r, i);
                if (0 === a) return e;
                e -= (d(e, r, i) - t) / a;
              }
              return e;
            })(e, h, t, s)
          : 0 === E
          ? h
          : (function (t, e, n, r, a) {
              var u,
                c,
                s = 0;
              do {
                (u = d((c = e + (n - e) / 2), r, a) - t) > 0
                  ? (n = c)
                  : (e = c);
              } while (Math.abs(u) > i && ++s < o);
              return c;
            })(e, c, c + u, t, s);
      }
      return function (n) {
        return t === e && s === f
          ? n
          : 0 === n
          ? 0
          : 1 === n
          ? 1
          : d(h(n), e, f);
      };
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(0)(n(109)),
      i = n(0),
      o = n(14);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.optimizeFloat = c),
      (e.createBezierEasing = function (t) {
        return u.default.apply(void 0, (0, r.default)(t));
      }),
      (e.applyEasing = function (t, e, n) {
        if (0 === e) return 0;
        if (1 === e) return 1;
        if (n) return c(e > 0 ? n(e) : e);
        return c(e > 0 && t && a[t] ? a[t](e) : e);
      });
    var a = o(n(106)),
      u = i(n(107));
    function c(t) {
      var e =
          arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : 5,
        n = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : 10,
        r = Math.pow(n, e),
        i = Number(Math.round(t * r) / r);
      return Math.abs(i) > 1e-4 ? i : 0;
    }
  },
  function (t, e, n) {
    var r = n(244),
      i = n(245),
      o = n(246);
    t.exports = function (t) {
      return r(t) || i(t) || o();
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(0)(n(27));
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.isPluginType = function (t) {
        return t === o.ActionTypeConsts.PLUGIN_LOTTIE;
      }),
      (e.clearPlugin =
        e.renderPlugin =
        e.createPluginInstance =
        e.getPluginDestination =
        e.getPluginDuration =
        e.getPluginOrigin =
        e.getPluginConfig =
          void 0);
    var i = n(248),
      o = n(2),
      a = n(44),
      u = (0, r.default)({}, o.ActionTypeConsts.PLUGIN_LOTTIE, {
        getConfig: i.getPluginConfig,
        getOrigin: i.getPluginOrigin,
        getDuration: i.getPluginDuration,
        getDestination: i.getPluginDestination,
        createInstance: i.createPluginInstance,
        render: i.renderPlugin,
        clear: i.clearPlugin,
      });
    var c = function (t) {
        return function (e) {
          if (!a.IS_BROWSER_ENV)
            return function () {
              return null;
            };
          var n = u[e];
          if (!n) throw new Error("IX2 no plugin configured for: ".concat(e));
          var r = n[t];
          if (!r) throw new Error("IX2 invalid plugin method: ".concat(t));
          return r;
        };
      },
      s = c("getConfig");
    e.getPluginConfig = s;
    var f = c("getOrigin");
    e.getPluginOrigin = f;
    var l = c("getDuration");
    e.getPluginDuration = l;
    var d = c("getDestination");
    e.getPluginDestination = d;
    var p = c("createInstance");
    e.createPluginInstance = p;
    var v = c("render");
    e.renderPlugin = v;
    var h = c("clear");
    e.clearPlugin = h;
  },
  function (t, e, n) {
    var r = n(112),
      i = n(255)(r);
    t.exports = i;
  },
  function (t, e, n) {
    var r = n(253),
      i = n(33);
    t.exports = function (t, e) {
      return t && r(t, e, i);
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(259);
    (e.__esModule = !0), (e.default = void 0);
    var i = r(n(260)).default;
    e.default = i;
  },
  function (t, e, n) {
    "use strict";
    var r = n(0)(n(109)),
      i = n(14),
      o = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.observeRequests = function (t) {
        P({
          store: t,
          select: function (t) {
            var e = t.ixRequest;
            return e.preview;
          },
          onChange: et,
        }),
          P({
            store: t,
            select: function (t) {
              var e = t.ixRequest;
              return e.playback;
            },
            onChange: rt,
          }),
          P({
            store: t,
            select: function (t) {
              var e = t.ixRequest;
              return e.stop;
            },
            onChange: it,
          }),
          P({
            store: t,
            select: function (t) {
              var e = t.ixRequest;
              return e.clear;
            },
            onChange: ot,
          });
      }),
      (e.startEngine = at),
      (e.stopEngine = ut),
      (e.stopAllActionGroups = ht),
      (e.stopActionGroup = Et),
      (e.startActionGroup = gt);
    var a = o(n(28)),
      u = o(n(263)),
      c = o(n(85)),
      s = o(n(56)),
      f = o(n(264)),
      l = o(n(270)),
      d = o(n(282)),
      p = o(n(283)),
      v = o(n(284)),
      h = o(n(287)),
      E = o(n(113)),
      g = n(2),
      _ = n(10),
      y = n(61),
      m = i(n(290)),
      I = o(n(291)),
      b = Object.keys(g.QuickEffectIds),
      T = function (t) {
        return b.includes(t);
      },
      O = g.IX2EngineConstants,
      w = O.COLON_DELIMITER,
      A = O.BOUNDARY_SELECTOR,
      S = O.HTML_ELEMENT,
      x = O.RENDER_GENERAL,
      R = O.W_MOD_IX,
      C = _.IX2VanillaUtils,
      N = C.getAffectedElements,
      L = C.getElementId,
      D = C.getDestinationValues,
      P = C.observeStore,
      M = C.getInstanceId,
      j = C.renderHTMLElement,
      F = C.clearAllStyles,
      k = C.getMaxDurationItemIndex,
      G = C.getComputedStyle,
      X = C.getInstanceOrigin,
      U = C.reduceListToGroup,
      V = C.shouldNamespaceEventParameter,
      W = C.getNamespacedParameterId,
      H = C.shouldAllowMediaQuery,
      B = C.cleanupHTMLElement,
      z = C.stringifyTarget,
      Y = C.mediaQueriesEqual,
      K = _.IX2VanillaPlugins,
      Q = K.isPluginType,
      q = K.createPluginInstance,
      $ = K.getPluginDuration,
      Z = navigator.userAgent,
      J = Z.match(/iPad/i) || Z.match(/iPhone/),
      tt = 12;
    function et(t, e) {
      var n = t.rawData,
        r = function () {
          at({ store: e, rawData: n, allowEvents: !0 }), nt();
        };
      t.defer ? setTimeout(r, 0) : r();
    }
    function nt() {
      document.dispatchEvent(new CustomEvent("IX2_PAGE_UPDATE"));
    }
    function rt(t, e) {
      var n = t.actionTypeId,
        r = t.actionListId,
        i = t.actionItemId,
        o = t.eventId,
        a = t.allowEvents,
        u = t.immediate,
        c = t.testManual,
        s = t.verbose,
        f = void 0 === s || s,
        l = t.rawData;
      if (r && i && l && u) {
        var d = l.actionLists[r];
        d && (l = U({ actionList: d, actionItemId: i, rawData: l }));
      }
      if (
        (at({ store: e, rawData: l, allowEvents: a, testManual: c }),
        (r && n === g.ActionTypeConsts.GENERAL_START_ACTION) || T(n))
      ) {
        Et({ store: e, actionListId: r }),
          vt({ store: e, actionListId: r, eventId: o });
        var p = gt({
          store: e,
          eventId: o,
          actionListId: r,
          immediate: u,
          verbose: f,
        });
        f &&
          p &&
          e.dispatch(
            (0, y.actionListPlaybackChanged)({ actionListId: r, isPlaying: !u })
          );
      }
    }
    function it(t, e) {
      var n = t.actionListId;
      n ? Et({ store: e, actionListId: n }) : ht({ store: e }), ut(e);
    }
    function ot(t, e) {
      ut(e), F({ store: e, elementApi: m });
    }
    function at(t) {
      var e,
        n = t.store,
        i = t.rawData,
        o = t.allowEvents,
        a = t.testManual,
        u = n.getState().ixSession;
      i && n.dispatch((0, y.rawDataImported)(i)),
        u.active ||
          (n.dispatch(
            (0, y.sessionInitialized)({
              hasBoundaryNodes: Boolean(document.querySelector(A)),
            })
          ),
          o &&
            ((function (t) {
              var e = t.getState().ixData.eventTypeMap;
              ft(t),
                (0, v.default)(e, function (e, n) {
                  var i = I.default[n];
                  i
                    ? (function (t) {
                        var e = t.logic,
                          n = t.store,
                          i = t.events;
                        !(function (t) {
                          if (J) {
                            var e = {},
                              n = "";
                            for (var r in t) {
                              var i = t[r],
                                o = i.eventTypeId,
                                a = i.target,
                                u = m.getQuerySelector(a);
                              e[u] ||
                                (o !== g.EventTypeConsts.MOUSE_CLICK &&
                                  o !== g.EventTypeConsts.MOUSE_SECOND_CLICK) ||
                                ((e[u] = !0),
                                (n +=
                                  u +
                                  "{cursor: pointer;touch-action: manipulation;}"));
                            }
                            if (n) {
                              var c = document.createElement("style");
                              (c.textContent = n), document.body.appendChild(c);
                            }
                          }
                        })(i);
                        var o = e.types,
                          a = e.handler,
                          u = n.getState().ixData,
                          l = u.actionLists,
                          d = lt(i, pt);
                        if ((0, f.default)(d)) {
                          (0, v.default)(d, function (t, e) {
                            var o = i[e],
                              a = o.action,
                              f = o.id,
                              d = o.mediaQueries,
                              p = void 0 === d ? u.mediaQueryKeys : d,
                              v = a.config.actionListId;
                            if (
                              (Y(p, u.mediaQueryKeys) ||
                                n.dispatch((0, y.mediaQueriesDefined)()),
                              a.actionTypeId ===
                                g.ActionTypeConsts.GENERAL_CONTINUOUS_ACTION)
                            ) {
                              var h = Array.isArray(o.config)
                                ? o.config
                                : [o.config];
                              h.forEach(function (e) {
                                var i = e.continuousParameterGroupId,
                                  o = (0, s.default)(
                                    l,
                                    "".concat(v, ".continuousParameterGroups"),
                                    []
                                  ),
                                  a = (0, c.default)(o, function (t) {
                                    var e = t.id;
                                    return e === i;
                                  }),
                                  u = (e.smoothing || 0) / 100,
                                  d = (e.restingState || 0) / 100;
                                a &&
                                  t.forEach(function (t, i) {
                                    var o = f + w + i;
                                    !(function (t) {
                                      var e = t.store,
                                        n = t.eventStateKey,
                                        i = t.eventTarget,
                                        o = t.eventId,
                                        a = t.eventConfig,
                                        u = t.actionListId,
                                        c = t.parameterGroup,
                                        f = t.smoothing,
                                        l = t.restingValue,
                                        d = e.getState(),
                                        p = d.ixData,
                                        v = d.ixSession,
                                        h = p.events[o],
                                        E = h.eventTypeId,
                                        g = {},
                                        _ = {},
                                        y = [],
                                        I = c.continuousActionGroups,
                                        b = c.id;
                                      V(E, a) && (b = W(n, b));
                                      var T =
                                        v.hasBoundaryNodes && i
                                          ? m.getClosestElement(i, A)
                                          : null;
                                      I.forEach(function (t) {
                                        var e = t.keyframe,
                                          n = t.actionItems;
                                        n.forEach(function (t) {
                                          var n = t.actionTypeId,
                                            o = t.config.target;
                                          if (o) {
                                            var a = o.boundaryMode ? T : null,
                                              u = z(o) + w + n;
                                            if (
                                              ((_[u] = (function () {
                                                var t,
                                                  e =
                                                    arguments.length > 0 &&
                                                    void 0 !== arguments[0]
                                                      ? arguments[0]
                                                      : [],
                                                  n =
                                                    arguments.length > 1
                                                      ? arguments[1]
                                                      : void 0,
                                                  i =
                                                    arguments.length > 2
                                                      ? arguments[2]
                                                      : void 0,
                                                  o = (0, r.default)(e);
                                                return (
                                                  o.some(function (e, r) {
                                                    return (
                                                      e.keyframe === n &&
                                                      ((t = r), !0)
                                                    );
                                                  }),
                                                  null == t &&
                                                    ((t = o.length),
                                                    o.push({
                                                      keyframe: n,
                                                      actionItems: [],
                                                    })),
                                                  o[t].actionItems.push(i),
                                                  o
                                                );
                                              })(_[u], e, t)),
                                              !g[u])
                                            ) {
                                              g[u] = !0;
                                              var c = t.config;
                                              N({
                                                config: c,
                                                event: h,
                                                eventTarget: i,
                                                elementRoot: a,
                                                elementApi: m,
                                              }).forEach(function (t) {
                                                y.push({ element: t, key: u });
                                              });
                                            }
                                          }
                                        });
                                      }),
                                        y.forEach(function (t) {
                                          var n = t.element,
                                            r = t.key,
                                            i = _[r],
                                            a = (0, s.default)(
                                              i,
                                              "[0].actionItems[0]",
                                              {}
                                            ),
                                            c = a.actionTypeId,
                                            d = Q(c) ? q(c)(n, a) : null,
                                            p = D(
                                              {
                                                element: n,
                                                actionItem: a,
                                                elementApi: m,
                                              },
                                              d
                                            );
                                          _t({
                                            store: e,
                                            element: n,
                                            eventId: o,
                                            actionListId: u,
                                            actionItem: a,
                                            destination: p,
                                            continuous: !0,
                                            parameterId: b,
                                            actionGroups: i,
                                            smoothing: f,
                                            restingValue: l,
                                            pluginInstance: d,
                                          });
                                        });
                                    })({
                                      store: n,
                                      eventStateKey: o,
                                      eventTarget: t,
                                      eventId: f,
                                      eventConfig: e,
                                      actionListId: v,
                                      parameterGroup: a,
                                      smoothing: u,
                                      restingValue: d,
                                    });
                                  });
                              });
                            }
                            (a.actionTypeId ===
                              g.ActionTypeConsts.GENERAL_START_ACTION ||
                              T(a.actionTypeId)) &&
                              vt({ store: n, actionListId: v, eventId: f });
                          });
                          var p = function (t) {
                              var e = n.getState(),
                                r = e.ixSession;
                              dt(d, function (e, o, c) {
                                var s = i[o],
                                  f = r.eventState[c],
                                  l = s.action,
                                  d = s.mediaQueries,
                                  p = void 0 === d ? u.mediaQueryKeys : d;
                                if (H(p, r.mediaQueryKey)) {
                                  var v = function () {
                                    var r =
                                        arguments.length > 0 &&
                                        void 0 !== arguments[0]
                                          ? arguments[0]
                                          : {},
                                      i = a(
                                        {
                                          store: n,
                                          element: e,
                                          event: s,
                                          eventConfig: r,
                                          nativeEvent: t,
                                          eventStateKey: c,
                                        },
                                        f
                                      );
                                    (0, E.default)(i, f) ||
                                      n.dispatch(
                                        (0, y.eventStateChanged)(c, i)
                                      );
                                  };
                                  if (
                                    l.actionTypeId ===
                                    g.ActionTypeConsts.GENERAL_CONTINUOUS_ACTION
                                  ) {
                                    var h = Array.isArray(s.config)
                                      ? s.config
                                      : [s.config];
                                    h.forEach(v);
                                  } else v();
                                }
                              });
                            },
                            _ = (0, h.default)(p, tt),
                            I = function (t) {
                              var e = t.target,
                                r = void 0 === e ? document : e,
                                i = t.types,
                                o = t.throttle;
                              i.split(" ")
                                .filter(Boolean)
                                .forEach(function (t) {
                                  var e = o ? _ : p;
                                  r.addEventListener(t, e),
                                    n.dispatch(
                                      (0, y.eventListenerAdded)(r, [t, e])
                                    );
                                });
                            };
                          Array.isArray(o)
                            ? o.forEach(I)
                            : "string" == typeof o && I(e);
                        }
                      })({ logic: i, store: t, events: e })
                    : console.warn("IX2 event type not configured: ".concat(n));
                }),
                t.getState().ixSession.eventListeners.length &&
                  (function (t) {
                    var e = function () {
                      ft(t);
                    };
                    st.forEach(function (n) {
                      window.addEventListener(n, e),
                        t.dispatch((0, y.eventListenerAdded)(window, [n, e]));
                    }),
                      e();
                  })(t);
            })(n),
            -1 === (e = document.documentElement).className.indexOf(R) &&
              (e.className += " ".concat(R)),
            n.getState().ixSession.hasDefinedMediaQueries &&
              (function (t) {
                P({
                  store: t,
                  select: function (t) {
                    return t.ixSession.mediaQueryKey;
                  },
                  onChange: function () {
                    ut(t),
                      F({ store: t, elementApi: m }),
                      at({ store: t, allowEvents: !0 }),
                      nt();
                  },
                });
              })(n)),
          n.dispatch((0, y.sessionStarted)()),
          (function (t, e) {
            !(function n(r) {
              var i = t.getState(),
                o = i.ixSession,
                a = i.ixParameters;
              o.active &&
                (t.dispatch((0, y.animationFrameChanged)(r, a)),
                e
                  ? (function (t, e) {
                      var n = P({
                        store: t,
                        select: function (t) {
                          return t.ixSession.tick;
                        },
                        onChange: function (t) {
                          e(t), n();
                        },
                      });
                    })(t, n)
                  : requestAnimationFrame(n));
            })(window.performance.now());
          })(n, a));
    }
    function ut(t) {
      var e = t.getState().ixSession;
      e.active &&
        (e.eventListeners.forEach(ct), t.dispatch((0, y.sessionStopped)()));
    }
    function ct(t) {
      var e = t.target,
        n = t.listenerParams;
      e.removeEventListener.apply(e, n);
    }
    var st = ["resize", "orientationchange"];
    function ft(t) {
      var e = t.getState(),
        n = e.ixSession,
        r = e.ixData,
        i = window.innerWidth;
      if (i !== n.viewportWidth) {
        var o = r.mediaQueries;
        t.dispatch((0, y.viewportWidthChanged)({ width: i, mediaQueries: o }));
      }
    }
    var lt = function (t, e) {
        return (0, l.default)((0, p.default)(t, e), d.default);
      },
      dt = function (t, e) {
        (0, v.default)(t, function (t, n) {
          t.forEach(function (t, r) {
            e(t, n, n + w + r);
          });
        });
      },
      pt = function (t) {
        var e = { target: t.target, targets: t.targets };
        return N({ config: e, elementApi: m });
      };
    function vt(t) {
      var e = t.store,
        n = t.actionListId,
        r = t.eventId,
        i = e.getState(),
        o = i.ixData,
        a = i.ixSession,
        u = o.actionLists,
        c = o.events[r],
        f = u[n];
      if (f && f.useFirstGroupAsInitialState) {
        var l = (0, s.default)(f, "actionItemGroups[0].actionItems", []),
          d = (0, s.default)(c, "mediaQueries", o.mediaQueryKeys);
        if (!H(d, a.mediaQueryKey)) return;
        l.forEach(function (t) {
          var i,
            o = t.config,
            a = t.actionTypeId,
            u =
              !0 ===
              (null == o
                ? void 0
                : null === (i = o.target) || void 0 === i
                ? void 0
                : i.useEventTarget)
                ? { target: c.target, targets: c.targets }
                : o,
            s = N({ config: u, event: c, elementApi: m }),
            f = Q(a);
          s.forEach(function (i) {
            var o = f ? q(a)(i, t) : null;
            _t({
              destination: D({ element: i, actionItem: t, elementApi: m }, o),
              immediate: !0,
              store: e,
              element: i,
              eventId: r,
              actionItem: t,
              actionListId: n,
              pluginInstance: o,
            });
          });
        });
      }
    }
    function ht(t) {
      var e = t.store,
        n = e.getState().ixInstances;
      (0, v.default)(n, function (t) {
        if (!t.continuous) {
          var n = t.actionListId,
            r = t.verbose;
          yt(t, e),
            r &&
              e.dispatch(
                (0, y.actionListPlaybackChanged)({
                  actionListId: n,
                  isPlaying: !1,
                })
              );
        }
      });
    }
    function Et(t) {
      var e = t.store,
        n = t.eventId,
        r = t.eventTarget,
        i = t.eventStateKey,
        o = t.actionListId,
        a = e.getState(),
        u = a.ixInstances,
        c =
          a.ixSession.hasBoundaryNodes && r ? m.getClosestElement(r, A) : null;
      (0, v.default)(u, function (t) {
        var r = (0, s.default)(t, "actionItem.config.target.boundaryMode"),
          a = !i || t.eventStateKey === i;
        if (t.actionListId === o && t.eventId === n && a) {
          if (c && r && !m.elementContains(c, t.element)) return;
          yt(t, e),
            t.verbose &&
              e.dispatch(
                (0, y.actionListPlaybackChanged)({
                  actionListId: o,
                  isPlaying: !1,
                })
              );
        }
      });
    }
    function gt(t) {
      var e,
        n = t.store,
        r = t.eventId,
        i = t.eventTarget,
        o = t.eventStateKey,
        a = t.actionListId,
        u = t.groupIndex,
        c = void 0 === u ? 0 : u,
        f = t.immediate,
        l = t.verbose,
        d = n.getState(),
        p = d.ixData,
        v = d.ixSession,
        h = p.events[r] || {},
        E = h.mediaQueries,
        g = void 0 === E ? p.mediaQueryKeys : E,
        _ = (0, s.default)(p, "actionLists.".concat(a), {}),
        y = _.actionItemGroups,
        I = _.useFirstGroupAsInitialState;
      if (!y || !y.length) return !1;
      c >= y.length && (0, s.default)(h, "config.loop") && (c = 0),
        0 === c && I && c++;
      var b =
          (0 === c || (1 === c && I)) &&
          T(null === (e = h.action) || void 0 === e ? void 0 : e.actionTypeId)
            ? h.config.delay
            : void 0,
        O = (0, s.default)(y, [c, "actionItems"], []);
      if (!O.length) return !1;
      if (!H(g, v.mediaQueryKey)) return !1;
      var w = v.hasBoundaryNodes && i ? m.getClosestElement(i, A) : null,
        S = k(O),
        x = !1;
      return (
        O.forEach(function (t, e) {
          var u = t.config,
            s = t.actionTypeId,
            d = Q(s),
            p = u.target;
          if (p) {
            var v = p.boundaryMode ? w : null;
            N({
              config: u,
              event: h,
              eventTarget: i,
              elementRoot: v,
              elementApi: m,
            }).forEach(function (u, p) {
              var v = d ? q(s)(u, t) : null,
                h = d ? $(s)(u, t) : null;
              x = !0;
              var E = S === e && 0 === p,
                g = G({ element: u, actionItem: t }),
                _ = D({ element: u, actionItem: t, elementApi: m }, v);
              _t({
                store: n,
                element: u,
                actionItem: t,
                eventId: r,
                eventTarget: i,
                eventStateKey: o,
                actionListId: a,
                groupIndex: c,
                isCarrier: E,
                computedStyle: g,
                destination: _,
                immediate: f,
                verbose: l,
                pluginInstance: v,
                pluginDuration: h,
                instanceDelay: b,
              });
            });
          }
        }),
        x
      );
    }
    function _t(t) {
      var e = t.store,
        n = t.computedStyle,
        r = (0, u.default)(t, ["store", "computedStyle"]),
        i = !r.continuous,
        o = r.element,
        c = r.actionItem,
        s = r.immediate,
        f = r.pluginInstance,
        l = M(),
        d = e.getState(),
        p = d.ixElements,
        v = d.ixSession,
        h = L(p, o),
        E = (p[h] || {}).refState,
        g = m.getRefType(o),
        _ = X(o, E, n, c, m, f);
      e.dispatch(
        (0, y.instanceAdded)(
          (0, a.default)(
            { instanceId: l, elementId: h, origin: _, refType: g },
            r
          )
        )
      ),
        mt(document.body, "ix2-animation-started", l),
        s
          ? (function (t, e) {
              var n = t.getState().ixParameters;
              t.dispatch((0, y.instanceStarted)(e, 0)),
                t.dispatch((0, y.animationFrameChanged)(performance.now(), n)),
                It(t.getState().ixInstances[e], t);
            })(e, l)
          : (P({
              store: e,
              select: function (t) {
                return t.ixInstances[l];
              },
              onChange: It,
            }),
            i && e.dispatch((0, y.instanceStarted)(l, v.tick)));
    }
    function yt(t, e) {
      mt(document.body, "ix2-animation-stopping", {
        instanceId: t.id,
        state: e.getState(),
      });
      var n = t.elementId,
        r = t.actionItem,
        i = e.getState().ixElements[n] || {},
        o = i.ref;
      i.refType === S && B(o, r, m), e.dispatch((0, y.instanceRemoved)(t.id));
    }
    function mt(t, e, n) {
      var r = document.createEvent("CustomEvent");
      r.initCustomEvent(e, !0, !0, n), t.dispatchEvent(r);
    }
    function It(t, e) {
      var n = t.active,
        r = t.continuous,
        i = t.complete,
        o = t.elementId,
        a = t.actionItem,
        u = t.actionTypeId,
        c = t.renderType,
        s = t.current,
        f = t.groupIndex,
        l = t.eventId,
        d = t.eventTarget,
        p = t.eventStateKey,
        v = t.actionListId,
        h = t.isCarrier,
        E = t.styleProp,
        g = t.verbose,
        _ = t.pluginInstance,
        I = e.getState(),
        b = I.ixData,
        T = I.ixSession,
        O = (b.events[l] || {}).mediaQueries,
        w = void 0 === O ? b.mediaQueryKeys : O;
      if (H(w, T.mediaQueryKey) && (r || n || i)) {
        if (s || (c === x && i)) {
          e.dispatch((0, y.elementStateChanged)(o, u, s, a));
          var A = e.getState().ixElements[o] || {},
            R = A.ref,
            C = A.refType,
            N = A.refState,
            L = N && N[u];
          switch (C) {
            case S:
              j(R, N, L, l, a, E, m, c, _);
          }
        }
        if (i) {
          if (h) {
            var D = gt({
              store: e,
              eventId: l,
              eventTarget: d,
              eventStateKey: p,
              actionListId: v,
              groupIndex: f + 1,
              verbose: g,
            });
            g &&
              !D &&
              e.dispatch(
                (0, y.actionListPlaybackChanged)({
                  actionListId: v,
                  isPlaying: !1,
                })
              );
          }
          yt(t, e);
        }
      }
    }
  },
  function (t, e, n) {
    var r = n(116);
    t.exports = function (t, e, n) {
      "__proto__" == e && r
        ? r(t, e, { configurable: !0, enumerable: !0, value: n, writable: !0 })
        : (t[e] = n);
    };
  },
  function (t, e, n) {
    var r = n(8),
      i = (function () {
        try {
          var t = r(Object, "defineProperty");
          return t({}, "", {}), t;
        } catch (t) {}
      })();
    t.exports = i;
  },
  function (t, e, n) {
    var r = n(6),
      i = Object.create,
      o = (function () {
        function t() {}
        return function (e) {
          if (!r(e)) return {};
          if (i) return i(e);
          t.prototype = e;
          var n = new t();
          return (t.prototype = void 0), n;
        };
      })();
    t.exports = o;
  },
  function (t, e, n) {
    var r = n(304),
      i = n(305),
      o = r
        ? function (t) {
            return r.get(t);
          }
        : i;
    t.exports = o;
  },
  function (t, e, n) {
    var r = n(306),
      i = Object.prototype.hasOwnProperty;
    t.exports = function (t) {
      for (
        var e = t.name + "", n = r[e], o = i.call(r, e) ? n.length : 0;
        o--;

      ) {
        var a = n[o],
          u = a.func;
        if (null == u || u == t) return a.name;
      }
      return e;
    };
  },
  function (t, e, n) {
    n(121),
      n(123),
      n(13),
      n(125),
      n(313),
      n(314),
      n(315),
      n(316),
      n(317),
      n(322),
      n(323),
      (t.exports = n(324));
  },
  function (t, e, n) {
    "use strict";
    var r = n(3);
    r.define(
      "brand",
      (t.exports = function (t) {
        var e,
          n = {},
          i = document,
          o = t("html"),
          a = t("body"),
          u = ".w-webflow-badge",
          c = window.location,
          s = /PhantomJS/i.test(navigator.userAgent),
          f =
            "fullscreenchange webkitfullscreenchange mozfullscreenchange msfullscreenchange";
        function l() {
          var n =
            i.fullScreen ||
            i.mozFullScreen ||
            i.webkitIsFullScreen ||
            i.msFullscreenElement ||
            Boolean(i.webkitFullscreenElement);
          t(e).attr("style", n ? "display: none !important;" : "");
        }
        function d() {
          var t = a.children(u),
            n = t.length && t.get(0) === e,
            i = r.env("editor");
          n ? i && t.remove() : (t.length && t.remove(), i || a.append(e));
        }
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = window.$,
      i = n(65) && r.tram;
    /*!
     * Webflow._ (aka) Underscore.js 1.6.0 (custom build)
     * _.each
     * _.map
     * _.find
     * _.filter
     * _.any
     * _.contains
     * _.delay
     * _.defer
     * _.throttle (webflow)
     * _.debounce
     * _.keys
     * _.has
     * _.now
     *
     * http://underscorejs.org
     * (c) 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
     * Underscore may be freely distributed under the MIT license.
     * @license MIT
     */
    t.exports = (function () {
      var t = { VERSION: "1.6.0-Webflow" },
        e = {},
        n = Array.prototype,
        r = Object.prototype,
        o = Function.prototype,
        a = (n.push, n.slice),
        u = (n.concat, r.toString, r.hasOwnProperty),
        c = n.forEach,
        s = n.map,
        f = (n.reduce, n.reduceRight, n.filter),
        l = (n.every, n.some),
        d = n.indexOf,
        p = (n.lastIndexOf, Array.isArray, Object.keys),
        v =
          (o.bind,
          (t.each = t.forEach =
            function (n, r, i) {
              if (null == n) return n;
              if (c && n.forEach === c) n.forEach(r, i);
              else if (n.length === +n.length) {
                for (var o = 0, a = n.length; o < a; o++)
                  if (r.call(i, n[o], o, n) === e) return;
              } else {
                var u = t.keys(n);
                for (o = 0, a = u.length; o < a; o++)
                  if (r.call(i, n[u[o]], u[o], n) === e) return;
              }
              return n;
            }));
      (t.map = t.collect =
        function (t, e, n) {
          var r = [];
          return null == t
            ? r
            : s && t.map === s
            ? t.map(e, n)
            : (v(t, function (t, i, o) {
                r.push(e.call(n, t, i, o));
              }),
              r);
        }),
        (t.find = t.detect =
          function (t, e, n) {
            var r;
            return (
              h(t, function (t, i, o) {
                if (e.call(n, t, i, o)) return (r = t), !0;
              }),
              r
            );
          }),
        (t.filter = t.select =
          function (t, e, n) {
            var r = [];
            return null == t
              ? r
              : f && t.filter === f
              ? t.filter(e, n)
              : (v(t, function (t, i, o) {
                  e.call(n, t, i, o) && r.push(t);
                }),
                r);
          });
      var h =
        (t.some =
        t.any =
          function (n, r, i) {
            r || (r = t.identity);
            var o = !1;
            return null == n
              ? o
              : l && n.some === l
              ? n.some(r, i)
              : (v(n, function (t, n, a) {
                  if (o || (o = r.call(i, t, n, a))) return e;
                }),
                !!o);
          });
      (t.contains = t.include =
        function (t, e) {
          return (
            null != t &&
            (d && t.indexOf === d
              ? -1 != t.indexOf(e)
              : h(t, function (t) {
                  return t === e;
                }))
          );
        }),
        (t.delay = function (t, e) {
          var n = a.call(arguments, 2);
          return setTimeout(function () {
            return t.apply(null, n);
          }, e);
        }),
        (t.defer = function (e) {
          return t.delay.apply(t, [e, 1].concat(a.call(arguments, 1)));
        }),
        (t.throttle = function (t) {
          var e, n, r;
          return function () {
            e ||
              ((e = !0),
              (n = arguments),
              (r = this),
              i.frame(function () {
                (e = !1), t.apply(r, n);
              }));
          };
        }),
        (t.debounce = function (e, n, r) {
          var i,
            o,
            a,
            u,
            c,
            s = function s() {
              var f = t.now() - u;
              f < n
                ? (i = setTimeout(s, n - f))
                : ((i = null), r || ((c = e.apply(a, o)), (a = o = null)));
            };
          return function () {
            (a = this), (o = arguments), (u = t.now());
            var f = r && !i;
            return (
              i || (i = setTimeout(s, n)),
              f && ((c = e.apply(a, o)), (a = o = null)),
              c
            );
          };
        }),
        (t.defaults = function (e) {
          if (!t.isObject(e)) return e;
          for (var n = 1, r = arguments.length; n < r; n++) {
            var i = arguments[n];
            for (var o in i) void 0 === e[o] && (e[o] = i[o]);
          }
          return e;
        }),
        (t.keys = function (e) {
          if (!t.isObject(e)) return [];
          if (p) return p(e);
          var n = [];
          for (var r in e) t.has(e, r) && n.push(r);
          return n;
        }),
        (t.has = function (t, e) {
          return u.call(t, e);
        }),
        (t.isObject = function (t) {
          return t === Object(t);
        }),
        (t.now =
          Date.now ||
          function () {
            return new Date().getTime();
          }),
        (t.templateSettings = {
          evaluate: /<%([\s\S]+?)%>/g,
          interpolate: /<%=([\s\S]+?)%>/g,
          escape: /<%-([\s\S]+?)%>/g,
        });
      var E = /(.)^/,
        g = {
          "'": "'",
          "\\": "\\",
          "\r": "r",
          "\n": "n",
          "\u2028": "u2028",
          "\u2029": "u2029",
        },
        _ = /\\|'|\r|\n|\u2028|\u2029/g,
        y = function (t) {
          return "\\" + g[t];
        };
      return (
        (t.template = function (e, n, r) {
          !n && r && (n = r), (n = t.defaults({}, n, t.templateSettings));
          var i = RegExp(
              [
                (n.escape || E).source,
                (n.interpolate || E).source,
                (n.evaluate || E).source,
              ].join("|") + "|$",
              "g"
            ),
            o = 0,
            a = "__p+='";
          e.replace(i, function (t, n, r, i, u) {
            return (
              (a += e.slice(o, u).replace(_, y)),
              (o = u + t.length),
              n
                ? (a += "'+\n((__t=(" + n + "))==null?'':_.escape(__t))+\n'")
                : r
                ? (a += "'+\n((__t=(" + r + "))==null?'':__t)+\n'")
                : i && (a += "';\n" + i + "\n__p+='"),
              t
            );
          }),
            (a += "';\n"),
            n.variable || (a = "with(obj||{}){\n" + a + "}\n"),
            (a =
              "var __t,__p='',__j=Array.prototype.join,print=function(){__p+=__j.call(arguments,'');};\n" +
              a +
              "return __p;\n");
          try {
            var u = new Function(n.variable || "obj", "_", a);
          } catch (t) {
            throw ((t.source = a), t);
          }
          var c = function (e) {
              return u.call(this, e, t);
            },
            s = n.variable || "obj";
          return (c.source = "function(" + s + "){\n" + a + "}"), c;
        }),
        t
      );
    })();
  },
  function (t, e, n) {
    "use strict";
    var r = n(3);
    r.define(
      "edit",
      (t.exports = function (t, e, n) {
        if (
          ((n = n || {}),
          (r.env("test") || r.env("frame")) &&
            !n.fixture &&
            !(function () {
              try {
                return window.top.__Cypress__;
              } catch (t) {
                return !1;
              }
            })())
        )
          return { exit: 1 };
        var i,
          o = t(window),
          a = t(document.documentElement),
          u = document.location,
          c = "hashchange",
          s =
            n.load ||
            function () {
              (i = !0),
                (window.WebflowEditor = !0),
                o.off(c, l),
                (function (t) {
                  var e = window.document.createElement("iframe");
                  (e.src =
                    "https://webflow.com/site/third-party-cookie-check.html"),
                    (e.style.display = "none"),
                    (e.sandbox = "allow-scripts allow-same-origin");
                  var n = function n(r) {
                    "WF_third_party_cookies_unsupported" === r.data
                      ? (g(e, n), t(!1))
                      : "WF_third_party_cookies_supported" === r.data &&
                        (g(e, n), t(!0));
                  };
                  (e.onerror = function () {
                    g(e, n), t(!1);
                  }),
                    window.addEventListener("message", n, !1),
                    window.document.body.appendChild(e);
                })(function (e) {
                  t.ajax({
                    url: E("https://editor-api.webflow.com/api/editor/view"),
                    data: { siteId: a.attr("data-wf-site") },
                    xhrFields: { withCredentials: !0 },
                    dataType: "json",
                    crossDomain: !0,
                    success: d(e),
                  });
                });
            },
          f = !1;
        try {
          f =
            localStorage &&
            localStorage.getItem &&
            localStorage.getItem("WebflowEditor");
        } catch (t) {}
        function l() {
          i || (/\?edit/.test(u.hash) && s());
        }
        function d(t) {
          return function (e) {
            e
              ? ((e.thirdPartyCookiesSupported = t),
                p(h(e.bugReporterScriptPath), function () {
                  p(h(e.scriptPath), function () {
                    window.WebflowEditor(e);
                  });
                }))
              : console.error("Could not load editor data");
          };
        }
        function p(e, n) {
          t.ajax({ type: "GET", url: e, dataType: "script", cache: !0 }).then(
            n,
            v
          );
        }
        function v(t, e, n) {
          throw (console.error("Could not load editor script: " + e), n);
        }
        function h(t) {
          return t.indexOf("//") >= 0
            ? t
            : E("https://editor-api.webflow.com" + t);
        }
        function E(t) {
          return t.replace(/([^:])\/\//g, "$1/");
        }
        function g(t, e) {
          window.removeEventListener("message", e, !1), t.remove();
        }
        return (
          f
            ? s()
            : u.search
            ? (/[?&](edit)(?:[=&?]|$)/.test(u.search) ||
                /\?edit$/.test(u.href)) &&
              s()
            : o.on(c, l).triggerHandler(c),
          {}
        );
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = window.jQuery,
      i = {},
      o = [],
      a = {
        reset: function (t, e) {
          e.__wf_intro = null;
        },
        intro: function (t, e) {
          e.__wf_intro ||
            ((e.__wf_intro = !0), r(e).triggerHandler(i.types.INTRO));
        },
        outro: function (t, e) {
          e.__wf_intro &&
            ((e.__wf_intro = null), r(e).triggerHandler(i.types.OUTRO));
        },
      };
    (i.triggers = {}),
      (i.types = { INTRO: "w-ix-intro.w-ix", OUTRO: "w-ix-outro.w-ix" }),
      (i.init = function () {
        for (var t = o.length, e = 0; e < t; e++) {
          var n = o[e];
          n[0](0, n[1]);
        }
        (o = []), r.extend(i.triggers, a);
      }),
      (i.async = function () {
        for (var t in a) {
          var e = a[t];
          a.hasOwnProperty(t) &&
            (i.triggers[t] = function (t, n) {
              o.push([e, n]);
            });
        }
      }),
      i.async(),
      (t.exports = i);
  },
  function (t, e, n) {
    "use strict";
    var r = n(3),
      i = n(126);
    i.setEnv(r.env),
      r.define(
        "ix2",
        (t.exports = function () {
          return i;
        })
      );
  },
  function (t, e, n) {
    "use strict";
    var r = n(14),
      i = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.setEnv = function (t) {
        t() && (0, u.observeRequests)(s);
      }),
      (e.init = function (t) {
        f(), (0, u.startEngine)({ store: s, rawData: t, allowEvents: !0 });
      }),
      (e.destroy = f),
      (e.actions = e.store = void 0),
      n(127);
    var o = n(78),
      a = i(n(168)),
      u = n(114),
      c = r(n(61));
    e.actions = c;
    var s = (0, o.createStore)(a.default);
    function f() {
      (0, u.stopEngine)(s);
    }
    e.store = s;
  },
  function (t, e, n) {
    t.exports = n(128);
  },
  function (t, e, n) {
    n(129);
    var r = n(154);
    t.exports = r("Array", "includes");
  },
  function (t, e, n) {
    "use strict";
    var r = n(130),
      i = n(76).includes,
      o = n(147);
    r(
      { target: "Array", proto: !0 },
      {
        includes: function (t) {
          return i(this, t, arguments.length > 1 ? arguments[1] : void 0);
        },
      }
    ),
      o("includes");
  },
  function (t, e, n) {
    var r = n(4),
      i = n(66).f,
      o = n(18),
      a = n(135),
      u = n(39),
      c = n(139),
      s = n(146);
    t.exports = function (t, e) {
      var n,
        f,
        l,
        d,
        p,
        v = t.target,
        h = t.global,
        E = t.stat;
      if ((n = h ? r : E ? r[v] || u(v, {}) : (r[v] || {}).prototype))
        for (f in e) {
          if (
            ((d = e[f]),
            (l = t.noTargetGet ? (p = i(n, f)) && p.value : n[f]),
            !s(h ? f : v + (E ? "." : "#") + f, t.forced) && void 0 !== l)
          ) {
            if (typeof d == typeof l) continue;
            c(d, l);
          }
          (t.sham || (l && l.sham)) && o(d, "sham", !0), a(n, f, d, t);
        }
    };
  },
  function (t, e, n) {
    "use strict";
    var r = {}.propertyIsEnumerable,
      i = Object.getOwnPropertyDescriptor,
      o = i && !r.call({ 1: 2 }, 1);
    e.f = o
      ? function (t) {
          var e = i(this, t);
          return !!e && e.enumerable;
        }
      : r;
  },
  function (t, e, n) {
    var r = n(16),
      i = n(133),
      o = "".split;
    t.exports = r(function () {
      return !Object("z").propertyIsEnumerable(0);
    })
      ? function (t) {
          return "String" == i(t) ? o.call(t, "") : Object(t);
        }
      : Object;
  },
  function (t, e) {
    var n = {}.toString;
    t.exports = function (t) {
      return n.call(t).slice(8, -1);
    };
  },
  function (t, e) {
    t.exports = function (t) {
      if (null == t) throw TypeError("Can't call method on " + t);
      return t;
    };
  },
  function (t, e, n) {
    var r = n(4),
      i = n(26),
      o = n(18),
      a = n(17),
      u = n(39),
      c = n(71),
      s = n(137),
      f = s.get,
      l = s.enforce,
      d = String(c).split("toString");
    i("inspectSource", function (t) {
      return c.call(t);
    }),
      (t.exports = function (t, e, n, i) {
        var c = !!i && !!i.unsafe,
          s = !!i && !!i.enumerable,
          f = !!i && !!i.noTargetGet;
        "function" == typeof n &&
          ("string" != typeof e || a(n, "name") || o(n, "name", e),
          (l(n).source = d.join("string" == typeof e ? e : ""))),
          t !== r
            ? (c ? !f && t[e] && (s = !0) : delete t[e],
              s ? (t[e] = n) : o(t, e, n))
            : s
            ? (t[e] = n)
            : u(e, n);
      })(Function.prototype, "toString", function () {
        return ("function" == typeof this && f(this).source) || c.call(this);
      });
  },
  function (t, e) {
    t.exports = !1;
  },
  function (t, e, n) {
    var r,
      i,
      o,
      a = n(138),
      u = n(4),
      c = n(24),
      s = n(18),
      f = n(17),
      l = n(72),
      d = n(40),
      p = u.WeakMap;
    if (a) {
      var v = new p(),
        h = v.get,
        E = v.has,
        g = v.set;
      (r = function (t, e) {
        return g.call(v, t, e), e;
      }),
        (i = function (t) {
          return h.call(v, t) || {};
        }),
        (o = function (t) {
          return E.call(v, t);
        });
    } else {
      var _ = l("state");
      (d[_] = !0),
        (r = function (t, e) {
          return s(t, _, e), e;
        }),
        (i = function (t) {
          return f(t, _) ? t[_] : {};
        }),
        (o = function (t) {
          return f(t, _);
        });
    }
    t.exports = {
      set: r,
      get: i,
      has: o,
      enforce: function (t) {
        return o(t) ? i(t) : r(t, {});
      },
      getterFor: function (t) {
        return function (e) {
          var n;
          if (!c(e) || (n = i(e)).type !== t)
            throw TypeError("Incompatible receiver, " + t + " required");
          return n;
        };
      },
    };
  },
  function (t, e, n) {
    var r = n(4),
      i = n(71),
      o = r.WeakMap;
    t.exports = "function" == typeof o && /native code/.test(i.call(o));
  },
  function (t, e, n) {
    var r = n(17),
      i = n(140),
      o = n(66),
      a = n(38);
    t.exports = function (t, e) {
      for (var n = i(e), u = a.f, c = o.f, s = 0; s < n.length; s++) {
        var f = n[s];
        r(t, f) || u(t, f, c(e, f));
      }
    };
  },
  function (t, e, n) {
    var r = n(74),
      i = n(142),
      o = n(145),
      a = n(25);
    t.exports =
      r("Reflect", "ownKeys") ||
      function (t) {
        var e = i.f(a(t)),
          n = o.f;
        return n ? e.concat(n(t)) : e;
      };
  },
  function (t, e, n) {
    t.exports = n(4);
  },
  function (t, e, n) {
    var r = n(75),
      i = n(41).concat("length", "prototype");
    e.f =
      Object.getOwnPropertyNames ||
      function (t) {
        return r(t, i);
      };
  },
  function (t, e, n) {
    var r = n(77),
      i = Math.min;
    t.exports = function (t) {
      return t > 0 ? i(r(t), 9007199254740991) : 0;
    };
  },
  function (t, e, n) {
    var r = n(77),
      i = Math.max,
      o = Math.min;
    t.exports = function (t, e) {
      var n = r(t);
      return n < 0 ? i(n + e, 0) : o(n, e);
    };
  },
  function (t, e) {
    e.f = Object.getOwnPropertySymbols;
  },
  function (t, e, n) {
    var r = n(16),
      i = /#|\.prototype\./,
      o = function (t, e) {
        var n = u[a(t)];
        return n == s || (n != c && ("function" == typeof e ? r(e) : !!e));
      },
      a = (o.normalize = function (t) {
        return String(t).replace(i, ".").toLowerCase();
      }),
      u = (o.data = {}),
      c = (o.NATIVE = "N"),
      s = (o.POLYFILL = "P");
    t.exports = o;
  },
  function (t, e, n) {
    var r = n(148),
      i = n(150),
      o = n(18),
      a = r("unscopables"),
      u = Array.prototype;
    null == u[a] && o(u, a, i(null)),
      (t.exports = function (t) {
        u[a][t] = !0;
      });
  },
  function (t, e, n) {
    var r = n(4),
      i = n(26),
      o = n(73),
      a = n(149),
      u = r.Symbol,
      c = i("wks");
    t.exports = function (t) {
      return c[t] || (c[t] = (a && u[t]) || (a ? u : o)("Symbol." + t));
    };
  },
  function (t, e, n) {
    var r = n(16);
    t.exports =
      !!Object.getOwnPropertySymbols &&
      !r(function () {
        return !String(Symbol());
      });
  },
  function (t, e, n) {
    var r = n(25),
      i = n(151),
      o = n(41),
      a = n(40),
      u = n(153),
      c = n(70),
      s = n(72)("IE_PROTO"),
      f = function () {},
      l = function () {
        var t,
          e = c("iframe"),
          n = o.length;
        for (
          e.style.display = "none",
            u.appendChild(e),
            e.src = String("javascript:"),
            (t = e.contentWindow.document).open(),
            t.write("<script>document.F=Object</script>"),
            t.close(),
            l = t.F;
          n--;

        )
          delete l.prototype[o[n]];
        return l();
      };
    (t.exports =
      Object.create ||
      function (t, e) {
        var n;
        return (
          null !== t
            ? ((f.prototype = r(t)),
              (n = new f()),
              (f.prototype = null),
              (n[s] = t))
            : (n = l()),
          void 0 === e ? n : i(n, e)
        );
      }),
      (a[s] = !0);
  },
  function (t, e, n) {
    var r = n(15),
      i = n(38),
      o = n(25),
      a = n(152);
    t.exports = r
      ? Object.defineProperties
      : function (t, e) {
          o(t);
          for (var n, r = a(e), u = r.length, c = 0; u > c; )
            i.f(t, (n = r[c++]), e[n]);
          return t;
        };
  },
  function (t, e, n) {
    var r = n(75),
      i = n(41);
    t.exports =
      Object.keys ||
      function (t) {
        return r(t, i);
      };
  },
  function (t, e, n) {
    var r = n(74);
    t.exports = r("document", "documentElement");
  },
  function (t, e, n) {
    var r = n(4),
      i = n(155),
      o = Function.call;
    t.exports = function (t, e, n) {
      return i(o, r[t].prototype[e], n);
    };
  },
  function (t, e, n) {
    var r = n(156);
    t.exports = function (t, e, n) {
      if ((r(t), void 0 === e)) return t;
      switch (n) {
        case 0:
          return function () {
            return t.call(e);
          };
        case 1:
          return function (n) {
            return t.call(e, n);
          };
        case 2:
          return function (n, r) {
            return t.call(e, n, r);
          };
        case 3:
          return function (n, r, i) {
            return t.call(e, n, r, i);
          };
      }
      return function () {
        return t.apply(e, arguments);
      };
    };
  },
  function (t, e) {
    t.exports = function (t) {
      if ("function" != typeof t)
        throw TypeError(String(t) + " is not a function");
      return t;
    };
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(80),
      i = n(160),
      o = n(161),
      a = "[object Null]",
      u = "[object Undefined]",
      c = r.default ? r.default.toStringTag : void 0;
    e.default = function (t) {
      return null == t
        ? void 0 === t
          ? u
          : a
        : c && c in Object(t)
        ? Object(i.default)(t)
        : Object(o.default)(t);
    };
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(159),
      i = "object" == typeof self && self && self.Object === Object && self,
      o = r.default || i || Function("return this")();
    e.default = o;
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      function (t) {
        var n = "object" == typeof t && t && t.Object === Object && t;
        e.default = n;
      }.call(this, n(23));
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(80),
      i = Object.prototype,
      o = i.hasOwnProperty,
      a = i.toString,
      u = r.default ? r.default.toStringTag : void 0;
    e.default = function (t) {
      var e = o.call(t, u),
        n = t[u];
      try {
        t[u] = void 0;
        var r = !0;
      } catch (t) {}
      var i = a.call(t);
      return r && (e ? (t[u] = n) : delete t[u]), i;
    };
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = Object.prototype.toString;
    e.default = function (t) {
      return r.call(t);
    };
  },
  function (t, e, n) {
    "use strict";
    n.r(e);
    var r = n(163),
      i = Object(r.default)(Object.getPrototypeOf, Object);
    e.default = i;
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      (e.default = function (t, e) {
        return function (n) {
          return t(e(n));
        };
      });
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      (e.default = function (t) {
        return null != t && "object" == typeof t;
      });
  },
  function (t, e, n) {
    "use strict";
    n.r(e),
      function (t, r) {
        var i,
          o = n(167);
        i =
          "undefined" != typeof self
            ? self
            : "undefined" != typeof window
            ? window
            : void 0 !== t
            ? t
            : r;
        var a = Object(o.default)(i);
        e.default = a;
      }.call(this, n(23), n(166)(t));
  },
  function (t, e) {
    t.exports = function (t) {
      if (!t.webpackPolyfill) {
        var e = Object.create(t);
        e.children || (e.children = []),
          Object.defineProperty(e, "loaded", {
            enumerable: !0,
            get: function () {
              return e.l;
            },
          }),
          Object.defineProperty(e, "id", {
            enumerable: !0,
            get: function () {
              return e.i;
            },
          }),
          Object.defineProperty(e, "exports", { enumerable: !0 }),
          (e.webpackPolyfill = 1);
      }
      return e;
    };
  },
  function (t, e, n) {
    "use strict";
    function r(t) {
      var e,
        n = t.Symbol;
      return (
        "function" == typeof n
          ? n.observable
            ? (e = n.observable)
            : ((e = n("observable")), (n.observable = e))
          : (e = "@@observable"),
        e
      );
    }
    n.r(e),
      n.d(e, "default", function () {
        return r;
      });
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }), (e.default = void 0);
    var r = n(78),
      i = n(169),
      o = n(175),
      a = n(176),
      u = n(10),
      c = n(261),
      s = n(262),
      f = u.IX2ElementsReducer.ixElements,
      l = (0, r.combineReducers)({
        ixData: i.ixData,
        ixRequest: o.ixRequest,
        ixSession: a.ixSession,
        ixElements: f,
        ixInstances: c.ixInstances,
        ixParameters: s.ixParameters,
      });
    e.default = l;
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }), (e.ixData = void 0);
    var r = n(2).IX2EngineActionTypes.IX2_RAW_DATA_IMPORTED;
    e.ixData = function () {
      var t =
          arguments.length > 0 && void 0 !== arguments[0]
            ? arguments[0]
            : Object.freeze({}),
        e = arguments.length > 1 ? arguments[1] : void 0;
      switch (e.type) {
        case r:
          return e.payload.ixData || Object.freeze({});
        default:
          return t;
      }
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.QuickEffectDirectionConsts =
        e.QuickEffectIds =
        e.EventLimitAffectedElements =
        e.EventContinuousMouseAxes =
        e.EventBasedOn =
        e.EventAppliesTo =
        e.EventTypeConsts =
          void 0);
    e.EventTypeConsts = {
      NAVBAR_OPEN: "NAVBAR_OPEN",
      NAVBAR_CLOSE: "NAVBAR_CLOSE",
      TAB_ACTIVE: "TAB_ACTIVE",
      TAB_INACTIVE: "TAB_INACTIVE",
      SLIDER_ACTIVE: "SLIDER_ACTIVE",
      SLIDER_INACTIVE: "SLIDER_INACTIVE",
      DROPDOWN_OPEN: "DROPDOWN_OPEN",
      DROPDOWN_CLOSE: "DROPDOWN_CLOSE",
      MOUSE_CLICK: "MOUSE_CLICK",
      MOUSE_SECOND_CLICK: "MOUSE_SECOND_CLICK",
      MOUSE_DOWN: "MOUSE_DOWN",
      MOUSE_UP: "MOUSE_UP",
      MOUSE_OVER: "MOUSE_OVER",
      MOUSE_OUT: "MOUSE_OUT",
      MOUSE_MOVE: "MOUSE_MOVE",
      MOUSE_MOVE_IN_VIEWPORT: "MOUSE_MOVE_IN_VIEWPORT",
      SCROLL_INTO_VIEW: "SCROLL_INTO_VIEW",
      SCROLL_OUT_OF_VIEW: "SCROLL_OUT_OF_VIEW",
      SCROLLING_IN_VIEW: "SCROLLING_IN_VIEW",
      ECOMMERCE_CART_OPEN: "ECOMMERCE_CART_OPEN",
      ECOMMERCE_CART_CLOSE: "ECOMMERCE_CART_CLOSE",
      PAGE_START: "PAGE_START",
      PAGE_FINISH: "PAGE_FINISH",
      PAGE_SCROLL_UP: "PAGE_SCROLL_UP",
      PAGE_SCROLL_DOWN: "PAGE_SCROLL_DOWN",
      PAGE_SCROLL: "PAGE_SCROLL",
    };
    e.EventAppliesTo = { ELEMENT: "ELEMENT", CLASS: "CLASS", PAGE: "PAGE" };
    e.EventBasedOn = { ELEMENT: "ELEMENT", VIEWPORT: "VIEWPORT" };
    e.EventContinuousMouseAxes = { X_AXIS: "X_AXIS", Y_AXIS: "Y_AXIS" };
    e.EventLimitAffectedElements = {
      CHILDREN: "CHILDREN",
      SIBLINGS: "SIBLINGS",
      IMMEDIATE_CHILDREN: "IMMEDIATE_CHILDREN",
    };
    e.QuickEffectIds = {
      FADE_EFFECT: "FADE_EFFECT",
      SLIDE_EFFECT: "SLIDE_EFFECT",
      GROW_EFFECT: "GROW_EFFECT",
      SHRINK_EFFECT: "SHRINK_EFFECT",
      SPIN_EFFECT: "SPIN_EFFECT",
      FLY_EFFECT: "FLY_EFFECT",
      POP_EFFECT: "POP_EFFECT",
      FLIP_EFFECT: "FLIP_EFFECT",
      JIGGLE_EFFECT: "JIGGLE_EFFECT",
      PULSE_EFFECT: "PULSE_EFFECT",
      DROP_EFFECT: "DROP_EFFECT",
      BLINK_EFFECT: "BLINK_EFFECT",
      BOUNCE_EFFECT: "BOUNCE_EFFECT",
      FLIP_LEFT_TO_RIGHT_EFFECT: "FLIP_LEFT_TO_RIGHT_EFFECT",
      FLIP_RIGHT_TO_LEFT_EFFECT: "FLIP_RIGHT_TO_LEFT_EFFECT",
      RUBBER_BAND_EFFECT: "RUBBER_BAND_EFFECT",
      JELLO_EFFECT: "JELLO_EFFECT",
      GROW_BIG_EFFECT: "GROW_BIG_EFFECT",
      SHRINK_BIG_EFFECT: "SHRINK_BIG_EFFECT",
      PLUGIN_LOTTIE_EFFECT: "PLUGIN_LOTTIE_EFFECT",
    };
    e.QuickEffectDirectionConsts = {
      LEFT: "LEFT",
      RIGHT: "RIGHT",
      BOTTOM: "BOTTOM",
      TOP: "TOP",
      BOTTOM_LEFT: "BOTTOM_LEFT",
      BOTTOM_RIGHT: "BOTTOM_RIGHT",
      TOP_RIGHT: "TOP_RIGHT",
      TOP_LEFT: "TOP_LEFT",
      CLOCKWISE: "CLOCKWISE",
      COUNTER_CLOCKWISE: "COUNTER_CLOCKWISE",
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.ActionAppliesTo = e.ActionTypeConsts = void 0);
    e.ActionTypeConsts = {
      TRANSFORM_MOVE: "TRANSFORM_MOVE",
      TRANSFORM_SCALE: "TRANSFORM_SCALE",
      TRANSFORM_ROTATE: "TRANSFORM_ROTATE",
      TRANSFORM_SKEW: "TRANSFORM_SKEW",
      STYLE_OPACITY: "STYLE_OPACITY",
      STYLE_SIZE: "STYLE_SIZE",
      STYLE_FILTER: "STYLE_FILTER",
      STYLE_BACKGROUND_COLOR: "STYLE_BACKGROUND_COLOR",
      STYLE_BORDER: "STYLE_BORDER",
      STYLE_TEXT_COLOR: "STYLE_TEXT_COLOR",
      PLUGIN_LOTTIE: "PLUGIN_LOTTIE",
      GENERAL_DISPLAY: "GENERAL_DISPLAY",
      GENERAL_START_ACTION: "GENERAL_START_ACTION",
      GENERAL_CONTINUOUS_ACTION: "GENERAL_CONTINUOUS_ACTION",
      GENERAL_COMBO_CLASS: "GENERAL_COMBO_CLASS",
      GENERAL_STOP_ACTION: "GENERAL_STOP_ACTION",
      GENERAL_LOOP: "GENERAL_LOOP",
      STYLE_BOX_SHADOW: "STYLE_BOX_SHADOW",
    };
    e.ActionAppliesTo = {
      ELEMENT: "ELEMENT",
      ELEMENT_CLASS: "ELEMENT_CLASS",
      TRIGGER_ELEMENT: "TRIGGER_ELEMENT",
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.InteractionTypeConsts = void 0);
    e.InteractionTypeConsts = {
      MOUSE_CLICK_INTERACTION: "MOUSE_CLICK_INTERACTION",
      MOUSE_HOVER_INTERACTION: "MOUSE_HOVER_INTERACTION",
      MOUSE_MOVE_INTERACTION: "MOUSE_MOVE_INTERACTION",
      SCROLL_INTO_VIEW_INTERACTION: "SCROLL_INTO_VIEW_INTERACTION",
      SCROLLING_IN_VIEW_INTERACTION: "SCROLLING_IN_VIEW_INTERACTION",
      MOUSE_MOVE_IN_VIEWPORT_INTERACTION: "MOUSE_MOVE_IN_VIEWPORT_INTERACTION",
      PAGE_IS_SCROLLING_INTERACTION: "PAGE_IS_SCROLLING_INTERACTION",
      PAGE_LOAD_INTERACTION: "PAGE_LOAD_INTERACTION",
      PAGE_SCROLLED_INTERACTION: "PAGE_SCROLLED_INTERACTION",
      NAVBAR_INTERACTION: "NAVBAR_INTERACTION",
      DROPDOWN_INTERACTION: "DROPDOWN_INTERACTION",
      ECOMMERCE_CART_INTERACTION: "ECOMMERCE_CART_INTERACTION",
      TAB_INTERACTION: "TAB_INTERACTION",
      SLIDER_INTERACTION: "SLIDER_INTERACTION",
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.IX2_TEST_FRAME_RENDERED =
        e.IX2_MEDIA_QUERIES_DEFINED =
        e.IX2_VIEWPORT_WIDTH_CHANGED =
        e.IX2_ACTION_LIST_PLAYBACK_CHANGED =
        e.IX2_ELEMENT_STATE_CHANGED =
        e.IX2_INSTANCE_REMOVED =
        e.IX2_INSTANCE_STARTED =
        e.IX2_INSTANCE_ADDED =
        e.IX2_PARAMETER_CHANGED =
        e.IX2_ANIMATION_FRAME_CHANGED =
        e.IX2_EVENT_STATE_CHANGED =
        e.IX2_EVENT_LISTENER_ADDED =
        e.IX2_CLEAR_REQUESTED =
        e.IX2_STOP_REQUESTED =
        e.IX2_PLAYBACK_REQUESTED =
        e.IX2_PREVIEW_REQUESTED =
        e.IX2_SESSION_STOPPED =
        e.IX2_SESSION_STARTED =
        e.IX2_SESSION_INITIALIZED =
        e.IX2_RAW_DATA_IMPORTED =
          void 0);
    e.IX2_RAW_DATA_IMPORTED = "IX2_RAW_DATA_IMPORTED";
    e.IX2_SESSION_INITIALIZED = "IX2_SESSION_INITIALIZED";
    e.IX2_SESSION_STARTED = "IX2_SESSION_STARTED";
    e.IX2_SESSION_STOPPED = "IX2_SESSION_STOPPED";
    e.IX2_PREVIEW_REQUESTED = "IX2_PREVIEW_REQUESTED";
    e.IX2_PLAYBACK_REQUESTED = "IX2_PLAYBACK_REQUESTED";
    e.IX2_STOP_REQUESTED = "IX2_STOP_REQUESTED";
    e.IX2_CLEAR_REQUESTED = "IX2_CLEAR_REQUESTED";
    e.IX2_EVENT_LISTENER_ADDED = "IX2_EVENT_LISTENER_ADDED";
    e.IX2_EVENT_STATE_CHANGED = "IX2_EVENT_STATE_CHANGED";
    e.IX2_ANIMATION_FRAME_CHANGED = "IX2_ANIMATION_FRAME_CHANGED";
    e.IX2_PARAMETER_CHANGED = "IX2_PARAMETER_CHANGED";
    e.IX2_INSTANCE_ADDED = "IX2_INSTANCE_ADDED";
    e.IX2_INSTANCE_STARTED = "IX2_INSTANCE_STARTED";
    e.IX2_INSTANCE_REMOVED = "IX2_INSTANCE_REMOVED";
    e.IX2_ELEMENT_STATE_CHANGED = "IX2_ELEMENT_STATE_CHANGED";
    e.IX2_ACTION_LIST_PLAYBACK_CHANGED = "IX2_ACTION_LIST_PLAYBACK_CHANGED";
    e.IX2_VIEWPORT_WIDTH_CHANGED = "IX2_VIEWPORT_WIDTH_CHANGED";
    e.IX2_MEDIA_QUERIES_DEFINED = "IX2_MEDIA_QUERIES_DEFINED";
    e.IX2_TEST_FRAME_RENDERED = "IX2_TEST_FRAME_RENDERED";
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.RENDER_PLUGIN =
        e.RENDER_STYLE =
        e.RENDER_GENERAL =
        e.RENDER_TRANSFORM =
        e.ABSTRACT_NODE =
        e.PLAIN_OBJECT =
        e.HTML_ELEMENT =
        e.PRESERVE_3D =
        e.PARENT =
        e.SIBLINGS =
        e.IMMEDIATE_CHILDREN =
        e.CHILDREN =
        e.BAR_DELIMITER =
        e.COLON_DELIMITER =
        e.COMMA_DELIMITER =
        e.AUTO =
        e.WILL_CHANGE =
        e.FLEX =
        e.DISPLAY =
        e.COLOR =
        e.BORDER_COLOR =
        e.BACKGROUND =
        e.BACKGROUND_COLOR =
        e.HEIGHT =
        e.WIDTH =
        e.FILTER =
        e.OPACITY =
        e.SKEW_Y =
        e.SKEW_X =
        e.SKEW =
        e.ROTATE_Z =
        e.ROTATE_Y =
        e.ROTATE_X =
        e.SCALE_3D =
        e.SCALE_Z =
        e.SCALE_Y =
        e.SCALE_X =
        e.TRANSLATE_3D =
        e.TRANSLATE_Z =
        e.TRANSLATE_Y =
        e.TRANSLATE_X =
        e.TRANSFORM =
        e.CONFIG_UNIT =
        e.CONFIG_Z_UNIT =
        e.CONFIG_Y_UNIT =
        e.CONFIG_X_UNIT =
        e.CONFIG_VALUE =
        e.CONFIG_Z_VALUE =
        e.CONFIG_Y_VALUE =
        e.CONFIG_X_VALUE =
        e.BOUNDARY_SELECTOR =
        e.W_MOD_IX =
        e.W_MOD_JS =
        e.WF_PAGE =
        e.IX2_ID_DELIMITER =
          void 0);
    e.IX2_ID_DELIMITER = "|";
    e.WF_PAGE = "data-wf-page";
    e.W_MOD_JS = "w-mod-js";
    e.W_MOD_IX = "w-mod-ix";
    e.BOUNDARY_SELECTOR = ".w-dyn-item";
    e.CONFIG_X_VALUE = "xValue";
    e.CONFIG_Y_VALUE = "yValue";
    e.CONFIG_Z_VALUE = "zValue";
    e.CONFIG_VALUE = "value";
    e.CONFIG_X_UNIT = "xUnit";
    e.CONFIG_Y_UNIT = "yUnit";
    e.CONFIG_Z_UNIT = "zUnit";
    e.CONFIG_UNIT = "unit";
    e.TRANSFORM = "transform";
    e.TRANSLATE_X = "translateX";
    e.TRANSLATE_Y = "translateY";
    e.TRANSLATE_Z = "translateZ";
    e.TRANSLATE_3D = "translate3d";
    e.SCALE_X = "scaleX";
    e.SCALE_Y = "scaleY";
    e.SCALE_Z = "scaleZ";
    e.SCALE_3D = "scale3d";
    e.ROTATE_X = "rotateX";
    e.ROTATE_Y = "rotateY";
    e.ROTATE_Z = "rotateZ";
    e.SKEW = "skew";
    e.SKEW_X = "skewX";
    e.SKEW_Y = "skewY";
    e.OPACITY = "opacity";
    e.FILTER = "filter";
    e.WIDTH = "width";
    e.HEIGHT = "height";
    e.BACKGROUND_COLOR = "backgroundColor";
    e.BACKGROUND = "background";
    e.BORDER_COLOR = "borderColor";
    e.COLOR = "color";
    e.DISPLAY = "display";
    e.FLEX = "flex";
    e.WILL_CHANGE = "willChange";
    e.AUTO = "AUTO";
    e.COMMA_DELIMITER = ",";
    e.COLON_DELIMITER = ":";
    e.BAR_DELIMITER = "|";
    e.CHILDREN = "CHILDREN";
    e.IMMEDIATE_CHILDREN = "IMMEDIATE_CHILDREN";
    e.SIBLINGS = "SIBLINGS";
    e.PARENT = "PARENT";
    e.PRESERVE_3D = "preserve-3d";
    e.HTML_ELEMENT = "HTML_ELEMENT";
    e.PLAIN_OBJECT = "PLAIN_OBJECT";
    e.ABSTRACT_NODE = "ABSTRACT_NODE";
    e.RENDER_TRANSFORM = "RENDER_TRANSFORM";
    e.RENDER_GENERAL = "RENDER_GENERAL";
    e.RENDER_STYLE = "RENDER_STYLE";
    e.RENDER_PLUGIN = "RENDER_PLUGIN";
  },
  function (t, e, n) {
    "use strict";
    var r,
      i = n(0)(n(27)),
      o = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.ixRequest = void 0);
    var a = o(n(28)),
      u = n(2),
      c = n(19),
      s = u.IX2EngineActionTypes,
      f = s.IX2_PREVIEW_REQUESTED,
      l = s.IX2_PLAYBACK_REQUESTED,
      d = s.IX2_STOP_REQUESTED,
      p = s.IX2_CLEAR_REQUESTED,
      v = { preview: {}, playback: {}, stop: {}, clear: {} },
      h = Object.create(
        null,
        ((r = {}),
        (0, i.default)(r, f, { value: "preview" }),
        (0, i.default)(r, l, { value: "playback" }),
        (0, i.default)(r, d, { value: "stop" }),
        (0, i.default)(r, p, { value: "clear" }),
        r)
      );
    e.ixRequest = function () {
      var t =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : v,
        e = arguments.length > 1 ? arguments[1] : void 0;
      if (e.type in h) {
        var n = [h[e.type]];
        return (0, c.setIn)(t, [n], (0, a.default)({}, e.payload));
      }
      return t;
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.ixSession = void 0);
    var r = n(2),
      i = n(19),
      o = r.IX2EngineActionTypes,
      a = o.IX2_SESSION_INITIALIZED,
      u = o.IX2_SESSION_STARTED,
      c = o.IX2_TEST_FRAME_RENDERED,
      s = o.IX2_SESSION_STOPPED,
      f = o.IX2_EVENT_LISTENER_ADDED,
      l = o.IX2_EVENT_STATE_CHANGED,
      d = o.IX2_ANIMATION_FRAME_CHANGED,
      p = o.IX2_ACTION_LIST_PLAYBACK_CHANGED,
      v = o.IX2_VIEWPORT_WIDTH_CHANGED,
      h = o.IX2_MEDIA_QUERIES_DEFINED,
      E = {
        active: !1,
        tick: 0,
        eventListeners: [],
        eventState: {},
        playbackState: {},
        viewportWidth: 0,
        mediaQueryKey: null,
        hasBoundaryNodes: !1,
        hasDefinedMediaQueries: !1,
      };
    e.ixSession = function () {
      var t =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : E,
        e = arguments.length > 1 ? arguments[1] : void 0;
      switch (e.type) {
        case a:
          var n = e.payload.hasBoundaryNodes;
          return (0, i.set)(t, "hasBoundaryNodes", n);
        case u:
          return (0, i.set)(t, "active", !0);
        case c:
          var r = e.payload.step,
            o = void 0 === r ? 20 : r;
          return (0, i.set)(t, "tick", t.tick + o);
        case s:
          return E;
        case d:
          var g = e.payload.now;
          return (0, i.set)(t, "tick", g);
        case f:
          var _ = (0, i.addLast)(t.eventListeners, e.payload);
          return (0, i.set)(t, "eventListeners", _);
        case l:
          var y = e.payload,
            m = y.stateKey,
            I = y.newState;
          return (0, i.setIn)(t, ["eventState", m], I);
        case p:
          var b = e.payload,
            T = b.actionListId,
            O = b.isPlaying;
          return (0, i.setIn)(t, ["playbackState", T], O);
        case v:
          for (
            var w = e.payload,
              A = w.width,
              S = w.mediaQueries,
              x = S.length,
              R = null,
              C = 0;
            C < x;
            C++
          ) {
            var N = S[C],
              L = N.key,
              D = N.min,
              P = N.max;
            if (A >= D && A <= P) {
              R = L;
              break;
            }
          }
          return (0, i.merge)(t, { viewportWidth: A, mediaQueryKey: R });
        case h:
          return (0, i.set)(t, "hasDefinedMediaQueries", !0);
        default:
          return t;
      }
    };
  },
  function (t, e, n) {
    var r = n(178),
      i = n(230),
      o = n(101);
    t.exports = function (t) {
      var e = i(t);
      return 1 == e.length && e[0][2]
        ? o(e[0][0], e[0][1])
        : function (n) {
            return n === t || r(n, t, e);
          };
    };
  },
  function (t, e, n) {
    var r = n(87),
      i = n(91),
      o = 1,
      a = 2;
    t.exports = function (t, e, n, u) {
      var c = n.length,
        s = c,
        f = !u;
      if (null == t) return !s;
      for (t = Object(t); c--; ) {
        var l = n[c];
        if (f && l[2] ? l[1] !== t[l[0]] : !(l[0] in t)) return !1;
      }
      for (; ++c < s; ) {
        var d = (l = n[c])[0],
          p = t[d],
          v = l[1];
        if (f && l[2]) {
          if (void 0 === p && !(d in t)) return !1;
        } else {
          var h = new r();
          if (u) var E = u(p, v, d, t, e, h);
          if (!(void 0 === E ? i(v, p, o | a, u, h) : E)) return !1;
        }
      }
      return !0;
    };
  },
  function (t, e) {
    t.exports = function () {
      (this.__data__ = []), (this.size = 0);
    };
  },
  function (t, e, n) {
    var r = n(30),
      i = Array.prototype.splice;
    t.exports = function (t) {
      var e = this.__data__,
        n = r(e, t);
      return !(
        n < 0 || (n == e.length - 1 ? e.pop() : i.call(e, n, 1), --this.size, 0)
      );
    };
  },
  function (t, e, n) {
    var r = n(30);
    t.exports = function (t) {
      var e = this.__data__,
        n = r(e, t);
      return n < 0 ? void 0 : e[n][1];
    };
  },
  function (t, e, n) {
    var r = n(30);
    t.exports = function (t) {
      return r(this.__data__, t) > -1;
    };
  },
  function (t, e, n) {
    var r = n(30);
    t.exports = function (t, e) {
      var n = this.__data__,
        i = r(n, t);
      return i < 0 ? (++this.size, n.push([t, e])) : (n[i][1] = e), this;
    };
  },
  function (t, e, n) {
    var r = n(29);
    t.exports = function () {
      (this.__data__ = new r()), (this.size = 0);
    };
  },
  function (t, e) {
    t.exports = function (t) {
      var e = this.__data__,
        n = e.delete(t);
      return (this.size = e.size), n;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return this.__data__.get(t);
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return this.__data__.has(t);
    };
  },
  function (t, e, n) {
    var r = n(29),
      i = n(46),
      o = n(47),
      a = 200;
    t.exports = function (t, e) {
      var n = this.__data__;
      if (n instanceof r) {
        var u = n.__data__;
        if (!i || u.length < a - 1)
          return u.push([t, e]), (this.size = ++n.size), this;
        n = this.__data__ = new o(u);
      }
      return n.set(t, e), (this.size = n.size), this;
    };
  },
  function (t, e, n) {
    var r = n(88),
      i = n(192),
      o = n(6),
      a = n(90),
      u = /^\[object .+?Constructor\]$/,
      c = Function.prototype,
      s = Object.prototype,
      f = c.toString,
      l = s.hasOwnProperty,
      d = RegExp(
        "^" +
          f
            .call(l)
            .replace(/[\\^$.*+?()[\]{}|]/g, "\\$&")
            .replace(
              /hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,
              "$1.*?"
            ) +
          "$"
      );
    t.exports = function (t) {
      return !(!o(t) || i(t)) && (r(t) ? d : u).test(a(t));
    };
  },
  function (t, e, n) {
    var r = n(20),
      i = Object.prototype,
      o = i.hasOwnProperty,
      a = i.toString,
      u = r ? r.toStringTag : void 0;
    t.exports = function (t) {
      var e = o.call(t, u),
        n = t[u];
      try {
        t[u] = void 0;
        var r = !0;
      } catch (t) {}
      var i = a.call(t);
      return r && (e ? (t[u] = n) : delete t[u]), i;
    };
  },
  function (t, e) {
    var n = Object.prototype.toString;
    t.exports = function (t) {
      return n.call(t);
    };
  },
  function (t, e, n) {
    var r,
      i = n(193),
      o = (r = /[^.]+$/.exec((i && i.keys && i.keys.IE_PROTO) || ""))
        ? "Symbol(src)_1." + r
        : "";
    t.exports = function (t) {
      return !!o && o in t;
    };
  },
  function (t, e, n) {
    var r = n(5)["__core-js_shared__"];
    t.exports = r;
  },
  function (t, e) {
    t.exports = function (t, e) {
      return null == t ? void 0 : t[e];
    };
  },
  function (t, e, n) {
    var r = n(196),
      i = n(29),
      o = n(46);
    t.exports = function () {
      (this.size = 0),
        (this.__data__ = {
          hash: new r(),
          map: new (o || i)(),
          string: new r(),
        });
    };
  },
  function (t, e, n) {
    var r = n(197),
      i = n(198),
      o = n(199),
      a = n(200),
      u = n(201);
    function c(t) {
      var e = -1,
        n = null == t ? 0 : t.length;
      for (this.clear(); ++e < n; ) {
        var r = t[e];
        this.set(r[0], r[1]);
      }
    }
    (c.prototype.clear = r),
      (c.prototype.delete = i),
      (c.prototype.get = o),
      (c.prototype.has = a),
      (c.prototype.set = u),
      (t.exports = c);
  },
  function (t, e, n) {
    var r = n(31);
    t.exports = function () {
      (this.__data__ = r ? r(null) : {}), (this.size = 0);
    };
  },
  function (t, e) {
    t.exports = function (t) {
      var e = this.has(t) && delete this.__data__[t];
      return (this.size -= e ? 1 : 0), e;
    };
  },
  function (t, e, n) {
    var r = n(31),
      i = "__lodash_hash_undefined__",
      o = Object.prototype.hasOwnProperty;
    t.exports = function (t) {
      var e = this.__data__;
      if (r) {
        var n = e[t];
        return n === i ? void 0 : n;
      }
      return o.call(e, t) ? e[t] : void 0;
    };
  },
  function (t, e, n) {
    var r = n(31),
      i = Object.prototype.hasOwnProperty;
    t.exports = function (t) {
      var e = this.__data__;
      return r ? void 0 !== e[t] : i.call(e, t);
    };
  },
  function (t, e, n) {
    var r = n(31),
      i = "__lodash_hash_undefined__";
    t.exports = function (t, e) {
      var n = this.__data__;
      return (
        (this.size += this.has(t) ? 0 : 1),
        (n[t] = r && void 0 === e ? i : e),
        this
      );
    };
  },
  function (t, e, n) {
    var r = n(32);
    t.exports = function (t) {
      var e = r(this, t).delete(t);
      return (this.size -= e ? 1 : 0), e;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      var e = typeof t;
      return "string" == e || "number" == e || "symbol" == e || "boolean" == e
        ? "__proto__" !== t
        : null === t;
    };
  },
  function (t, e, n) {
    var r = n(32);
    t.exports = function (t) {
      return r(this, t).get(t);
    };
  },
  function (t, e, n) {
    var r = n(32);
    t.exports = function (t) {
      return r(this, t).has(t);
    };
  },
  function (t, e, n) {
    var r = n(32);
    t.exports = function (t, e) {
      var n = r(this, t),
        i = n.size;
      return n.set(t, e), (this.size += n.size == i ? 0 : 1), this;
    };
  },
  function (t, e, n) {
    var r = n(87),
      i = n(92),
      o = n(213),
      a = n(217),
      u = n(55),
      c = n(1),
      s = n(49),
      f = n(51),
      l = 1,
      d = "[object Arguments]",
      p = "[object Array]",
      v = "[object Object]",
      h = Object.prototype.hasOwnProperty;
    t.exports = function (t, e, n, E, g, _) {
      var y = c(t),
        m = c(e),
        I = y ? p : u(t),
        b = m ? p : u(e),
        T = (I = I == d ? v : I) == v,
        O = (b = b == d ? v : b) == v,
        w = I == b;
      if (w && s(t)) {
        if (!s(e)) return !1;
        (y = !0), (T = !1);
      }
      if (w && !T)
        return (
          _ || (_ = new r()),
          y || f(t) ? i(t, e, n, E, g, _) : o(t, e, I, n, E, g, _)
        );
      if (!(n & l)) {
        var A = T && h.call(t, "__wrapped__"),
          S = O && h.call(e, "__wrapped__");
        if (A || S) {
          var x = A ? t.value() : t,
            R = S ? e.value() : e;
          return _ || (_ = new r()), g(x, R, n, E, _);
        }
      }
      return !!w && (_ || (_ = new r()), a(t, e, n, E, g, _));
    };
  },
  function (t, e, n) {
    var r = n(47),
      i = n(209),
      o = n(210);
    function a(t) {
      var e = -1,
        n = null == t ? 0 : t.length;
      for (this.__data__ = new r(); ++e < n; ) this.add(t[e]);
    }
    (a.prototype.add = a.prototype.push = i),
      (a.prototype.has = o),
      (t.exports = a);
  },
  function (t, e) {
    var n = "__lodash_hash_undefined__";
    t.exports = function (t) {
      return this.__data__.set(t, n), this;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return this.__data__.has(t);
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      for (var n = -1, r = null == t ? 0 : t.length; ++n < r; )
        if (e(t[n], n, t)) return !0;
      return !1;
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      return t.has(e);
    };
  },
  function (t, e, n) {
    var r = n(20),
      i = n(214),
      o = n(45),
      a = n(92),
      u = n(215),
      c = n(216),
      s = 1,
      f = 2,
      l = "[object Boolean]",
      d = "[object Date]",
      p = "[object Error]",
      v = "[object Map]",
      h = "[object Number]",
      E = "[object RegExp]",
      g = "[object Set]",
      _ = "[object String]",
      y = "[object Symbol]",
      m = "[object ArrayBuffer]",
      I = "[object DataView]",
      b = r ? r.prototype : void 0,
      T = b ? b.valueOf : void 0;
    t.exports = function (t, e, n, r, b, O, w) {
      switch (n) {
        case I:
          if (t.byteLength != e.byteLength || t.byteOffset != e.byteOffset)
            return !1;
          (t = t.buffer), (e = e.buffer);
        case m:
          return !(t.byteLength != e.byteLength || !O(new i(t), new i(e)));
        case l:
        case d:
        case h:
          return o(+t, +e);
        case p:
          return t.name == e.name && t.message == e.message;
        case E:
        case _:
          return t == e + "";
        case v:
          var A = u;
        case g:
          var S = r & s;
          if ((A || (A = c), t.size != e.size && !S)) return !1;
          var x = w.get(t);
          if (x) return x == e;
          (r |= f), w.set(t, e);
          var R = a(A(t), A(e), r, b, O, w);
          return w.delete(t), R;
        case y:
          if (T) return T.call(t) == T.call(e);
      }
      return !1;
    };
  },
  function (t, e, n) {
    var r = n(5).Uint8Array;
    t.exports = r;
  },
  function (t, e) {
    t.exports = function (t) {
      var e = -1,
        n = Array(t.size);
      return (
        t.forEach(function (t, r) {
          n[++e] = [r, t];
        }),
        n
      );
    };
  },
  function (t, e) {
    t.exports = function (t) {
      var e = -1,
        n = Array(t.size);
      return (
        t.forEach(function (t) {
          n[++e] = t;
        }),
        n
      );
    };
  },
  function (t, e, n) {
    var r = n(218),
      i = 1,
      o = Object.prototype.hasOwnProperty;
    t.exports = function (t, e, n, a, u, c) {
      var s = n & i,
        f = r(t),
        l = f.length;
      if (l != r(e).length && !s) return !1;
      for (var d = l; d--; ) {
        var p = f[d];
        if (!(s ? p in e : o.call(e, p))) return !1;
      }
      var v = c.get(t),
        h = c.get(e);
      if (v && h) return v == e && h == t;
      var E = !0;
      c.set(t, e), c.set(e, t);
      for (var g = s; ++d < l; ) {
        var _ = t[(p = f[d])],
          y = e[p];
        if (a) var m = s ? a(y, _, p, e, t, c) : a(_, y, p, t, e, c);
        if (!(void 0 === m ? _ === y || u(_, y, n, a, c) : m)) {
          E = !1;
          break;
        }
        g || (g = "constructor" == p);
      }
      if (E && !g) {
        var I = t.constructor,
          b = e.constructor;
        I != b &&
          "constructor" in t &&
          "constructor" in e &&
          !(
            "function" == typeof I &&
            I instanceof I &&
            "function" == typeof b &&
            b instanceof b
          ) &&
          (E = !1);
      }
      return c.delete(t), c.delete(e), E;
    };
  },
  function (t, e, n) {
    var r = n(93),
      i = n(94),
      o = n(33);
    t.exports = function (t) {
      return r(t, o, i);
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      for (var n = -1, r = null == t ? 0 : t.length, i = 0, o = []; ++n < r; ) {
        var a = t[n];
        e(a, n, t) && (o[i++] = a);
      }
      return o;
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      for (var n = -1, r = Array(t); ++n < t; ) r[n] = e(n);
      return r;
    };
  },
  function (t, e, n) {
    var r = n(11),
      i = n(9),
      o = "[object Arguments]";
    t.exports = function (t) {
      return i(t) && r(t) == o;
    };
  },
  function (t, e) {
    t.exports = function () {
      return !1;
    };
  },
  function (t, e, n) {
    var r = n(11),
      i = n(52),
      o = n(9),
      a = {};
    (a["[object Float32Array]"] =
      a["[object Float64Array]"] =
      a["[object Int8Array]"] =
      a["[object Int16Array]"] =
      a["[object Int32Array]"] =
      a["[object Uint8Array]"] =
      a["[object Uint8ClampedArray]"] =
      a["[object Uint16Array]"] =
      a["[object Uint32Array]"] =
        !0),
      (a["[object Arguments]"] =
        a["[object Array]"] =
        a["[object ArrayBuffer]"] =
        a["[object Boolean]"] =
        a["[object DataView]"] =
        a["[object Date]"] =
        a["[object Error]"] =
        a["[object Function]"] =
        a["[object Map]"] =
        a["[object Number]"] =
        a["[object Object]"] =
        a["[object RegExp]"] =
        a["[object Set]"] =
        a["[object String]"] =
        a["[object WeakMap]"] =
          !1),
      (t.exports = function (t) {
        return o(t) && i(t.length) && !!a[r(t)];
      });
  },
  function (t, e) {
    t.exports = function (t) {
      return function (e) {
        return t(e);
      };
    };
  },
  function (t, e, n) {
    (function (t) {
      var r = n(89),
        i = e && !e.nodeType && e,
        o = i && "object" == typeof t && t && !t.nodeType && t,
        a = o && o.exports === i && r.process,
        u = (function () {
          try {
            var t = o && o.require && o.require("util").types;
            return t || (a && a.binding && a.binding("util"));
          } catch (t) {}
        })();
      t.exports = u;
    }).call(this, n(97)(t));
  },
  function (t, e, n) {
    var r = n(98)(Object.keys, Object);
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(8)(n(5), "DataView");
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(8)(n(5), "Promise");
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(8)(n(5), "Set");
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(100),
      i = n(33);
    t.exports = function (t) {
      for (var e = i(t), n = e.length; n--; ) {
        var o = e[n],
          a = t[o];
        e[n] = [o, a, r(a)];
      }
      return e;
    };
  },
  function (t, e, n) {
    var r = n(91),
      i = n(56),
      o = n(237),
      a = n(58),
      u = n(100),
      c = n(101),
      s = n(21),
      f = 1,
      l = 2;
    t.exports = function (t, e) {
      return a(t) && u(e)
        ? c(s(t), e)
        : function (n) {
            var a = i(n, t);
            return void 0 === a && a === e ? o(n, t) : r(e, a, f | l);
          };
    };
  },
  function (t, e, n) {
    var r = n(233),
      i =
        /[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g,
      o = /\\(\\)?/g,
      a = r(function (t) {
        var e = [];
        return (
          46 === t.charCodeAt(0) && e.push(""),
          t.replace(i, function (t, n, r, i) {
            e.push(r ? i.replace(o, "$1") : n || t);
          }),
          e
        );
      });
    t.exports = a;
  },
  function (t, e, n) {
    var r = n(234),
      i = 500;
    t.exports = function (t) {
      var e = r(t, function (t) {
          return n.size === i && n.clear(), t;
        }),
        n = e.cache;
      return e;
    };
  },
  function (t, e, n) {
    var r = n(47),
      i = "Expected a function";
    function o(t, e) {
      if ("function" != typeof t || (null != e && "function" != typeof e))
        throw new TypeError(i);
      var n = function () {
        var r = arguments,
          i = e ? e.apply(this, r) : r[0],
          o = n.cache;
        if (o.has(i)) return o.get(i);
        var a = t.apply(this, r);
        return (n.cache = o.set(i, a) || o), a;
      };
      return (n.cache = new (o.Cache || r)()), n;
    }
    (o.Cache = r), (t.exports = o);
  },
  function (t, e, n) {
    var r = n(236);
    t.exports = function (t) {
      return null == t ? "" : r(t);
    };
  },
  function (t, e, n) {
    var r = n(20),
      i = n(102),
      o = n(1),
      a = n(36),
      u = 1 / 0,
      c = r ? r.prototype : void 0,
      s = c ? c.toString : void 0;
    t.exports = function t(e) {
      if ("string" == typeof e) return e;
      if (o(e)) return i(e, t) + "";
      if (a(e)) return s ? s.call(e) : "";
      var n = e + "";
      return "0" == n && 1 / e == -u ? "-0" : n;
    };
  },
  function (t, e, n) {
    var r = n(238),
      i = n(239);
    t.exports = function (t, e) {
      return null != t && i(t, e, r);
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      return null != t && e in Object(t);
    };
  },
  function (t, e, n) {
    var r = n(35),
      i = n(34),
      o = n(1),
      a = n(50),
      u = n(52),
      c = n(21);
    t.exports = function (t, e, n) {
      for (var s = -1, f = (e = r(e, t)).length, l = !1; ++s < f; ) {
        var d = c(e[s]);
        if (!(l = null != t && n(t, d))) break;
        t = t[d];
      }
      return l || ++s != f
        ? l
        : !!(f = null == t ? 0 : t.length) && u(f) && a(d, f) && (o(t) || i(t));
    };
  },
  function (t, e, n) {
    var r = n(103),
      i = n(241),
      o = n(58),
      a = n(21);
    t.exports = function (t) {
      return o(t) ? r(a(t)) : i(t);
    };
  },
  function (t, e, n) {
    var r = n(57);
    t.exports = function (t) {
      return function (e) {
        return r(e, t);
      };
    };
  },
  function (t, e, n) {
    var r = n(104),
      i = n(7),
      o = n(105),
      a = Math.max;
    t.exports = function (t, e, n) {
      var u = null == t ? 0 : t.length;
      if (!u) return -1;
      var c = null == n ? 0 : o(n);
      return c < 0 && (c = a(u + c, 0)), r(t, i(e, 3), c);
    };
  },
  function (t, e, n) {
    var r = n(60),
      i = 1 / 0,
      o = 1.7976931348623157e308;
    t.exports = function (t) {
      return t
        ? (t = r(t)) === i || t === -i
          ? (t < 0 ? -1 : 1) * o
          : t == t
          ? t
          : 0
        : 0 === t
        ? t
        : 0;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      if (Array.isArray(t)) {
        for (var e = 0, n = new Array(t.length); e < t.length; e++) n[e] = t[e];
        return n;
      }
    };
  },
  function (t, e) {
    t.exports = function (t) {
      if (
        Symbol.iterator in Object(t) ||
        "[object Arguments]" === Object.prototype.toString.call(t)
      )
        return Array.from(t);
    };
  },
  function (t, e) {
    t.exports = function () {
      throw new TypeError("Invalid attempt to spread non-iterable instance");
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.createElementState = I),
      (e.mergeActionState = b),
      (e.ixElements = void 0);
    var r = n(19),
      i = n(2),
      o = i.IX2EngineConstants,
      a = (o.HTML_ELEMENT, o.PLAIN_OBJECT),
      u = (o.ABSTRACT_NODE, o.CONFIG_X_VALUE),
      c = o.CONFIG_Y_VALUE,
      s = o.CONFIG_Z_VALUE,
      f = o.CONFIG_VALUE,
      l = o.CONFIG_X_UNIT,
      d = o.CONFIG_Y_UNIT,
      p = o.CONFIG_Z_UNIT,
      v = o.CONFIG_UNIT,
      h = i.IX2EngineActionTypes,
      E = h.IX2_SESSION_STOPPED,
      g = h.IX2_INSTANCE_ADDED,
      _ = h.IX2_ELEMENT_STATE_CHANGED,
      y = {},
      m = "refState";
    function I(t, e, n, i, o) {
      var u =
        n === a ? (0, r.getIn)(o, ["config", "target", "objectId"]) : null;
      return (0, r.mergeIn)(t, [i], { id: i, ref: e, refId: u, refType: n });
    }
    function b(t, e, n, i, o) {
      var a = (function (t) {
          var e = t.config;
          return T.reduce(function (t, n) {
            var r = n[0],
              i = n[1],
              o = e[r],
              a = e[i];
            return null != o && null != a && (t[i] = a), t;
          }, {});
        })(o),
        u = [e, m, n];
      return (0, r.mergeIn)(t, u, i, a);
    }
    e.ixElements = function () {
      var t =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : y,
        e = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
      switch (e.type) {
        case E:
          return y;
        case g:
          var n = e.payload,
            i = n.elementId,
            o = n.element,
            a = n.origin,
            u = n.actionItem,
            c = n.refType,
            s = u.actionTypeId,
            f = t;
          return (
            (0, r.getIn)(f, [i, o]) !== o && (f = I(f, o, c, i, u)),
            b(f, i, s, a, u)
          );
        case _:
          var l = e.payload;
          return b(t, l.elementId, l.actionTypeId, l.current, l.actionItem);
        default:
          return t;
      }
    };
    var T = [
      [u, l],
      [c, d],
      [s, p],
      [f, v],
    ];
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.clearPlugin =
        e.renderPlugin =
        e.createPluginInstance =
        e.getPluginDestination =
        e.getPluginOrigin =
        e.getPluginDuration =
        e.getPluginConfig =
          void 0);
    e.getPluginConfig = function (t) {
      return t.value;
    };
    e.getPluginDuration = function (t, e) {
      if ("auto" !== e.config.duration) return null;
      var n = parseFloat(t.getAttribute("data-duration"));
      return n > 0
        ? 1e3 * n
        : 1e3 * parseFloat(t.getAttribute("data-default-duration"));
    };
    e.getPluginOrigin = function (t) {
      return t || { value: 0 };
    };
    e.getPluginDestination = function (t) {
      return { value: t.value };
    };
    e.createPluginInstance = function (t) {
      var e = window.Webflow.require("lottie").createInstance(t);
      return e.stop(), e.setSubframe(!0), e;
    };
    e.renderPlugin = function (t, e, n) {
      if (t) {
        var r = e[n.actionTypeId].value / 100;
        t.goToFrame(t.frames * r);
      }
    };
    e.clearPlugin = function (t) {
      window.Webflow.require("lottie").createInstance(t).stop();
    };
  },
  function (t, e, n) {
    "use strict";
    var r,
      i,
      o,
      a = n(0),
      u = a(n(22)),
      c = a(n(27)),
      s = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.getInstanceId = function () {
        return "i" + vt++;
      }),
      (e.getElementId = function (t, e) {
        for (var n in t) {
          var r = t[n];
          if (r && r.ref === e) return r.id;
        }
        return "e" + ht++;
      }),
      (e.reifyState = function () {
        var t =
            arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
          e = t.events,
          n = t.actionLists,
          r = t.site,
          i = (0, l.default)(
            e,
            function (t, e) {
              var n = e.eventTypeId;
              return t[n] || (t[n] = {}), (t[n][e.id] = e), t;
            },
            {}
          ),
          o = r && r.mediaQueries,
          a = [];
        o
          ? (a = o.map(function (t) {
              return t.key;
            }))
          : ((o = []), console.warn("IX2 missing mediaQueries in site data"));
        return {
          ixData: {
            events: e,
            actionLists: n,
            eventTypeMap: i,
            mediaQueries: o,
            mediaQueryKeys: a,
          },
        };
      }),
      (e.observeStore = function (t) {
        var e = t.store,
          n = t.select,
          r = t.onChange,
          i = t.comparator,
          o = void 0 === i ? Et : i,
          a = e.getState,
          u = (0, e.subscribe)(function () {
            var t = n(a());
            if (null == t) return void u();
            o(t, c) || r((c = t), e);
          }),
          c = n(a());
        return u;
      }),
      (e.getAffectedElements = _t),
      (e.getComputedStyle = function (t) {
        var e = t.element,
          n = t.actionItem;
        if (!_.IS_BROWSER_ENV) return {};
        switch (n.actionTypeId) {
          case it:
          case ot:
          case at:
          case ut:
          case ct:
            return window.getComputedStyle(e);
          default:
            return {};
        }
      }),
      (e.getInstanceOrigin = function (t) {
        var e =
            arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {},
          n =
            arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : {},
          r = arguments.length > 3 ? arguments[3] : void 0,
          i = (arguments.length > 4 ? arguments[4] : void 0).getStyle,
          o = r.actionTypeId,
          a = r.config;
        if ((0, g.isPluginType)(o)) return (0, g.getPluginOrigin)(o)(e[o]);
        switch (o) {
          case Z:
          case J:
          case tt:
          case et:
            return e[o] || Tt[o];
          case rt:
            return mt(e[o], r.config.filters);
          case nt:
            return { value: (0, f.default)(parseFloat(i(t, C)), 1) };
          case it:
            var u,
              c,
              s = i(t, L),
              l = i(t, D);
            return (
              (u =
                a.widthUnit === W
                  ? yt.test(s)
                    ? parseFloat(s)
                    : parseFloat(n.width)
                  : (0, f.default)(parseFloat(s), parseFloat(n.width))),
              (c =
                a.heightUnit === W
                  ? yt.test(l)
                    ? parseFloat(l)
                    : parseFloat(n.height)
                  : (0, f.default)(parseFloat(l), parseFloat(n.height))),
              { widthValue: u, heightValue: c }
            );
          case ot:
          case at:
          case ut:
            return (function (t) {
              var e = t.element,
                n = t.actionTypeId,
                r = t.computedStyle,
                i = t.getStyle,
                o = lt[n],
                a = i(e, o),
                u = St.test(a) ? a : r[o],
                c = (function (t, e) {
                  var n = t.exec(e);
                  return n ? n[1] : "";
                })(xt, u).split(H);
              return {
                rValue: (0, f.default)(parseInt(c[0], 10), 255),
                gValue: (0, f.default)(parseInt(c[1], 10), 255),
                bValue: (0, f.default)(parseInt(c[2], 10), 255),
                aValue: (0, f.default)(parseFloat(c[3]), 1),
              };
            })({ element: t, actionTypeId: o, computedStyle: n, getStyle: i });
          case ct:
            return { value: (0, f.default)(i(t, U), n.display) };
          case st:
            return e[o] || { value: 0 };
          default:
            return;
        }
      }),
      (e.getDestinationValues = function (t) {
        var e = t.element,
          n = t.actionItem,
          r = t.elementApi,
          i = n.actionTypeId;
        if ((0, g.isPluginType)(i))
          return (0, g.getPluginDestination)(i)(n.config);
        switch (i) {
          case Z:
          case J:
          case tt:
          case et:
            var o = n.config,
              a = o.xValue,
              u = o.yValue,
              c = o.zValue;
            return { xValue: a, yValue: u, zValue: c };
          case it:
            var s = r.getStyle,
              f = r.setStyle,
              l = r.getProperty,
              d = n.config,
              p = d.widthUnit,
              v = d.heightUnit,
              h = n.config,
              E = h.widthValue,
              y = h.heightValue;
            if (!_.IS_BROWSER_ENV) return { widthValue: E, heightValue: y };
            if (p === W) {
              var m = s(e, L);
              f(e, L, ""), (E = l(e, "offsetWidth")), f(e, L, m);
            }
            if (v === W) {
              var I = s(e, D);
              f(e, D, ""), (y = l(e, "offsetHeight")), f(e, D, I);
            }
            return { widthValue: E, heightValue: y };
          case ot:
          case at:
          case ut:
            var b = n.config,
              T = b.rValue,
              O = b.gValue,
              w = b.bValue,
              A = b.aValue;
            return { rValue: T, gValue: O, bValue: w, aValue: A };
          case rt:
            return n.config.filters.reduce(It, {});
          default:
            var S = n.config.value;
            return { value: S };
        }
      }),
      (e.getRenderType = bt),
      (e.getStyleProp = function (t, e) {
        return t === Q ? e.replace("STYLE_", "").toLowerCase() : null;
      }),
      (e.renderHTMLElement = function (t, e, n, r, i, o, a, u, c) {
        switch (u) {
          case Y:
            return (function (t, e, n, r, i) {
              var o = At.map(function (t) {
                  var n = Tt[t],
                    r = e[t] || {},
                    i = r.xValue,
                    o = void 0 === i ? n.xValue : i,
                    a = r.yValue,
                    u = void 0 === a ? n.yValue : a,
                    c = r.zValue,
                    s = void 0 === c ? n.zValue : c,
                    f = r.xUnit,
                    l = void 0 === f ? "" : f,
                    d = r.yUnit,
                    p = void 0 === d ? "" : d,
                    v = r.zUnit,
                    h = void 0 === v ? "" : v;
                  switch (t) {
                    case Z:
                      return ""
                        .concat(b, "(")
                        .concat(o)
                        .concat(l, ", ")
                        .concat(u)
                        .concat(p, ", ")
                        .concat(s)
                        .concat(h, ")");
                    case J:
                      return ""
                        .concat(T, "(")
                        .concat(o)
                        .concat(l, ", ")
                        .concat(u)
                        .concat(p, ", ")
                        .concat(s)
                        .concat(h, ")");
                    case tt:
                      return ""
                        .concat(O, "(")
                        .concat(o)
                        .concat(l, ") ")
                        .concat(w, "(")
                        .concat(u)
                        .concat(p, ") ")
                        .concat(A, "(")
                        .concat(s)
                        .concat(h, ")");
                    case et:
                      return ""
                        .concat(S, "(")
                        .concat(o)
                        .concat(l, ", ")
                        .concat(u)
                        .concat(p, ")");
                    default:
                      return "";
                  }
                }).join(" "),
                a = i.setStyle;
              Rt(t, _.TRANSFORM_PREFIXED, i),
                a(t, _.TRANSFORM_PREFIXED, o),
                (u = r),
                (c = n),
                (s = u.actionTypeId),
                (f = c.xValue),
                (l = c.yValue),
                (d = c.zValue),
                ((s === Z && void 0 !== d) ||
                  (s === J && void 0 !== d) ||
                  (s === tt && (void 0 !== f || void 0 !== l))) &&
                  a(t, _.TRANSFORM_STYLE_PREFIXED, x);
              var u, c, s, f, l, d;
            })(t, e, n, i, a);
          case Q:
            return (function (t, e, n, r, i, o) {
              var a = o.setStyle,
                u = r.actionTypeId,
                c = r.config;
              switch (u) {
                case it:
                  var s = r.config,
                    f = s.widthUnit,
                    d = void 0 === f ? "" : f,
                    p = s.heightUnit,
                    v = void 0 === p ? "" : p,
                    h = n.widthValue,
                    E = n.heightValue;
                  void 0 !== h &&
                    (d === W && (d = "px"), Rt(t, L, o), a(t, L, h + d)),
                    void 0 !== E &&
                      (v === W && (v = "px"), Rt(t, D, o), a(t, D, E + v));
                  break;
                case rt:
                  !(function (t, e, n, r) {
                    var i = (0, l.default)(
                        e,
                        function (t, e, r) {
                          return ""
                            .concat(t, " ")
                            .concat(r, "(")
                            .concat(e)
                            .concat(wt(r, n), ")");
                        },
                        ""
                      ),
                      o = r.setStyle;
                    Rt(t, N, r), o(t, N, i);
                  })(t, n, c, o);
                  break;
                case ot:
                case at:
                case ut:
                  var g = lt[u],
                    _ = Math.round(n.rValue),
                    y = Math.round(n.gValue),
                    m = Math.round(n.bValue),
                    I = n.aValue;
                  Rt(t, g, o),
                    a(
                      t,
                      g,
                      I >= 1
                        ? "rgb(".concat(_, ",").concat(y, ",").concat(m, ")")
                        : "rgba("
                            .concat(_, ",")
                            .concat(y, ",")
                            .concat(m, ",")
                            .concat(I, ")")
                    );
                  break;
                default:
                  var b = c.unit,
                    T = void 0 === b ? "" : b;
                  Rt(t, i, o), a(t, i, n.value + T);
              }
            })(t, 0, n, i, o, a);
          case K:
            return (function (t, e, n) {
              var r = n.setStyle;
              switch (e.actionTypeId) {
                case ct:
                  var i = e.config.value;
                  return void (i === R && _.IS_BROWSER_ENV
                    ? r(t, U, _.FLEX_PREFIXED)
                    : r(t, U, i));
              }
            })(t, i, a);
          case q:
            var s = i.actionTypeId;
            if ((0, g.isPluginType)(s)) return (0, g.renderPlugin)(s)(c, e, i);
        }
      }),
      (e.clearAllStyles = function (t) {
        var e = t.store,
          n = t.elementApi,
          r = e.getState().ixData,
          i = r.events,
          o = void 0 === i ? {} : i,
          a = r.actionLists,
          u = void 0 === a ? {} : a;
        Object.keys(o).forEach(function (t) {
          var e = o[t],
            r = e.action.config,
            i = r.actionListId,
            a = u[i];
          a && Nt({ actionList: a, event: e, elementApi: n });
        }),
          Object.keys(u).forEach(function (t) {
            Nt({ actionList: u[t], elementApi: n });
          });
      }),
      (e.cleanupHTMLElement = function (t, e, n) {
        var r = n.setStyle,
          i = n.getStyle,
          o = e.actionTypeId;
        if (o === it) {
          var a = e.config;
          a.widthUnit === W && r(t, L, ""), a.heightUnit === W && r(t, D, "");
        }
        i(t, V) && Dt({ effect: Ct, actionTypeId: o, elementApi: n })(t);
      }),
      (e.getMaxDurationItemIndex = Mt),
      (e.getActionListProgress = function (t, e) {
        var n = t.actionItemGroups,
          r = t.useFirstGroupAsInitialState,
          i = e.actionItem,
          o = e.verboseTimeElapsed,
          a = void 0 === o ? 0 : o,
          u = 0,
          c = 0;
        return (
          n.forEach(function (t, e) {
            if (!r || 0 !== e) {
              var n = t.actionItems,
                o = n[Mt(n)],
                s = o.config,
                f = o.actionTypeId;
              i.id === o.id && (c = u + a);
              var l = bt(f) === K ? 0 : s.duration;
              u += s.delay + l;
            }
          }),
          u > 0 ? (0, E.optimizeFloat)(c / u) : 0
        );
      }),
      (e.reduceListToGroup = function (t) {
        var e = t.actionList,
          n = t.actionItemId,
          r = t.rawData,
          i = e.actionItemGroups,
          o = e.continuousParameterGroups,
          a = [],
          u = function (t) {
            return (
              a.push((0, p.mergeIn)(t, ["config"], { delay: 0, duration: 0 })),
              t.id === n
            );
          };
        return (
          i &&
            i.some(function (t) {
              return t.actionItems.some(u);
            }),
          o &&
            o.some(function (t) {
              return t.continuousActionGroups.some(function (t) {
                return t.actionItems.some(u);
              });
            }),
          (0, p.setIn)(
            r,
            ["actionLists"],
            (0, c.default)({}, e.id, {
              id: e.id,
              actionItemGroups: [{ actionItems: a }],
            })
          )
        );
      }),
      (e.shouldNamespaceEventParameter = function (t, e) {
        var n = e.basedOn;
        return (
          (t === h.EventTypeConsts.SCROLLING_IN_VIEW &&
            (n === h.EventBasedOn.ELEMENT || null == n)) ||
          (t === h.EventTypeConsts.MOUSE_MOVE && n === h.EventBasedOn.ELEMENT)
        );
      }),
      (e.getNamespacedParameterId = function (t, e) {
        return t + B + e;
      }),
      (e.shouldAllowMediaQuery = function (t, e) {
        if (null == e) return !0;
        return -1 !== t.indexOf(e);
      }),
      (e.mediaQueriesEqual = function (t, e) {
        return (0, v.default)(t && t.sort(), e && e.sort());
      }),
      (e.stringifyTarget = function (t) {
        if ("string" == typeof t) return t;
        var e = t.id,
          n = void 0 === e ? "" : e,
          r = t.selector,
          i = void 0 === r ? "" : r,
          o = t.useEventTarget;
        return n + z + i + z + (void 0 === o ? "" : o);
      }),
      (e.getItemConfigByKey = void 0);
    var f = s(n(250)),
      l = s(n(251)),
      d = s(n(257)),
      p = n(19),
      v = s(n(113)),
      h = n(2),
      E = n(108),
      g = n(110),
      _ = n(44),
      y = h.IX2EngineConstants,
      m = y.BACKGROUND,
      I = y.TRANSFORM,
      b = y.TRANSLATE_3D,
      T = y.SCALE_3D,
      O = y.ROTATE_X,
      w = y.ROTATE_Y,
      A = y.ROTATE_Z,
      S = y.SKEW,
      x = y.PRESERVE_3D,
      R = y.FLEX,
      C = y.OPACITY,
      N = y.FILTER,
      L = y.WIDTH,
      D = y.HEIGHT,
      P = y.BACKGROUND_COLOR,
      M = y.BORDER_COLOR,
      j = y.COLOR,
      F = y.CHILDREN,
      k = y.IMMEDIATE_CHILDREN,
      G = y.SIBLINGS,
      X = y.PARENT,
      U = y.DISPLAY,
      V = y.WILL_CHANGE,
      W = y.AUTO,
      H = y.COMMA_DELIMITER,
      B = y.COLON_DELIMITER,
      z = y.BAR_DELIMITER,
      Y = y.RENDER_TRANSFORM,
      K = y.RENDER_GENERAL,
      Q = y.RENDER_STYLE,
      q = y.RENDER_PLUGIN,
      $ = h.ActionTypeConsts,
      Z = $.TRANSFORM_MOVE,
      J = $.TRANSFORM_SCALE,
      tt = $.TRANSFORM_ROTATE,
      et = $.TRANSFORM_SKEW,
      nt = $.STYLE_OPACITY,
      rt = $.STYLE_FILTER,
      it = $.STYLE_SIZE,
      ot = $.STYLE_BACKGROUND_COLOR,
      at = $.STYLE_BORDER,
      ut = $.STYLE_TEXT_COLOR,
      ct = $.GENERAL_DISPLAY,
      st = "OBJECT_VALUE",
      ft = function (t) {
        return t.trim();
      },
      lt = Object.freeze(
        ((r = {}),
        (0, c.default)(r, ot, P),
        (0, c.default)(r, at, M),
        (0, c.default)(r, ut, j),
        r)
      ),
      dt = Object.freeze(
        ((i = {}),
        (0, c.default)(i, _.TRANSFORM_PREFIXED, I),
        (0, c.default)(i, P, m),
        (0, c.default)(i, C, C),
        (0, c.default)(i, N, N),
        (0, c.default)(i, L, L),
        (0, c.default)(i, D, D),
        i)
      ),
      pt = {},
      vt = 1;
    var ht = 1;
    var Et = function (t, e) {
      return t === e;
    };
    function gt(t) {
      var e = (0, u.default)(t);
      return "string" === e
        ? { id: t }
        : null != t && "object" === e
        ? {
            id: t.id,
            objectId: t.objectId,
            selector: t.selector,
            selectorGuids: t.selectorGuids,
            appliesTo: t.appliesTo,
            useEventTarget: t.useEventTarget,
          }
        : {};
    }
    function _t(t) {
      var e,
        n,
        r,
        i = t.config,
        o = t.event,
        a = t.eventTarget,
        u = t.elementRoot,
        c = t.elementApi;
      if (!c) throw new Error("IX2 missing elementApi");
      var s = i.targets;
      if (Array.isArray(s) && s.length > 0)
        return s.reduce(function (t, e) {
          return t.concat(
            _t({
              config: { target: e },
              event: o,
              eventTarget: a,
              elementRoot: u,
              elementApi: c,
            })
          );
        }, []);
      var f = c.getValidDocument,
        l = c.getQuerySelector,
        d = c.queryDocument,
        p = c.getChildElements,
        v = c.getSiblingElements,
        E = c.matchSelector,
        g = c.elementContains,
        y = c.isSiblingNode,
        m = i.target;
      if (!m) return [];
      var I = gt(m),
        b = I.id,
        T = I.objectId,
        O = I.selector,
        w = I.selectorGuids,
        A = I.appliesTo,
        S = I.useEventTarget;
      if (T) return [pt[T] || (pt[T] = {})];
      if (A === h.EventAppliesTo.PAGE) {
        var x = f(b);
        return x ? [x] : [];
      }
      var R,
        C,
        N,
        L =
          (null !==
            (e =
              null == o
                ? void 0
                : null === (n = o.action) || void 0 === n
                ? void 0
                : null === (r = n.config) || void 0 === r
                ? void 0
                : r.affectedElements) && void 0 !== e
            ? e
            : {})[b || O] || {},
        D = Boolean(L.id || L.selector),
        P = o && l(gt(o.target));
      if (
        (D
          ? ((R = L.limitAffectedElements), (C = P), (N = l(L)))
          : (C = N = l({ id: b, selector: O, selectorGuids: w })),
        o && S)
      ) {
        var M = a && (N || !0 === S) ? [a] : d(P);
        if (N) {
          if (S === X)
            return d(N).filter(function (t) {
              return M.some(function (e) {
                return g(t, e);
              });
            });
          if (S === F)
            return d(N).filter(function (t) {
              return M.some(function (e) {
                return g(e, t);
              });
            });
          if (S === G)
            return d(N).filter(function (t) {
              return M.some(function (e) {
                return y(e, t);
              });
            });
        }
        return M;
      }
      return null == C || null == N
        ? []
        : _.IS_BROWSER_ENV && u
        ? d(N).filter(function (t) {
            return u.contains(t);
          })
        : R === F
        ? d(C, N)
        : R === k
        ? p(d(C)).filter(E(N))
        : R === G
        ? v(d(C)).filter(E(N))
        : d(N);
    }
    var yt = /px/,
      mt = function (t, e) {
        return e.reduce(function (t, e) {
          return null == t[e.type] && (t[e.type] = Ot[e.type]), t;
        }, t || {});
      };
    var It = function (t, e) {
      return e && (t[e.type] = e.value || 0), t;
    };
    function bt(t) {
      return /^TRANSFORM_/.test(t)
        ? Y
        : /^STYLE_/.test(t)
        ? Q
        : /^GENERAL_/.test(t)
        ? K
        : /^PLUGIN_/.test(t)
        ? q
        : void 0;
    }
    e.getItemConfigByKey = function (t, e, n) {
      if ((0, g.isPluginType)(t)) return (0, g.getPluginConfig)(t)(n, e);
      switch (t) {
        case rt:
          var r = (0, d.default)(n.filters, function (t) {
            return t.type === e;
          });
          return r ? r.value : 0;
        default:
          return n[e];
      }
    };
    var Tt =
        ((o = {}),
        (0, c.default)(
          o,
          Z,
          Object.freeze({ xValue: 0, yValue: 0, zValue: 0 })
        ),
        (0, c.default)(
          o,
          J,
          Object.freeze({ xValue: 1, yValue: 1, zValue: 1 })
        ),
        (0, c.default)(
          o,
          tt,
          Object.freeze({ xValue: 0, yValue: 0, zValue: 0 })
        ),
        (0, c.default)(o, et, Object.freeze({ xValue: 0, yValue: 0 })),
        o),
      Ot = Object.freeze({
        blur: 0,
        "hue-rotate": 0,
        invert: 0,
        grayscale: 0,
        saturate: 100,
        sepia: 0,
        contrast: 100,
        brightness: 100,
      }),
      wt = function (t, e) {
        var n = (0, d.default)(e.filters, function (e) {
          return e.type === t;
        });
        if (n && n.unit) return n.unit;
        switch (t) {
          case "blur":
            return "px";
          case "hue-rotate":
            return "deg";
          default:
            return "%";
        }
      },
      At = Object.keys(Tt);
    var St = /^rgb/,
      xt = RegExp("rgba?".concat("\\(([^)]+)\\)"));
    function Rt(t, e, n) {
      if (_.IS_BROWSER_ENV) {
        var r = dt[e];
        if (r) {
          var i = n.getStyle,
            o = n.setStyle,
            a = i(t, V);
          if (a) {
            var u = a.split(H).map(ft);
            -1 === u.indexOf(r) && o(t, V, u.concat(r).join(H));
          } else o(t, V, r);
        }
      }
    }
    function Ct(t, e, n) {
      if (_.IS_BROWSER_ENV) {
        var r = dt[e];
        if (r) {
          var i = n.getStyle,
            o = n.setStyle,
            a = i(t, V);
          a &&
            -1 !== a.indexOf(r) &&
            o(
              t,
              V,
              a
                .split(H)
                .map(ft)
                .filter(function (t) {
                  return t !== r;
                })
                .join(H)
            );
        }
      }
    }
    function Nt(t) {
      var e = t.actionList,
        n = void 0 === e ? {} : e,
        r = t.event,
        i = t.elementApi,
        o = n.actionItemGroups,
        a = n.continuousParameterGroups;
      o &&
        o.forEach(function (t) {
          Lt({ actionGroup: t, event: r, elementApi: i });
        }),
        a &&
          a.forEach(function (t) {
            t.continuousActionGroups.forEach(function (t) {
              Lt({ actionGroup: t, event: r, elementApi: i });
            });
          });
    }
    function Lt(t) {
      var e = t.actionGroup,
        n = t.event,
        r = t.elementApi;
      e.actionItems.forEach(function (t) {
        var e,
          i = t.actionTypeId,
          o = t.config;
        (e = (0, g.isPluginType)(i)
          ? (0, g.clearPlugin)(i)
          : Dt({ effect: Pt, actionTypeId: i, elementApi: r })),
          _t({ config: o, event: n, elementApi: r }).forEach(e);
      });
    }
    var Dt = function (t) {
      var e = t.effect,
        n = t.actionTypeId,
        r = t.elementApi;
      return function (t) {
        switch (n) {
          case Z:
          case J:
          case tt:
          case et:
            e(t, _.TRANSFORM_PREFIXED, r);
            break;
          case rt:
            e(t, N, r);
            break;
          case nt:
            e(t, C, r);
            break;
          case it:
            e(t, L, r), e(t, D, r);
            break;
          case ot:
          case at:
          case ut:
            e(t, lt[n], r);
            break;
          case ct:
            e(t, U, r);
        }
      };
    };
    function Pt(t, e, n) {
      var r = n.setStyle;
      Ct(t, e, n),
        r(t, e, ""),
        e === _.TRANSFORM_PREFIXED && r(t, _.TRANSFORM_STYLE_PREFIXED, "");
    }
    function Mt(t) {
      var e = 0,
        n = 0;
      return (
        t.forEach(function (t, r) {
          var i = t.config,
            o = i.delay + i.duration;
          o >= e && ((e = o), (n = r));
        }),
        n
      );
    }
  },
  function (t, e) {
    t.exports = function (t, e) {
      return null == t || t != t ? e : t;
    };
  },
  function (t, e, n) {
    var r = n(252),
      i = n(111),
      o = n(7),
      a = n(256),
      u = n(1);
    t.exports = function (t, e, n) {
      var c = u(t) ? r : a,
        s = arguments.length < 3;
      return c(t, o(e, 4), n, s, i);
    };
  },
  function (t, e) {
    t.exports = function (t, e, n, r) {
      var i = -1,
        o = null == t ? 0 : t.length;
      for (r && o && (n = t[++i]); ++i < o; ) n = e(n, t[i], i, t);
      return n;
    };
  },
  function (t, e, n) {
    var r = n(254)();
    t.exports = r;
  },
  function (t, e) {
    t.exports = function (t) {
      return function (e, n, r) {
        for (var i = -1, o = Object(e), a = r(e), u = a.length; u--; ) {
          var c = a[t ? u : ++i];
          if (!1 === n(o[c], c, o)) break;
        }
        return e;
      };
    };
  },
  function (t, e, n) {
    var r = n(12);
    t.exports = function (t, e) {
      return function (n, i) {
        if (null == n) return n;
        if (!r(n)) return t(n, i);
        for (
          var o = n.length, a = e ? o : -1, u = Object(n);
          (e ? a-- : ++a < o) && !1 !== i(u[a], a, u);

        );
        return n;
      };
    };
  },
  function (t, e) {
    t.exports = function (t, e, n, r, i) {
      return (
        i(t, function (t, i, o) {
          n = r ? ((r = !1), t) : e(n, t, i, o);
        }),
        n
      );
    };
  },
  function (t, e, n) {
    var r = n(86)(n(258));
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(104),
      i = n(7),
      o = n(105),
      a = Math.max,
      u = Math.min;
    t.exports = function (t, e, n) {
      var c = null == t ? 0 : t.length;
      if (!c) return -1;
      var s = c - 1;
      return (
        void 0 !== n && ((s = o(n)), (s = n < 0 ? a(c + s, 0) : u(s, c - 1))),
        r(t, i(e, 3), s, !0)
      );
    };
  },
  function (t, e) {
    t.exports = function (t) {
      return t && t.__esModule ? t : { default: t };
    };
  },
  function (t, e, n) {
    "use strict";
    var r = Object.prototype.hasOwnProperty;
    function i(t, e) {
      return t === e ? 0 !== t || 0 !== e || 1 / t == 1 / e : t != t && e != e;
    }
    t.exports = function (t, e) {
      if (i(t, e)) return !0;
      if (
        "object" != typeof t ||
        null === t ||
        "object" != typeof e ||
        null === e
      )
        return !1;
      var n = Object.keys(t),
        o = Object.keys(e);
      if (n.length !== o.length) return !1;
      for (var a = 0; a < n.length; a++)
        if (!r.call(e, n[a]) || !i(t[n[a]], e[n[a]])) return !1;
      return !0;
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.ixInstances = void 0);
    var r = n(2),
      i = n(10),
      o = n(19),
      a = r.IX2EngineActionTypes,
      u = a.IX2_RAW_DATA_IMPORTED,
      c = a.IX2_SESSION_STOPPED,
      s = a.IX2_INSTANCE_ADDED,
      f = a.IX2_INSTANCE_STARTED,
      l = a.IX2_INSTANCE_REMOVED,
      d = a.IX2_ANIMATION_FRAME_CHANGED,
      p = i.IX2EasingUtils,
      v = p.optimizeFloat,
      h = p.applyEasing,
      E = p.createBezierEasing,
      g = r.IX2EngineConstants.RENDER_GENERAL,
      _ = i.IX2VanillaUtils,
      y = _.getItemConfigByKey,
      m = _.getRenderType,
      I = _.getStyleProp,
      b = function (t, e) {
        var n = t.position,
          r = t.parameterId,
          i = t.actionGroups,
          a = t.destinationKeys,
          u = t.smoothing,
          c = t.restingValue,
          s = t.actionTypeId,
          f = t.customEasingFn,
          l = e.payload.parameters,
          d = Math.max(1 - u, 0.01),
          p = l[r];
        null == p && ((d = 1), (p = c));
        var E,
          g,
          _,
          m,
          I = Math.max(p, 0) || 0,
          b = v(I - n),
          T = v(n + b * d),
          O = 100 * T;
        if (T === n && t.current) return t;
        for (var w = 0, A = i.length; w < A; w++) {
          var S = i[w],
            x = S.keyframe,
            R = S.actionItems;
          if ((0 === w && (E = R[0]), O >= x)) {
            E = R[0];
            var C = i[w + 1],
              N = C && O !== x;
            (g = N ? C.actionItems[0] : null),
              N && ((_ = x / 100), (m = (C.keyframe - x) / 100));
          }
        }
        var L = {};
        if (E && !g)
          for (var D = 0, P = a.length; D < P; D++) {
            var M = a[D];
            L[M] = y(s, M, E.config);
          }
        else if (E && g && void 0 !== _ && void 0 !== m)
          for (
            var j = (T - _) / m,
              F = E.config.easing,
              k = h(F, j, f),
              G = 0,
              X = a.length;
            G < X;
            G++
          ) {
            var U = a[G],
              V = y(s, U, E.config),
              W = (y(s, U, g.config) - V) * k + V;
            L[U] = W;
          }
        return (0, o.merge)(t, { position: T, current: L });
      },
      T = function (t, e) {
        var n = t,
          r = n.active,
          i = n.origin,
          a = n.start,
          u = n.immediate,
          c = n.renderType,
          s = n.verbose,
          f = n.actionItem,
          l = n.destination,
          d = n.destinationKeys,
          p = n.pluginDuration,
          E = n.instanceDelay,
          _ = n.customEasingFn,
          y = f.config.easing,
          m = f.config,
          I = m.duration,
          b = m.delay;
        null != p && (I = p),
          (b = null != E ? E : b),
          c === g ? (I = 0) : u && (I = b = 0);
        var T = e.payload.now;
        if (r && i) {
          var O = T - (a + b);
          if (s) {
            var w = T - a,
              A = I + b,
              S = v(Math.min(Math.max(0, w / A), 1));
            t = (0, o.set)(t, "verboseTimeElapsed", A * S);
          }
          if (O < 0) return t;
          var x = v(Math.min(Math.max(0, O / I), 1)),
            R = h(y, x, _),
            C = {},
            N = null;
          return (
            d.length &&
              (N = d.reduce(function (t, e) {
                var n = l[e],
                  r = parseFloat(i[e]) || 0,
                  o = (parseFloat(n) - r) * R + r;
                return (t[e] = o), t;
              }, {})),
            (C.current = N),
            (C.position = x),
            1 === x && ((C.active = !1), (C.complete = !0)),
            (0, o.merge)(t, C)
          );
        }
        return t;
      };
    e.ixInstances = function () {
      var t =
          arguments.length > 0 && void 0 !== arguments[0]
            ? arguments[0]
            : Object.freeze({}),
        e = arguments.length > 1 ? arguments[1] : void 0;
      switch (e.type) {
        case u:
          return e.payload.ixInstances || Object.freeze({});
        case c:
          return Object.freeze({});
        case s:
          var n = e.payload,
            r = n.instanceId,
            i = n.elementId,
            a = n.actionItem,
            p = n.eventId,
            v = n.eventTarget,
            h = n.eventStateKey,
            g = n.actionListId,
            _ = n.groupIndex,
            y = n.isCarrier,
            O = n.origin,
            w = n.destination,
            A = n.immediate,
            S = n.verbose,
            x = n.continuous,
            R = n.parameterId,
            C = n.actionGroups,
            N = n.smoothing,
            L = n.restingValue,
            D = n.pluginInstance,
            P = n.pluginDuration,
            M = n.instanceDelay,
            j = a.actionTypeId,
            F = m(j),
            k = I(F, j),
            G = Object.keys(w).filter(function (t) {
              return null != w[t];
            }),
            X = a.config.easing;
          return (0, o.set)(t, r, {
            id: r,
            elementId: i,
            active: !1,
            position: 0,
            start: 0,
            origin: O,
            destination: w,
            destinationKeys: G,
            immediate: A,
            verbose: S,
            current: null,
            actionItem: a,
            actionTypeId: j,
            eventId: p,
            eventTarget: v,
            eventStateKey: h,
            actionListId: g,
            groupIndex: _,
            renderType: F,
            isCarrier: y,
            styleProp: k,
            continuous: x,
            parameterId: R,
            actionGroups: C,
            smoothing: N,
            restingValue: L,
            pluginInstance: D,
            pluginDuration: P,
            instanceDelay: M,
            customEasingFn: Array.isArray(X) && 4 === X.length ? E(X) : void 0,
          });
        case f:
          var U = e.payload,
            V = U.instanceId,
            W = U.time;
          return (0, o.mergeIn)(t, [V], { active: !0, complete: !1, start: W });
        case l:
          var H = e.payload.instanceId;
          if (!t[H]) return t;
          for (
            var B = {}, z = Object.keys(t), Y = z.length, K = 0;
            K < Y;
            K++
          ) {
            var Q = z[K];
            Q !== H && (B[Q] = t[Q]);
          }
          return B;
        case d:
          for (var q = t, $ = Object.keys(t), Z = $.length, J = 0; J < Z; J++) {
            var tt = $[J],
              et = t[tt],
              nt = et.continuous ? b : T;
            q = (0, o.set)(q, tt, nt(et, e));
          }
          return q;
        default:
          return t;
      }
    };
  },
  function (t, e, n) {
    "use strict";
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.ixParameters = void 0);
    var r = n(2).IX2EngineActionTypes,
      i = r.IX2_RAW_DATA_IMPORTED,
      o = r.IX2_SESSION_STOPPED,
      a = r.IX2_PARAMETER_CHANGED;
    e.ixParameters = function () {
      var t =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        e = arguments.length > 1 ? arguments[1] : void 0;
      switch (e.type) {
        case i:
          return e.payload.ixParameters || {};
        case o:
          return {};
        case a:
          var n = e.payload,
            r = n.key,
            u = n.value;
          return (t[r] = u), t;
        default:
          return t;
      }
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      if (null == t) return {};
      var n,
        r,
        i = {},
        o = Object.keys(t);
      for (r = 0; r < o.length; r++)
        (n = o[r]), e.indexOf(n) >= 0 || (i[n] = t[n]);
      return i;
    };
  },
  function (t, e, n) {
    var r = n(53),
      i = n(55),
      o = n(12),
      a = n(265),
      u = n(266),
      c = "[object Map]",
      s = "[object Set]";
    t.exports = function (t) {
      if (null == t) return 0;
      if (o(t)) return a(t) ? u(t) : t.length;
      var e = i(t);
      return e == c || e == s ? t.size : r(t).length;
    };
  },
  function (t, e, n) {
    var r = n(11),
      i = n(1),
      o = n(9),
      a = "[object String]";
    t.exports = function (t) {
      return "string" == typeof t || (!i(t) && o(t) && r(t) == a);
    };
  },
  function (t, e, n) {
    var r = n(267),
      i = n(268),
      o = n(269);
    t.exports = function (t) {
      return i(t) ? o(t) : r(t);
    };
  },
  function (t, e, n) {
    var r = n(103)("length");
    t.exports = r;
  },
  function (t, e) {
    var n = RegExp(
      "[\\u200d\\ud800-\\udfff\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff\\ufe0e\\ufe0f]"
    );
    t.exports = function (t) {
      return n.test(t);
    };
  },
  function (t, e) {
    var n = "[\\ud800-\\udfff]",
      r = "[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]",
      i = "\\ud83c[\\udffb-\\udfff]",
      o = "[^\\ud800-\\udfff]",
      a = "(?:\\ud83c[\\udde6-\\uddff]){2}",
      u = "[\\ud800-\\udbff][\\udc00-\\udfff]",
      c = "(?:" + r + "|" + i + ")" + "?",
      s =
        "[\\ufe0e\\ufe0f]?" +
        c +
        ("(?:\\u200d(?:" +
          [o, a, u].join("|") +
          ")[\\ufe0e\\ufe0f]?" +
          c +
          ")*"),
      f = "(?:" + [o + r + "?", r, a, u, n].join("|") + ")",
      l = RegExp(i + "(?=" + i + ")|" + f + s, "g");
    t.exports = function (t) {
      for (var e = (l.lastIndex = 0); l.test(t); ) ++e;
      return e;
    };
  },
  function (t, e, n) {
    var r = n(7),
      i = n(271),
      o = n(272);
    t.exports = function (t, e) {
      return o(t, i(r(e)));
    };
  },
  function (t, e) {
    var n = "Expected a function";
    t.exports = function (t) {
      if ("function" != typeof t) throw new TypeError(n);
      return function () {
        var e = arguments;
        switch (e.length) {
          case 0:
            return !t.call(this);
          case 1:
            return !t.call(this, e[0]);
          case 2:
            return !t.call(this, e[0], e[1]);
          case 3:
            return !t.call(this, e[0], e[1], e[2]);
        }
        return !t.apply(this, e);
      };
    };
  },
  function (t, e, n) {
    var r = n(102),
      i = n(7),
      o = n(273),
      a = n(276);
    t.exports = function (t, e) {
      if (null == t) return {};
      var n = r(a(t), function (t) {
        return [t];
      });
      return (
        (e = i(e)),
        o(t, n, function (t, n) {
          return e(t, n[0]);
        })
      );
    };
  },
  function (t, e, n) {
    var r = n(57),
      i = n(274),
      o = n(35);
    t.exports = function (t, e, n) {
      for (var a = -1, u = e.length, c = {}; ++a < u; ) {
        var s = e[a],
          f = r(t, s);
        n(f, s) && i(c, o(s, t), f);
      }
      return c;
    };
  },
  function (t, e, n) {
    var r = n(275),
      i = n(35),
      o = n(50),
      a = n(6),
      u = n(21);
    t.exports = function (t, e, n, c) {
      if (!a(t)) return t;
      for (
        var s = -1, f = (e = i(e, t)).length, l = f - 1, d = t;
        null != d && ++s < f;

      ) {
        var p = u(e[s]),
          v = n;
        if ("__proto__" === p || "constructor" === p || "prototype" === p)
          return t;
        if (s != l) {
          var h = d[p];
          void 0 === (v = c ? c(h, p, d) : void 0) &&
            (v = a(h) ? h : o(e[s + 1]) ? [] : {});
        }
        r(d, p, v), (d = d[p]);
      }
      return t;
    };
  },
  function (t, e, n) {
    var r = n(115),
      i = n(45),
      o = Object.prototype.hasOwnProperty;
    t.exports = function (t, e, n) {
      var a = t[e];
      (o.call(t, e) && i(a, n) && (void 0 !== n || e in t)) || r(t, e, n);
    };
  },
  function (t, e, n) {
    var r = n(93),
      i = n(277),
      o = n(279);
    t.exports = function (t) {
      return r(t, o, i);
    };
  },
  function (t, e, n) {
    var r = n(48),
      i = n(278),
      o = n(94),
      a = n(95),
      u = Object.getOwnPropertySymbols
        ? function (t) {
            for (var e = []; t; ) r(e, o(t)), (t = i(t));
            return e;
          }
        : a;
    t.exports = u;
  },
  function (t, e, n) {
    var r = n(98)(Object.getPrototypeOf, Object);
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(96),
      i = n(280),
      o = n(12);
    t.exports = function (t) {
      return o(t) ? r(t, !0) : i(t);
    };
  },
  function (t, e, n) {
    var r = n(6),
      i = n(54),
      o = n(281),
      a = Object.prototype.hasOwnProperty;
    t.exports = function (t) {
      if (!r(t)) return o(t);
      var e = i(t),
        n = [];
      for (var u in t)
        ("constructor" != u || (!e && a.call(t, u))) && n.push(u);
      return n;
    };
  },
  function (t, e) {
    t.exports = function (t) {
      var e = [];
      if (null != t) for (var n in Object(t)) e.push(n);
      return e;
    };
  },
  function (t, e, n) {
    var r = n(53),
      i = n(55),
      o = n(34),
      a = n(1),
      u = n(12),
      c = n(49),
      s = n(54),
      f = n(51),
      l = "[object Map]",
      d = "[object Set]",
      p = Object.prototype.hasOwnProperty;
    t.exports = function (t) {
      if (null == t) return !0;
      if (
        u(t) &&
        (a(t) ||
          "string" == typeof t ||
          "function" == typeof t.splice ||
          c(t) ||
          f(t) ||
          o(t))
      )
        return !t.length;
      var e = i(t);
      if (e == l || e == d) return !t.size;
      if (s(t)) return !r(t).length;
      for (var n in t) if (p.call(t, n)) return !1;
      return !0;
    };
  },
  function (t, e, n) {
    var r = n(115),
      i = n(112),
      o = n(7);
    t.exports = function (t, e) {
      var n = {};
      return (
        (e = o(e, 3)),
        i(t, function (t, i, o) {
          r(n, i, e(t, i, o));
        }),
        n
      );
    };
  },
  function (t, e, n) {
    var r = n(285),
      i = n(111),
      o = n(286),
      a = n(1);
    t.exports = function (t, e) {
      return (a(t) ? r : i)(t, o(e));
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      for (
        var n = -1, r = null == t ? 0 : t.length;
        ++n < r && !1 !== e(t[n], n, t);

      );
      return t;
    };
  },
  function (t, e, n) {
    var r = n(59);
    t.exports = function (t) {
      return "function" == typeof t ? t : r;
    };
  },
  function (t, e, n) {
    var r = n(288),
      i = n(6),
      o = "Expected a function";
    t.exports = function (t, e, n) {
      var a = !0,
        u = !0;
      if ("function" != typeof t) throw new TypeError(o);
      return (
        i(n) &&
          ((a = "leading" in n ? !!n.leading : a),
          (u = "trailing" in n ? !!n.trailing : u)),
        r(t, e, { leading: a, maxWait: e, trailing: u })
      );
    };
  },
  function (t, e, n) {
    var r = n(6),
      i = n(289),
      o = n(60),
      a = "Expected a function",
      u = Math.max,
      c = Math.min;
    t.exports = function (t, e, n) {
      var s,
        f,
        l,
        d,
        p,
        v,
        h = 0,
        E = !1,
        g = !1,
        _ = !0;
      if ("function" != typeof t) throw new TypeError(a);
      function y(e) {
        var n = s,
          r = f;
        return (s = f = void 0), (h = e), (d = t.apply(r, n));
      }
      function m(t) {
        var n = t - v;
        return void 0 === v || n >= e || n < 0 || (g && t - h >= l);
      }
      function I() {
        var t = i();
        if (m(t)) return b(t);
        p = setTimeout(
          I,
          (function (t) {
            var n = e - (t - v);
            return g ? c(n, l - (t - h)) : n;
          })(t)
        );
      }
      function b(t) {
        return (p = void 0), _ && s ? y(t) : ((s = f = void 0), d);
      }
      function T() {
        var t = i(),
          n = m(t);
        if (((s = arguments), (f = this), (v = t), n)) {
          if (void 0 === p)
            return (function (t) {
              return (h = t), (p = setTimeout(I, e)), E ? y(t) : d;
            })(v);
          if (g) return clearTimeout(p), (p = setTimeout(I, e)), y(v);
        }
        return void 0 === p && (p = setTimeout(I, e)), d;
      }
      return (
        (e = o(e) || 0),
        r(n) &&
          ((E = !!n.leading),
          (l = (g = "maxWait" in n) ? u(o(n.maxWait) || 0, e) : l),
          (_ = "trailing" in n ? !!n.trailing : _)),
        (T.cancel = function () {
          void 0 !== p && clearTimeout(p), (h = 0), (s = v = f = p = void 0);
        }),
        (T.flush = function () {
          return void 0 === p ? d : b(i());
        }),
        T
      );
    };
  },
  function (t, e, n) {
    var r = n(5);
    t.exports = function () {
      return r.Date.now();
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(0)(n(22));
    Object.defineProperty(e, "__esModule", { value: !0 }),
      (e.setStyle = function (t, e, n) {
        t.style[e] = n;
      }),
      (e.getStyle = function (t, e) {
        return t.style[e];
      }),
      (e.getProperty = function (t, e) {
        return t[e];
      }),
      (e.matchSelector = function (t) {
        return function (e) {
          return e[a](t);
        };
      }),
      (e.getQuerySelector = function (t) {
        var e = t.id,
          n = t.selector;
        if (e) {
          var r = e;
          if (-1 !== e.indexOf(c)) {
            var i = e.split(c),
              o = i[0];
            if (((r = i[1]), o !== document.documentElement.getAttribute(l)))
              return null;
          }
          return '[data-w-id="'
            .concat(r, '"], [data-w-id^="')
            .concat(r, '_instance"]');
        }
        return n;
      }),
      (e.getValidDocument = function (t) {
        if (null == t || t === document.documentElement.getAttribute(l))
          return document;
        return null;
      }),
      (e.queryDocument = function (t, e) {
        return Array.prototype.slice.call(
          document.querySelectorAll(e ? t + " " + e : t)
        );
      }),
      (e.elementContains = function (t, e) {
        return t.contains(e);
      }),
      (e.isSiblingNode = function (t, e) {
        return t !== e && t.parentNode === e.parentNode;
      }),
      (e.getChildElements = function (t) {
        for (var e = [], n = 0, r = (t || []).length; n < r; n++) {
          var i = t[n].children,
            o = i.length;
          if (o) for (var a = 0; a < o; a++) e.push(i[a]);
        }
        return e;
      }),
      (e.getSiblingElements = function () {
        for (
          var t =
              arguments.length > 0 && void 0 !== arguments[0]
                ? arguments[0]
                : [],
            e = [],
            n = [],
            r = 0,
            i = t.length;
          r < i;
          r++
        ) {
          var o = t[r].parentNode;
          if (o && o.children && o.children.length && -1 === n.indexOf(o)) {
            n.push(o);
            for (var a = o.firstElementChild; null != a; )
              -1 === t.indexOf(a) && e.push(a), (a = a.nextElementSibling);
          }
        }
        return e;
      }),
      (e.getRefType = function (t) {
        if (null != t && "object" == (0, r.default)(t))
          return t instanceof Element ? s : f;
        return null;
      }),
      (e.getClosestElement = void 0);
    var i = n(10),
      o = n(2),
      a = i.IX2BrowserSupport.ELEMENT_MATCHES,
      u = o.IX2EngineConstants,
      c = u.IX2_ID_DELIMITER,
      s = u.HTML_ELEMENT,
      f = u.PLAIN_OBJECT,
      l = u.WF_PAGE;
    var d = Element.prototype.closest
      ? function (t, e) {
          return document.documentElement.contains(t) ? t.closest(e) : null;
        }
      : function (t, e) {
          if (!document.documentElement.contains(t)) return null;
          var n = t;
          do {
            if (n[a] && n[a](e)) return n;
            n = n.parentNode;
          } while (null != n);
          return null;
        };
    e.getClosestElement = d;
  },
  function (t, e, n) {
    "use strict";
    var r,
      i = n(0),
      o = i(n(27)),
      a = i(n(22)),
      u = n(0);
    Object.defineProperty(e, "__esModule", { value: !0 }), (e.default = void 0);
    var c,
      s,
      f,
      l = u(n(28)),
      d = u(n(292)),
      p = u(n(56)),
      v = u(n(311)),
      h = n(2),
      E = n(114),
      g = n(61),
      _ = n(10),
      y = h.EventTypeConsts,
      m = y.MOUSE_CLICK,
      I = y.MOUSE_SECOND_CLICK,
      b = y.MOUSE_DOWN,
      T = y.MOUSE_UP,
      O = y.MOUSE_OVER,
      w = y.MOUSE_OUT,
      A = y.DROPDOWN_CLOSE,
      S = y.DROPDOWN_OPEN,
      x = y.SLIDER_ACTIVE,
      R = y.SLIDER_INACTIVE,
      C = y.TAB_ACTIVE,
      N = y.TAB_INACTIVE,
      L = y.NAVBAR_CLOSE,
      D = y.NAVBAR_OPEN,
      P = y.MOUSE_MOVE,
      M = y.PAGE_SCROLL_DOWN,
      j = y.SCROLL_INTO_VIEW,
      F = y.SCROLL_OUT_OF_VIEW,
      k = y.PAGE_SCROLL_UP,
      G = y.SCROLLING_IN_VIEW,
      X = y.PAGE_FINISH,
      U = y.ECOMMERCE_CART_CLOSE,
      V = y.ECOMMERCE_CART_OPEN,
      W = y.PAGE_START,
      H = y.PAGE_SCROLL,
      B = "COMPONENT_ACTIVE",
      z = "COMPONENT_INACTIVE",
      Y = h.IX2EngineConstants.COLON_DELIMITER,
      K = _.IX2VanillaUtils.getNamespacedParameterId,
      Q = function (t) {
        return function (e) {
          return !("object" !== (0, a.default)(e) || !t(e)) || e;
        };
      },
      q = Q(function (t) {
        return t.element === t.nativeEvent.target;
      }),
      $ = Q(function (t) {
        var e = t.element,
          n = t.nativeEvent;
        return e.contains(n.target);
      }),
      Z = (0, d.default)([q, $]),
      J = function (t, e) {
        if (e) {
          var n = t.getState().ixData.events[e];
          if (n && !at[n.eventTypeId]) return n;
        }
        return null;
      },
      tt = function (t, e) {
        var n = t.store,
          r = t.event,
          i = t.element,
          o = t.eventStateKey,
          a = r.action,
          u = r.id,
          c = a.config,
          s = c.actionListId,
          f = c.autoStopEventId,
          l = J(n, f);
        return (
          l &&
            (0, E.stopActionGroup)({
              store: n,
              eventId: f,
              eventTarget: i,
              eventStateKey: f + Y + o.split(Y)[1],
              actionListId: (0, p.default)(l, "action.config.actionListId"),
            }),
          (0, E.stopActionGroup)({
            store: n,
            eventId: u,
            eventTarget: i,
            eventStateKey: o,
            actionListId: s,
          }),
          (0, E.startActionGroup)({
            store: n,
            eventId: u,
            eventTarget: i,
            eventStateKey: o,
            actionListId: s,
          }),
          e
        );
      },
      et = function (t, e) {
        return function (n, r) {
          return !0 === t(n, r) ? e(n, r) : r;
        };
      },
      nt = { handler: et(Z, tt) },
      rt = (0, l.default)({}, nt, { types: [B, z].join(" ") }),
      it = [
        { target: window, types: "resize orientationchange", throttle: !0 },
        {
          target: document,
          types: "scroll wheel readystatechange IX2_PAGE_UPDATE",
          throttle: !0,
        },
      ],
      ot = { types: it },
      at = { PAGE_START: W, PAGE_FINISH: X },
      ut =
        ((c = void 0 !== window.pageXOffset),
        (s =
          "CSS1Compat" === document.compatMode
            ? document.documentElement
            : document.body),
        function () {
          return {
            scrollLeft: c ? window.pageXOffset : s.scrollLeft,
            scrollTop: c ? window.pageYOffset : s.scrollTop,
            stiffScrollTop: (0, v.default)(
              c ? window.pageYOffset : s.scrollTop,
              0,
              s.scrollHeight - window.innerHeight
            ),
            scrollWidth: s.scrollWidth,
            scrollHeight: s.scrollHeight,
            clientWidth: s.clientWidth,
            clientHeight: s.clientHeight,
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
          };
        }),
      ct = function (t) {
        var e = t.element,
          n = t.nativeEvent,
          r = n.type,
          i = n.target,
          o = n.relatedTarget,
          a = e.contains(i);
        if ("mouseover" === r && a) return !0;
        var u = e.contains(o);
        return !("mouseout" !== r || !a || !u);
      },
      st = function (t) {
        var e,
          n,
          r = t.element,
          i = t.event.config,
          o = ut(),
          a = o.clientWidth,
          u = o.clientHeight,
          c = i.scrollOffsetValue,
          s = "PX" === i.scrollOffsetUnit ? c : (u * (c || 0)) / 100;
        return (
          (e = r.getBoundingClientRect()),
          (n = { left: 0, top: s, right: a, bottom: u - s }),
          !(
            e.left > n.right ||
            e.right < n.left ||
            e.top > n.bottom ||
            e.bottom < n.top
          )
        );
      },
      ft = function (t) {
        return function (e, n) {
          var r = e.nativeEvent.type,
            i = -1 !== [B, z].indexOf(r) ? r === B : n.isActive,
            o = (0, l.default)({}, n, { isActive: i });
          return n && o.isActive === n.isActive ? o : t(e, o) || o;
        };
      },
      lt = function (t) {
        return function (e, n) {
          var r = { elementHovered: ct(e) };
          return (
            ((n ? r.elementHovered !== n.elementHovered : r.elementHovered) &&
              t(e, r)) ||
            r
          );
        };
      },
      dt = function (t) {
        return function (e) {
          var n =
              arguments.length > 1 && void 0 !== arguments[1]
                ? arguments[1]
                : {},
            r = ut(),
            i = r.stiffScrollTop,
            o = r.scrollHeight,
            a = r.innerHeight,
            u = e.event,
            c = u.config,
            s = u.eventTypeId,
            f = c.scrollOffsetValue,
            d = "PX" === c.scrollOffsetUnit,
            p = o - a,
            v = Number((i / p).toFixed(2));
          if (n && n.percentTop === v) return n;
          var h,
            E,
            g = (d ? f : (a * (f || 0)) / 100) / p,
            _ = 0;
          n &&
            ((h = v > n.percentTop),
            (_ = (E = n.scrollingDown !== h) ? v : n.anchorTop));
          var y = s === M ? v >= _ + g : v <= _ - g,
            m = (0, l.default)({}, n, {
              percentTop: v,
              inBounds: y,
              anchorTop: _,
              scrollingDown: h,
            });
          return (n && y && (E || m.inBounds !== n.inBounds) && t(e, m)) || m;
        };
      },
      pt = function (t) {
        return function (e) {
          var n =
              arguments.length > 1 && void 0 !== arguments[1]
                ? arguments[1]
                : { clickCount: 0 },
            r = { clickCount: (n.clickCount % 2) + 1 };
          return (r.clickCount !== n.clickCount && t(e, r)) || r;
        };
      },
      vt = function () {
        var t =
          !(arguments.length > 0 && void 0 !== arguments[0]) || arguments[0];
        return (0, l.default)({}, rt, {
          handler: et(
            t ? Z : q,
            ft(function (t, e) {
              return e.isActive ? nt.handler(t, e) : e;
            })
          ),
        });
      },
      ht = function () {
        var t =
          !(arguments.length > 0 && void 0 !== arguments[0]) || arguments[0];
        return (0, l.default)({}, rt, {
          handler: et(
            t ? Z : q,
            ft(function (t, e) {
              return e.isActive ? e : nt.handler(t, e);
            })
          ),
        });
      },
      Et = (0, l.default)({}, ot, {
        handler:
          ((f = function (t, e) {
            var n = e.elementVisible,
              r = t.event;
            return !t.store.getState().ixData.events[
              r.action.config.autoStopEventId
            ] && e.triggered
              ? e
              : (r.eventTypeId === j) === n
              ? (tt(t), (0, l.default)({}, e, { triggered: !0 }))
              : e;
          }),
          function (t, e) {
            var n = (0, l.default)({}, e, { elementVisible: st(t) });
            return (
              ((e ? n.elementVisible !== e.elementVisible : n.elementVisible) &&
                f(t, n)) ||
              n
            );
          }),
      }),
      gt =
        ((r = {}),
        (0, o.default)(r, x, vt()),
        (0, o.default)(r, R, ht()),
        (0, o.default)(r, S, vt()),
        (0, o.default)(r, A, ht()),
        (0, o.default)(r, D, vt(!1)),
        (0, o.default)(r, L, ht(!1)),
        (0, o.default)(r, C, vt()),
        (0, o.default)(r, N, ht()),
        (0, o.default)(r, V, {
          types: "ecommerce-cart-open",
          handler: et(Z, tt),
        }),
        (0, o.default)(r, U, {
          types: "ecommerce-cart-close",
          handler: et(Z, tt),
        }),
        (0, o.default)(r, m, {
          types: "click",
          handler: et(
            Z,
            pt(function (t, e) {
              var n,
                r,
                i,
                o = e.clickCount;
              (r = (n = t).store),
                (i = n.event.action.config.autoStopEventId),
                Boolean(J(r, i)) ? 1 === o && tt(t) : tt(t);
            })
          ),
        }),
        (0, o.default)(r, I, {
          types: "click",
          handler: et(
            Z,
            pt(function (t, e) {
              2 === e.clickCount && tt(t);
            })
          ),
        }),
        (0, o.default)(r, b, (0, l.default)({}, nt, { types: "mousedown" })),
        (0, o.default)(r, T, (0, l.default)({}, nt, { types: "mouseup" })),
        (0, o.default)(r, O, {
          types: "mouseover mouseout",
          handler: et(
            Z,
            lt(function (t, e) {
              e.elementHovered && tt(t);
            })
          ),
        }),
        (0, o.default)(r, w, {
          types: "mouseover mouseout",
          handler: et(
            Z,
            lt(function (t, e) {
              e.elementHovered || tt(t);
            })
          ),
        }),
        (0, o.default)(r, P, {
          types: "mousemove mouseout scroll",
          handler: function (t) {
            var e = t.store,
              n = t.element,
              r = t.eventConfig,
              i = t.nativeEvent,
              o = t.eventStateKey,
              a =
                arguments.length > 1 && void 0 !== arguments[1]
                  ? arguments[1]
                  : { clientX: 0, clientY: 0, pageX: 0, pageY: 0 },
              u = r.basedOn,
              c = r.selectedAxis,
              s = r.continuousParameterGroupId,
              f = r.reverse,
              l = r.restingState,
              d = void 0 === l ? 0 : l,
              p = i.clientX,
              v = void 0 === p ? a.clientX : p,
              E = i.clientY,
              _ = void 0 === E ? a.clientY : E,
              y = i.pageX,
              m = void 0 === y ? a.pageX : y,
              I = i.pageY,
              b = void 0 === I ? a.pageY : I,
              T = "X_AXIS" === c,
              O = "mouseout" === i.type,
              w = d / 100,
              A = s,
              S = !1;
            switch (u) {
              case h.EventBasedOn.VIEWPORT:
                w = T
                  ? Math.min(v, window.innerWidth) / window.innerWidth
                  : Math.min(_, window.innerHeight) / window.innerHeight;
                break;
              case h.EventBasedOn.PAGE:
                var x = ut(),
                  R = x.scrollLeft,
                  C = x.scrollTop,
                  N = x.scrollWidth,
                  L = x.scrollHeight;
                w = T ? Math.min(R + m, N) / N : Math.min(C + b, L) / L;
                break;
              case h.EventBasedOn.ELEMENT:
              default:
                A = K(o, s);
                var D = 0 === i.type.indexOf("mouse");
                if (D && !0 !== Z({ element: n, nativeEvent: i })) break;
                var P = n.getBoundingClientRect(),
                  M = P.left,
                  j = P.top,
                  F = P.width,
                  k = P.height;
                if (
                  !D &&
                  !(function (t, e) {
                    return (
                      t.left > e.left &&
                      t.left < e.right &&
                      t.top > e.top &&
                      t.top < e.bottom
                    );
                  })({ left: v, top: _ }, P)
                )
                  break;
                (S = !0), (w = T ? (v - M) / F : (_ - j) / k);
            }
            return (
              O && (w > 0.95 || w < 0.05) && (w = Math.round(w)),
              (u !== h.EventBasedOn.ELEMENT || S || S !== a.elementHovered) &&
                ((w = f ? 1 - w : w),
                e.dispatch((0, g.parameterChanged)(A, w))),
              { elementHovered: S, clientX: v, clientY: _, pageX: m, pageY: b }
            );
          },
        }),
        (0, o.default)(r, H, {
          types: it,
          handler: function (t) {
            var e = t.store,
              n = t.eventConfig,
              r = n.continuousParameterGroupId,
              i = n.reverse,
              o = ut(),
              a = o.scrollTop / (o.scrollHeight - o.clientHeight);
            (a = i ? 1 - a : a), e.dispatch((0, g.parameterChanged)(r, a));
          },
        }),
        (0, o.default)(r, G, {
          types: it,
          handler: function (t) {
            var e = t.element,
              n = t.store,
              r = t.eventConfig,
              i = t.eventStateKey,
              o =
                arguments.length > 1 && void 0 !== arguments[1]
                  ? arguments[1]
                  : { scrollPercent: 0 },
              a = ut(),
              u = a.scrollLeft,
              c = a.scrollTop,
              s = a.scrollWidth,
              f = a.scrollHeight,
              l = a.clientHeight,
              d = r.basedOn,
              p = r.selectedAxis,
              v = r.continuousParameterGroupId,
              E = r.startsEntering,
              _ = r.startsExiting,
              y = r.addEndOffset,
              m = r.addStartOffset,
              I = r.addOffsetValue,
              b = void 0 === I ? 0 : I,
              T = r.endOffsetValue,
              O = void 0 === T ? 0 : T,
              w = "X_AXIS" === p;
            if (d === h.EventBasedOn.VIEWPORT) {
              var A = w ? u / s : c / f;
              return (
                A !== o.scrollPercent &&
                  n.dispatch((0, g.parameterChanged)(v, A)),
                { scrollPercent: A }
              );
            }
            var S = K(i, v),
              x = e.getBoundingClientRect(),
              R = (m ? b : 0) / 100,
              C = (y ? O : 0) / 100;
            (R = E ? R : 1 - R), (C = _ ? C : 1 - C);
            var N = x.top + Math.min(x.height * R, l),
              L = x.top + x.height * C - N,
              D = Math.min(l + L, f),
              P = Math.min(Math.max(0, l - N), D) / D;
            return (
              P !== o.scrollPercent &&
                n.dispatch((0, g.parameterChanged)(S, P)),
              { scrollPercent: P }
            );
          },
        }),
        (0, o.default)(r, j, Et),
        (0, o.default)(r, F, Et),
        (0, o.default)(
          r,
          M,
          (0, l.default)({}, ot, {
            handler: dt(function (t, e) {
              e.scrollingDown && tt(t);
            }),
          })
        ),
        (0, o.default)(
          r,
          k,
          (0, l.default)({}, ot, {
            handler: dt(function (t, e) {
              e.scrollingDown || tt(t);
            }),
          })
        ),
        (0, o.default)(r, X, {
          types: "readystatechange IX2_PAGE_UPDATE",
          handler: et(
            q,
            (function (t) {
              return function (e, n) {
                var r = { finished: "complete" === document.readyState };
                return !r.finished || (n && n.finshed) || t(e), r;
              };
            })(tt)
          ),
        }),
        (0, o.default)(r, W, {
          types: "readystatechange IX2_PAGE_UPDATE",
          handler: et(
            q,
            (function (t) {
              return function (e, n) {
                return n || t(e), { started: !0 };
              };
            })(tt)
          ),
        }),
        r);
    e.default = gt;
  },
  function (t, e, n) {
    var r = n(293)();
    t.exports = r;
  },
  function (t, e, n) {
    var r = n(62),
      i = n(294),
      o = n(118),
      a = n(119),
      u = n(1),
      c = n(307),
      s = "Expected a function",
      f = 8,
      l = 32,
      d = 128,
      p = 256;
    t.exports = function (t) {
      return i(function (e) {
        var n = e.length,
          i = n,
          v = r.prototype.thru;
        for (t && e.reverse(); i--; ) {
          var h = e[i];
          if ("function" != typeof h) throw new TypeError(s);
          if (v && !E && "wrapper" == a(h)) var E = new r([], !0);
        }
        for (i = E ? i : n; ++i < n; ) {
          h = e[i];
          var g = a(h),
            _ = "wrapper" == g ? o(h) : void 0;
          E =
            _ && c(_[0]) && _[1] == (d | f | l | p) && !_[4].length && 1 == _[9]
              ? E[a(_[0])].apply(E, _[3])
              : 1 == h.length && c(h)
              ? E[g]()
              : E.thru(h);
        }
        return function () {
          var t = arguments,
            r = t[0];
          if (E && 1 == t.length && u(r)) return E.plant(r).value();
          for (var i = 0, o = n ? e[i].apply(this, t) : r; ++i < n; )
            o = e[i].call(this, o);
          return o;
        };
      });
    };
  },
  function (t, e, n) {
    var r = n(295),
      i = n(298),
      o = n(300);
    t.exports = function (t) {
      return o(i(t, void 0, r), t + "");
    };
  },
  function (t, e, n) {
    var r = n(296);
    t.exports = function (t) {
      return null != t && t.length ? r(t, 1) : [];
    };
  },
  function (t, e, n) {
    var r = n(48),
      i = n(297);
    t.exports = function t(e, n, o, a, u) {
      var c = -1,
        s = e.length;
      for (o || (o = i), u || (u = []); ++c < s; ) {
        var f = e[c];
        n > 0 && o(f)
          ? n > 1
            ? t(f, n - 1, o, a, u)
            : r(u, f)
          : a || (u[u.length] = f);
      }
      return u;
    };
  },
  function (t, e, n) {
    var r = n(20),
      i = n(34),
      o = n(1),
      a = r ? r.isConcatSpreadable : void 0;
    t.exports = function (t) {
      return o(t) || i(t) || !!(a && t && t[a]);
    };
  },
  function (t, e, n) {
    var r = n(299),
      i = Math.max;
    t.exports = function (t, e, n) {
      return (
        (e = i(void 0 === e ? t.length - 1 : e, 0)),
        function () {
          for (
            var o = arguments, a = -1, u = i(o.length - e, 0), c = Array(u);
            ++a < u;

          )
            c[a] = o[e + a];
          a = -1;
          for (var s = Array(e + 1); ++a < e; ) s[a] = o[a];
          return (s[e] = n(c)), r(t, this, s);
        }
      );
    };
  },
  function (t, e) {
    t.exports = function (t, e, n) {
      switch (n.length) {
        case 0:
          return t.call(e);
        case 1:
          return t.call(e, n[0]);
        case 2:
          return t.call(e, n[0], n[1]);
        case 3:
          return t.call(e, n[0], n[1], n[2]);
      }
      return t.apply(e, n);
    };
  },
  function (t, e, n) {
    var r = n(301),
      i = n(303)(r);
    t.exports = i;
  },
  function (t, e, n) {
    var r = n(302),
      i = n(116),
      o = n(59),
      a = i
        ? function (t, e) {
            return i(t, "toString", {
              configurable: !0,
              enumerable: !1,
              value: r(e),
              writable: !0,
            });
          }
        : o;
    t.exports = a;
  },
  function (t, e) {
    t.exports = function (t) {
      return function () {
        return t;
      };
    };
  },
  function (t, e) {
    var n = 800,
      r = 16,
      i = Date.now;
    t.exports = function (t) {
      var e = 0,
        o = 0;
      return function () {
        var a = i(),
          u = r - (a - o);
        if (((o = a), u > 0)) {
          if (++e >= n) return arguments[0];
        } else e = 0;
        return t.apply(void 0, arguments);
      };
    };
  },
  function (t, e, n) {
    var r = n(99),
      i = r && new r();
    t.exports = i;
  },
  function (t, e) {
    t.exports = function () {};
  },
  function (t, e) {
    t.exports = {};
  },
  function (t, e, n) {
    var r = n(64),
      i = n(118),
      o = n(119),
      a = n(308);
    t.exports = function (t) {
      var e = o(t),
        n = a[e];
      if ("function" != typeof n || !(e in r.prototype)) return !1;
      if (t === n) return !0;
      var u = i(n);
      return !!u && t === u[0];
    };
  },
  function (t, e, n) {
    var r = n(64),
      i = n(62),
      o = n(63),
      a = n(1),
      u = n(9),
      c = n(309),
      s = Object.prototype.hasOwnProperty;
    function f(t) {
      if (u(t) && !a(t) && !(t instanceof r)) {
        if (t instanceof i) return t;
        if (s.call(t, "__wrapped__")) return c(t);
      }
      return new i(t);
    }
    (f.prototype = o.prototype), (f.prototype.constructor = f), (t.exports = f);
  },
  function (t, e, n) {
    var r = n(64),
      i = n(62),
      o = n(310);
    t.exports = function (t) {
      if (t instanceof r) return t.clone();
      var e = new i(t.__wrapped__, t.__chain__);
      return (
        (e.__actions__ = o(t.__actions__)),
        (e.__index__ = t.__index__),
        (e.__values__ = t.__values__),
        e
      );
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      var n = -1,
        r = t.length;
      for (e || (e = Array(r)); ++n < r; ) e[n] = t[n];
      return e;
    };
  },
  function (t, e, n) {
    var r = n(312),
      i = n(60);
    t.exports = function (t, e, n) {
      return (
        void 0 === n && ((n = e), (e = void 0)),
        void 0 !== n && (n = (n = i(n)) == n ? n : 0),
        void 0 !== e && (e = (e = i(e)) == e ? e : 0),
        r(i(t), e, n)
      );
    };
  },
  function (t, e) {
    t.exports = function (t, e, n) {
      return (
        t == t &&
          (void 0 !== n && (t = t <= n ? t : n),
          void 0 !== e && (t = t >= e ? t : e)),
        t
      );
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(3);
    r.define(
      "links",
      (t.exports = function (t, e) {
        var n,
          i,
          o,
          a = {},
          u = t(window),
          c = r.env(),
          s = window.location,
          f = document.createElement("a"),
          l = "w--current",
          d = /index\.(html|php)$/,
          p = /\/$/;
        function v(e) {
          var r =
            (n && e.getAttribute("href-disabled")) || e.getAttribute("href");
          if (((f.href = r), !(r.indexOf(":") >= 0))) {
            var a = t(e);
            if (
              f.hash.length > 1 &&
              f.host + f.pathname === s.host + s.pathname
            ) {
              if (!/^#[a-zA-Z0-9\-\_]+$/.test(f.hash)) return;
              var u = t(f.hash);
              u.length && i.push({ link: a, sec: u, active: !1 });
            } else if ("#" !== r && "" !== r) {
              var c = f.href === s.href || r === o || (d.test(r) && p.test(o));
              E(a, l, c);
            }
          }
        }
        function h() {
          var t = u.scrollTop(),
            n = u.height();
          e.each(i, function (e) {
            var r = e.link,
              i = e.sec,
              o = i.offset().top,
              a = i.outerHeight(),
              u = 0.5 * n,
              c = i.is(":visible") && o + a - u >= t && o + u <= t + n;
            e.active !== c && ((e.active = c), E(r, l, c));
          });
        }
        function E(t, e, n) {
          var r = t.hasClass(e);
          (n && r) || ((n || r) && (n ? t.addClass(e) : t.removeClass(e)));
        }
        return (
          (a.ready =
            a.design =
            a.preview =
              function () {
                (n = c && r.env("design")),
                  (o = r.env("slug") || s.pathname || ""),
                  r.scroll.off(h),
                  (i = []);
                for (var t = document.links, e = 0; e < t.length; ++e) v(t[e]);
                i.length && (r.scroll.on(h), h());
              }),
          a
        );
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = n(3);
    r.define(
      "scroll",
      (t.exports = function (t) {
        var e,
          n = {
            WF_CHANGE: "change.wf-change",
            WF_CLICK_EMPTY: "click.wf-empty-link",
            WF_CLICK_SCROLL: "click.wf-scroll",
          },
          i = t(document),
          o = window,
          a = o.location,
          u = (function () {
            try {
              return Boolean(o.frameElement);
            } catch (t) {
              return !0;
            }
          })()
            ? null
            : o.history,
          c = /^[a-zA-Z0-9][\w:.-]*$/,
          s = 'a[href="#"]',
          f = 'a[href*="#"]:not(.w-tab-link):not(' + s + ")";
        function l(n) {
          if (
            !(
              r.env("design") ||
              (window.$.mobile && t(n.currentTarget).hasClass("ui-link"))
            )
          ) {
            var i = this.href.split("#"),
              s = i[0] === e ? i[1] : null;
            s &&
              (function (e, n) {
                if (!c.test(e)) return;
                var i = t("#" + e);
                if (!i.length) return;
                n && (n.preventDefault(), n.stopPropagation());
                if (
                  a.hash !== e &&
                  u &&
                  u.pushState &&
                  (!r.env.chrome || "file:" !== a.protocol)
                ) {
                  var s = u.state && u.state.hash;
                  s !== e && u.pushState({ hash: e }, "", "#" + e);
                }
                var f = r.env("editor") ? ".w-editor-body" : "body",
                  l = t(
                    "header, " +
                      f +
                      " > .header, " +
                      f +
                      " > .w-nav:not([data-no-scroll])"
                  ),
                  d = "fixed" === l.css("position") ? l.outerHeight() : 0;
                o.setTimeout(
                  function () {
                    !(function (e, n) {
                      var r = t(o).scrollTop(),
                        i = e.offset().top - n;
                      if ("mid" === e.data("scroll")) {
                        var a = t(o).height() - n,
                          u = e.outerHeight();
                        u < a && (i -= Math.round((a - u) / 2));
                      }
                      var c = 1;
                      t("body")
                        .add(e)
                        .each(function () {
                          var e = parseFloat(
                            t(this).attr("data-scroll-time"),
                            10
                          );
                          !isNaN(e) && (0 === e || e > 0) && (c = e);
                        }),
                        Date.now ||
                          (Date.now = function () {
                            return new Date().getTime();
                          });
                      var s = Date.now(),
                        f =
                          o.requestAnimationFrame ||
                          o.mozRequestAnimationFrame ||
                          o.webkitRequestAnimationFrame ||
                          function (t) {
                            o.setTimeout(t, 15);
                          },
                        l =
                          (472.143 * Math.log(Math.abs(r - i) + 125) - 2e3) * c;
                      !(function t() {
                        var e = Date.now() - s;
                        o.scroll(
                          0,
                          (function (t, e, n, r) {
                            if (n > r) return e;
                            return (
                              t +
                              (e - t) *
                                ((i = n / r),
                                i < 0.5
                                  ? 4 * i * i * i
                                  : (i - 1) * (2 * i - 2) * (2 * i - 2) + 1)
                            );
                            var i;
                          })(r, i, e, l)
                        ),
                          e <= l && f(t);
                      })();
                    })(i, d);
                  },
                  n ? 0 : 300
                );
              })(s, n);
          }
        }
        return {
          ready: function () {
            n.WF_CHANGE;
            var t = n.WF_CLICK_EMPTY,
              r = n.WF_CLICK_SCROLL;
            (e = a.href.split("#")[0]),
              i.on(r, f, l),
              i.on(t, s, function (t) {
                t.preventDefault();
              });
          },
        };
      })
    );
  },
  function (t, e, n) {
    "use strict";
    n(3).define(
      "touch",
      (t.exports = function (t) {
        var e = {},
          n = window.getSelection;
        function r(e) {
          var r,
            i,
            o = !1,
            a = !1,
            u = Math.min(Math.round(0.04 * window.innerWidth), 40);
          function c(t) {
            var e = t.touches;
            (e && e.length > 1) ||
              ((o = !0),
              e ? ((a = !0), (r = e[0].clientX)) : (r = t.clientX),
              (i = r));
          }
          function s(e) {
            if (o) {
              if (a && "mousemove" === e.type)
                return e.preventDefault(), void e.stopPropagation();
              var r = e.touches,
                c = r ? r[0].clientX : e.clientX,
                s = c - i;
              (i = c),
                Math.abs(s) > u &&
                  n &&
                  "" === String(n()) &&
                  (!(function (e, n, r) {
                    var i = t.Event(e, { originalEvent: n });
                    t(n.target).trigger(i, r);
                  })("swipe", e, { direction: s > 0 ? "right" : "left" }),
                  l());
            }
          }
          function f(t) {
            if (o)
              return (
                (o = !1),
                a && "mouseup" === t.type
                  ? (t.preventDefault(), t.stopPropagation(), void (a = !1))
                  : void 0
              );
          }
          function l() {
            o = !1;
          }
          e.addEventListener("touchstart", c, !1),
            e.addEventListener("touchmove", s, !1),
            e.addEventListener("touchend", f, !1),
            e.addEventListener("touchcancel", l, !1),
            e.addEventListener("mousedown", c, !1),
            e.addEventListener("mousemove", s, !1),
            e.addEventListener("mouseup", f, !1),
            e.addEventListener("mouseout", l, !1),
            (this.destroy = function () {
              e.removeEventListener("touchstart", c, !1),
                e.removeEventListener("touchmove", s, !1),
                e.removeEventListener("touchend", f, !1),
                e.removeEventListener("touchcancel", l, !1),
                e.removeEventListener("mousedown", c, !1),
                e.removeEventListener("mousemove", s, !1),
                e.removeEventListener("mouseup", f, !1),
                e.removeEventListener("mouseout", l, !1),
                (e = null);
            });
        }
        return (
          (t.event.special.tap = { bindType: "click", delegateType: "click" }),
          (e.init = function (e) {
            return (e = "string" == typeof e ? t(e).get(0) : e)
              ? new r(e)
              : null;
          }),
          (e.instance = e.init(document)),
          e
        );
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = n(3),
      i = n(13),
      o = {
        ARROW_LEFT: 37,
        ARROW_UP: 38,
        ARROW_RIGHT: 39,
        ARROW_DOWN: 40,
        ESCAPE: 27,
        SPACE: 32,
        ENTER: 13,
        HOME: 36,
        END: 35,
      },
      a = !0,
      u = /^#[a-zA-Z0-9\-_]+$/;
    r.define(
      "dropdown",
      (t.exports = function (t, e) {
        var n,
          c,
          s = e.debounce,
          f = {},
          l = r.env(),
          d = !1,
          p = r.env.touch,
          v = ".w-dropdown",
          h = "w--open",
          E = i.triggers,
          g = 900,
          _ = "focusout" + v,
          y = "keydown" + v,
          m = "mouseenter" + v,
          I = "mousemove" + v,
          b = "mouseleave" + v,
          T = (p ? "click" : "mouseup") + v,
          O = "w-close" + v,
          w = "setting" + v,
          A = t(document);
        function S() {
          (n = l && r.env("design")), (c = A.find(v)).each(x);
        }
        function x(e, i) {
          var c = t(i),
            f = t.data(i, v);
          f ||
            (f = t.data(i, v, {
              open: !1,
              el: c,
              config: {},
              selectedIdx: -1,
            })),
            (f.toggle = f.el.children(".w-dropdown-toggle")),
            (f.list = f.el.children(".w-dropdown-list")),
            (f.links = f.list.find("a:not(.w-dropdown .w-dropdown a)")),
            (f.complete = (function (t) {
              return function () {
                t.list.removeClass(h),
                  t.toggle.removeClass(h),
                  t.manageZ && t.el.css("z-index", "");
              };
            })(f)),
            (f.mouseLeave = (function (t) {
              return function () {
                (t.hovering = !1), t.links.is(":focus") || L(t);
              };
            })(f)),
            (f.mouseUpOutside = (function (e) {
              e.mouseUpOutside && A.off(T, e.mouseUpOutside);
              return s(function (n) {
                if (e.open) {
                  var i = t(n.target);
                  if (!i.closest(".w-dropdown-toggle").length) {
                    var o = -1 === t.inArray(e.el[0], i.parents(v)),
                      a = r.env("editor");
                    if (o) {
                      if (a) {
                        var u =
                            1 === i.parents().length &&
                            1 === i.parents("svg").length,
                          c = i.parents(
                            ".w-editor-bem-EditorHoverControls"
                          ).length;
                        if (u || c) return;
                      }
                      L(e);
                    }
                  }
                }
              });
            })(f)),
            (f.mouseMoveOutside = (function (e) {
              return s(function (n) {
                if (e.open) {
                  var r = t(n.target),
                    i = -1 === t.inArray(e.el[0], r.parents(v));
                  if (i) {
                    var o = r.parents(
                        ".w-editor-bem-EditorHoverControls"
                      ).length,
                      a = r.parents(".w-editor-bem-RTToolbar").length,
                      u = t(".w-editor-bem-EditorOverlay"),
                      c =
                        u.find(".w-editor-edit-outline").length ||
                        u.find(".w-editor-bem-RTToolbar").length;
                    if (o || a || c) return;
                    (e.hovering = !1), L(e);
                  }
                }
              });
            })(f)),
            R(f);
          var p = f.toggle.attr("id"),
            E = f.list.attr("id");
          p || (p = "w-dropdown-toggle-" + e),
            E || (E = "w-dropdown-list-" + e),
            f.toggle.attr("id", p),
            f.toggle.attr("aria-controls", E),
            f.toggle.attr("aria-haspopup", "menu"),
            f.toggle.attr("aria-expanded", "false"),
            "BUTTON" !== f.toggle.prop("tagName") &&
              (f.toggle.attr("role", "button"),
              f.toggle.attr("tabindex") || f.toggle.attr("tabindex", "0")),
            f.list.attr("id", E),
            f.list.attr("aria-labelledby", p),
            f.links.each(function (t, e) {
              e.hasAttribute("tabindex") || e.setAttribute("tabindex", "0"),
                u.test(e.hash) && e.addEventListener("click", L.bind(null, f));
            }),
            f.el.off(v),
            f.toggle.off(v),
            f.nav && f.nav.off(v);
          var g = C(f, a);
          n &&
            f.el.on(
              w,
              (function (t) {
                return function (e, n) {
                  (n = n || {}),
                    R(t),
                    !0 === n.open && N(t),
                    !1 === n.open && L(t, { immediate: !0 });
                };
              })(f)
            ),
            n ||
              (l && ((f.hovering = !1), L(f)),
              f.config.hover &&
                f.toggle.on(
                  m,
                  (function (t) {
                    return function () {
                      (t.hovering = !0), N(t);
                    };
                  })(f)
                ),
              f.el.on(O, g),
              f.el.on(
                y,
                (function (t) {
                  return function (e) {
                    if (!n && !d && t.open)
                      switch (
                        ((t.selectedIdx = t.links.index(
                          document.activeElement
                        )),
                        e.keyCode)
                      ) {
                        case o.HOME:
                          if (!t.open) return;
                          return (t.selectedIdx = 0), D(t), e.preventDefault();
                        case o.END:
                          if (!t.open) return;
                          return (
                            (t.selectedIdx = t.links.length - 1),
                            D(t),
                            e.preventDefault()
                          );
                        case o.ESCAPE:
                          return L(t), t.toggle.focus(), e.stopPropagation();
                        case o.ARROW_RIGHT:
                        case o.ARROW_DOWN:
                          return (
                            (t.selectedIdx = Math.min(
                              t.links.length - 1,
                              t.selectedIdx + 1
                            )),
                            D(t),
                            e.preventDefault()
                          );
                        case o.ARROW_LEFT:
                        case o.ARROW_UP:
                          return (
                            (t.selectedIdx = Math.max(-1, t.selectedIdx - 1)),
                            D(t),
                            e.preventDefault()
                          );
                      }
                  };
                })(f)
              ),
              f.el.on(
                _,
                (function (t) {
                  return s(function (e) {
                    var n = e.relatedTarget,
                      r = e.target,
                      i = t.el[0],
                      o = i.contains(n) || i.contains(r);
                    return o || L(t), e.stopPropagation();
                  });
                })(f)
              ),
              f.toggle.on(T, g),
              f.toggle.on(
                y,
                (function (t) {
                  var e = C(t, a);
                  return function (r) {
                    if (!n && !d) {
                      if (!t.open)
                        switch (r.keyCode) {
                          case o.ARROW_UP:
                          case o.ARROW_DOWN:
                            return r.stopPropagation();
                        }
                      switch (r.keyCode) {
                        case o.SPACE:
                        case o.ENTER:
                          return e(), r.stopPropagation(), r.preventDefault();
                      }
                    }
                  };
                })(f)
              ),
              (f.nav = f.el.closest(".w-nav")),
              f.nav.on(O, g));
        }
        function R(t) {
          var e = Number(t.el.css("z-index"));
          (t.manageZ = e === g || e === g + 1),
            (t.config = {
              hover:
                (!0 === t.el.attr("data-hover") ||
                  "1" === t.el.attr("data-hover")) &&
                !p,
              delay: Number(t.el.attr("data-delay")) || 0,
            });
        }
        function C(t, e) {
          return s(function (n) {
            if (t.open || (n && "w-close" === n.type))
              return L(t, { forceClose: e });
            N(t);
          });
        }
        function N(e) {
          if (!e.open) {
            !(function (e) {
              var n = e.el[0];
              c.each(function (e, r) {
                var i = t(r);
                i.is(n) || i.has(n).length || i.triggerHandler(O);
              });
            })(e),
              (e.open = !0),
              e.list.addClass(h),
              e.toggle.addClass(h),
              e.toggle.attr("aria-expanded", "true"),
              E.intro(0, e.el[0]),
              r.redraw.up(),
              e.manageZ && e.el.css("z-index", g + 1);
            var i = r.env("editor");
            n || A.on(T, e.mouseUpOutside),
              e.hovering && !i && e.el.on(b, e.mouseLeave),
              e.hovering && i && A.on(I, e.mouseMoveOutside),
              window.clearTimeout(e.delayId);
          }
        }
        function L(t) {
          var e =
              arguments.length > 1 && void 0 !== arguments[1]
                ? arguments[1]
                : {},
            n = e.immediate,
            r = e.forceClose;
          if (t.open && (!t.config.hover || !t.hovering || r)) {
            t.toggle.attr("aria-expanded", "false"), (t.open = !1);
            var i = t.config;
            if (
              (E.outro(0, t.el[0]),
              A.off(T, t.mouseUpOutside),
              A.off(I, t.mouseMoveOutside),
              t.el.off(b, t.mouseLeave),
              window.clearTimeout(t.delayId),
              !i.delay || n)
            )
              return t.complete();
            t.delayId = window.setTimeout(t.complete, i.delay);
          }
        }
        function D(t) {
          t.links[t.selectedIdx] && t.links[t.selectedIdx].focus();
        }
        return (
          (f.ready = S),
          (f.design = function () {
            d &&
              A.find(v).each(function (e, n) {
                t(n).triggerHandler(O);
              }),
              (d = !1),
              S();
          }),
          (f.preview = function () {
            (d = !0), S();
          }),
          f
        );
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = n(0)(n(318)),
      i = n(3);
    i.define(
      "forms",
      (t.exports = function (t, e) {
        var n,
          o,
          a,
          u,
          c,
          s = {},
          f = t(document),
          l = window.location,
          d = window.XDomainRequest && !window.atob,
          p = ".w-form",
          v = /e(-)?mail/i,
          h = /^\S+@\S+$/,
          E = window.alert,
          g = i.env(),
          _ = /list-manage[1-9]?.com/i,
          y = e.debounce(function () {
            E(
              "Oops! This page has improperly configured forms. Please contact your website administrator to fix this issue."
            );
          }, 100);
        function m(e, n) {
          var r = t(n),
            i = t.data(n, p);
          i || (i = t.data(n, p, { form: r })), I(i);
          var a = r.closest("div.w-form");
          (i.done = a.find("> .w-form-done")),
            (i.fail = a.find("> .w-form-fail")),
            (i.fileUploads = a.find(".w-file-upload")),
            i.fileUploads.each(function (e) {
              !(function (e, n) {
                if (!n.fileUploads || !n.fileUploads[e]) return;
                var r,
                  i = t(n.fileUploads[e]),
                  o = i.find("> .w-file-upload-default"),
                  a = i.find("> .w-file-upload-uploading"),
                  u = i.find("> .w-file-upload-success"),
                  s = i.find("> .w-file-upload-error"),
                  f = o.find(".w-file-upload-input"),
                  l = o.find(".w-file-upload-label"),
                  d = l.children(),
                  p = s.find(".w-file-upload-error-msg"),
                  v = u.find(".w-file-upload-file"),
                  h = u.find(".w-file-remove-link"),
                  E = v.find(".w-file-upload-file-name"),
                  _ = p.attr("data-w-size-error"),
                  y = p.attr("data-w-type-error"),
                  m = p.attr("data-w-generic-error");
                if (g)
                  f.on("click", function (t) {
                    t.preventDefault();
                  }),
                    l.on("click", function (t) {
                      t.preventDefault();
                    }),
                    d.on("click", function (t) {
                      t.preventDefault();
                    });
                else {
                  h.on("click", function () {
                    f.removeAttr("data-value"),
                      f.val(""),
                      E.html(""),
                      o.toggle(!0),
                      u.toggle(!1);
                  }),
                    f.on("change", function (i) {
                      (r = i.target && i.target.files && i.target.files[0]) &&
                        (o.toggle(!1),
                        s.toggle(!1),
                        a.toggle(!0),
                        E.text(r.name),
                        S() || b(n),
                        (n.fileUploads[e].uploading = !0),
                        (function (e, n) {
                          var r = { name: e.name, size: e.size };
                          t.ajax({
                            type: "POST",
                            url: c,
                            data: r,
                            dataType: "json",
                            crossDomain: !0,
                          })
                            .done(function (t) {
                              n(null, t);
                            })
                            .fail(function (t) {
                              n(t);
                            });
                        })(r, w));
                    });
                  var T = l.outerHeight();
                  f.height(T), f.width(1);
                }
                function O(t) {
                  var r = t.responseJSON && t.responseJSON.msg,
                    i = m;
                  "string" == typeof r &&
                  0 === r.indexOf("InvalidFileTypeError")
                    ? (i = y)
                    : "string" == typeof r &&
                      0 === r.indexOf("MaxFileSizeError") &&
                      (i = _),
                    p.text(i),
                    f.removeAttr("data-value"),
                    f.val(""),
                    a.toggle(!1),
                    o.toggle(!0),
                    s.toggle(!0),
                    (n.fileUploads[e].uploading = !1),
                    S() || I(n);
                }
                function w(e, n) {
                  if (e) return O(e);
                  var i = n.fileName,
                    o = n.postData,
                    a = n.fileId,
                    u = n.s3Url;
                  f.attr("data-value", a),
                    (function (e, n, r, i, o) {
                      var a = new FormData();
                      for (var u in n) a.append(u, n[u]);
                      a.append("file", r, i),
                        t
                          .ajax({
                            type: "POST",
                            url: e,
                            data: a,
                            processData: !1,
                            contentType: !1,
                          })
                          .done(function () {
                            o(null);
                          })
                          .fail(function (t) {
                            o(t);
                          });
                    })(u, o, r, i, A);
                }
                function A(t) {
                  if (t) return O(t);
                  a.toggle(!1),
                    u.css("display", "inline-block"),
                    (n.fileUploads[e].uploading = !1),
                    S() || I(n);
                }
                function S() {
                  var t = (n.fileUploads && n.fileUploads.toArray()) || [];
                  return t.some(function (t) {
                    return t.uploading;
                  });
                }
              })(e, i);
            });
          var u = (i.action = r.attr("action"));
          (i.handler = null),
            (i.redirect = r.attr("data-redirect")),
            _.test(u) ? (i.handler = w) : u || (o ? (i.handler = O) : y());
        }
        function I(t) {
          var e = (t.btn = t.form.find(':input[type="submit"]'));
          (t.wait = t.btn.attr("data-wait") || null),
            (t.success = !1),
            e.prop("disabled", !1),
            t.label && e.val(t.label);
        }
        function b(t) {
          var e = t.btn,
            n = t.wait;
          e.prop("disabled", !0), n && ((t.label = e.val()), e.val(n));
        }
        function T(e, n) {
          var r = null;
          return (
            (n = n || {}),
            e
              .find(':input:not([type="submit"]):not([type="file"])')
              .each(function (i, o) {
                var a = t(o),
                  u = a.attr("type"),
                  c =
                    a.attr("data-name") || a.attr("name") || "Field " + (i + 1),
                  s = a.val();
                if ("checkbox" === u) s = a.is(":checked");
                else if ("radio" === u) {
                  if (null === n[c] || "string" == typeof n[c]) return;
                  s =
                    e
                      .find('input[name="' + a.attr("name") + '"]:checked')
                      .val() || null;
                }
                "string" == typeof s && (s = t.trim(s)),
                  (n[c] = s),
                  (r =
                    r ||
                    (function (t, e, n, r) {
                      var i = null;
                      "password" === e
                        ? (i = "Passwords cannot be submitted.")
                        : t.attr("required")
                        ? r
                          ? v.test(t.attr("type")) &&
                            (h.test(r) ||
                              (i =
                                "Please enter a valid email address for: " + n))
                          : (i = "Please fill out the required field: " + n)
                        : "g-recaptcha-response" !== n ||
                          r ||
                          (i = "Please confirm youâ€™re not a robot.");
                      return i;
                    })(a, u, c, s));
              }),
            r
          );
        }
        function O(e) {
          I(e);
          var n = e.form,
            r = {
              name: n.attr("data-name") || n.attr("name") || "Untitled Form",
              source: l.href,
              test: i.env(),
              fields: {},
              fileUploads: {},
              dolphin: /pass[\s-_]?(word|code)|secret|login|credentials/i.test(
                n.html()
              ),
            };
          S(e);
          var a = T(n, r.fields);
          if (a) return E(a);
          (r.fileUploads = (function (e) {
            var n = {};
            return (
              e.find(':input[type="file"]').each(function (e, r) {
                var i = t(r),
                  o =
                    i.attr("data-name") || i.attr("name") || "File " + (e + 1),
                  a = i.attr("data-value");
                "string" == typeof a && (a = t.trim(a)), (n[o] = a);
              }),
              n
            );
          })(n)),
            b(e),
            o
              ? t
                  .ajax({
                    url: u,
                    type: "POST",
                    data: r,
                    dataType: "json",
                    crossDomain: !0,
                  })
                  .done(function (t) {
                    t && 200 === t.code && (e.success = !0), A(e);
                  })
                  .fail(function () {
                    A(e);
                  })
              : A(e);
        }
        function w(n) {
          I(n);
          var r = n.form,
            i = {};
          if (!/^https/.test(l.href) || /^https/.test(n.action)) {
            S(n);
            var o,
              a = T(r, i);
            if (a) return E(a);
            b(n),
              e.each(i, function (t, e) {
                v.test(e) && (i.EMAIL = t),
                  /^((full[ _-]?)?name)$/i.test(e) && (o = t),
                  /^(first[ _-]?name)$/i.test(e) && (i.FNAME = t),
                  /^(last[ _-]?name)$/i.test(e) && (i.LNAME = t);
              }),
              o &&
                !i.FNAME &&
                ((o = o.split(" ")),
                (i.FNAME = o[0]),
                (i.LNAME = i.LNAME || o[1]));
            var u = n.action.replace("/post?", "/post-json?") + "&c=?",
              c = u.indexOf("u=") + 2;
            c = u.substring(c, u.indexOf("&", c));
            var s = u.indexOf("id=") + 3;
            (s = u.substring(s, u.indexOf("&", s))),
              (i["b_" + c + "_" + s] = ""),
              t
                .ajax({ url: u, data: i, dataType: "jsonp" })
                .done(function (t) {
                  (n.success = "success" === t.result || /already/.test(t.msg)),
                    n.success || console.info("MailChimp error: " + t.msg),
                    A(n);
                })
                .fail(function () {
                  A(n);
                });
          } else r.attr("method", "post");
        }
        function A(t) {
          var e = t.form,
            n = t.redirect,
            r = t.success;
          r && n
            ? i.location(n)
            : (t.done.toggle(r), t.fail.toggle(!r), e.toggle(!r), I(t));
        }
        function S(t) {
          t.evt && t.evt.preventDefault(), (t.evt = null);
        }
        return (
          (s.ready =
            s.design =
            s.preview =
              function () {
                !(function () {
                  (o = t("html").attr("data-wf-site")),
                    (u = "https://webflow.com/api/v1/form/" + o),
                    d &&
                      u.indexOf("https://webflow.com") >= 0 &&
                      (u = u.replace(
                        "https://webflow.com",
                        "http://formdata.webflow.com"
                      ));
                  if (
                    ((c = "".concat(u, "/signFile")),
                    !(n = t(p + " form")).length)
                  )
                    return;
                  n.each(m);
                })(),
                  g ||
                    a ||
                    (function () {
                      (a = !0),
                        f.on("submit", p + " form", function (e) {
                          var n = t.data(this, p);
                          n.handler && ((n.evt = e), n.handler(n));
                        });
                      var e = [
                        ["checkbox", ".w-checkbox-input"],
                        ["radio", ".w-radio-input"],
                      ];
                      f.on(
                        "change",
                        p +
                          ' form input[type="checkbox"]:not(.w-checkbox-input)',
                        function (e) {
                          t(e.target)
                            .siblings(".w-checkbox-input")
                            .toggleClass("w--redirected-checked");
                        }
                      ),
                        f.on(
                          "change",
                          p + ' form input[type="radio"]',
                          function (e) {
                            t(
                              'input[name="'
                                .concat(e.target.name, '"]:not(')
                                .concat(".w-checkbox-input", ")")
                            ).map(function (e, n) {
                              return t(n)
                                .siblings(".w-radio-input")
                                .removeClass("w--redirected-checked");
                            });
                            var n = t(e.target);
                            n.hasClass("w-radio-input") ||
                              n
                                .siblings(".w-radio-input")
                                .addClass("w--redirected-checked");
                          }
                        ),
                        e.forEach(function (e) {
                          var n = (0, r.default)(e, 2),
                            i = n[0],
                            o = n[1];
                          f.on(
                            "focus",
                            p +
                              ' form input[type="'.concat(i, '"]:not(') +
                              o +
                              ")",
                            function (e) {
                              t(e.target)
                                .siblings(o)
                                .addClass("w--redirected-focus");
                            }
                          ),
                            f.on(
                              "blur",
                              p +
                                ' form input[type="'.concat(i, '"]:not(') +
                                o +
                                ")",
                              function (e) {
                                t(e.target)
                                  .siblings(o)
                                  .removeClass("w--redirected-focus");
                              }
                            );
                        });
                    })();
              }),
          s
        );
      })
    );
  },
  function (t, e, n) {
    var r = n(319),
      i = n(320),
      o = n(321);
    t.exports = function (t, e) {
      return r(t) || i(t, e) || o();
    };
  },
  function (t, e) {
    t.exports = function (t) {
      if (Array.isArray(t)) return t;
    };
  },
  function (t, e) {
    t.exports = function (t, e) {
      var n = [],
        r = !0,
        i = !1,
        o = void 0;
      try {
        for (
          var a, u = t[Symbol.iterator]();
          !(r = (a = u.next()).done) && (n.push(a.value), !e || n.length !== e);
          r = !0
        );
      } catch (t) {
        (i = !0), (o = t);
      } finally {
        try {
          r || null == u.return || u.return();
        } finally {
          if (i) throw o;
        }
      }
      return n;
    };
  },
  function (t, e) {
    t.exports = function () {
      throw new TypeError(
        "Invalid attempt to destructure non-iterable instance"
      );
    };
  },
  function (t, e, n) {
    "use strict";
    var r = n(3),
      i = n(13),
      o = {
        ARROW_LEFT: 37,
        ARROW_UP: 38,
        ARROW_RIGHT: 39,
        ARROW_DOWN: 40,
        ESCAPE: 27,
        SPACE: 32,
        ENTER: 13,
        HOME: 36,
        END: 35,
      };
    r.define(
      "navbar",
      (t.exports = function (t, e) {
        var n,
          a,
          u,
          c,
          s = {},
          f = t.tram,
          l = t(window),
          d = t(document),
          p = e.debounce,
          v = r.env(),
          h = '<div class="w-nav-overlay" data-wf-ignore />',
          E = ".w-nav",
          g = "w--open",
          _ = "w--nav-dropdown-open",
          y = "w--nav-dropdown-toggle-open",
          m = "w--nav-dropdown-list-open",
          I = "w--nav-link-open",
          b = i.triggers,
          T = t();
        function O() {
          r.resize.off(w);
        }
        function w() {
          a.each(M);
        }
        function A(n, r) {
          var i = t(r),
            a = t.data(r, E);
          a ||
            (a = t.data(r, E, {
              open: !1,
              el: i,
              config: {},
              selectedIdx: -1,
            })),
            (a.menu = i.find(".w-nav-menu")),
            (a.links = a.menu.find(".w-nav-link")),
            (a.dropdowns = a.menu.find(".w-dropdown")),
            (a.dropdownToggle = a.menu.find(".w-dropdown-toggle")),
            (a.dropdownList = a.menu.find(".w-dropdown-list")),
            (a.button = i.find(".w-nav-button")),
            (a.container = i.find(".w-container")),
            (a.overlayContainerId = "w-nav-overlay-" + n),
            (a.outside = (function (e) {
              e.outside && d.off("click" + E, e.outside);
              return function (n) {
                var r = t(n.target);
                (c && r.closest(".w-editor-bem-EditorOverlay").length) ||
                  P(e, r);
              };
            })(a));
          var s = i.find(".w-nav-brand");
          s &&
            "/" === s.attr("href") &&
            null == s.attr("aria-label") &&
            s.attr("aria-label", "home"),
            a.button.attr("style", "-webkit-user-select: text;"),
            null == a.button.attr("aria-label") &&
              a.button.attr("aria-label", "menu"),
            a.button.attr("role", "button"),
            a.button.attr("tabindex", "0"),
            a.button.attr("aria-controls", a.overlayContainerId),
            a.button.attr("aria-haspopup", "menu"),
            a.button.attr("aria-expanded", "false"),
            a.el.off(E),
            a.button.off(E),
            a.menu.off(E),
            R(a),
            u
              ? (x(a),
                a.el.on(
                  "setting" + E,
                  (function (t) {
                    return function (n, r) {
                      r = r || {};
                      var i = l.width();
                      R(t),
                        !0 === r.open && G(t, !0),
                        !1 === r.open && U(t, !0),
                        t.open &&
                          e.defer(function () {
                            i !== l.width() && N(t);
                          });
                    };
                  })(a)
                ))
              : (!(function (e) {
                  if (e.overlay) return;
                  (e.overlay = t(h).appendTo(e.el)),
                    e.overlay.attr("id", e.overlayContainerId),
                    (e.parent = e.menu.parent()),
                    U(e, !0);
                })(a),
                a.button.on("click" + E, L(a)),
                a.menu.on("click" + E, "a", D(a)),
                a.button.on(
                  "keydown" + E,
                  (function (t) {
                    return function (e) {
                      switch (e.keyCode) {
                        case o.SPACE:
                        case o.ENTER:
                          return (
                            L(t)(), e.preventDefault(), e.stopPropagation()
                          );
                        case o.ESCAPE:
                          return U(t), e.preventDefault(), e.stopPropagation();
                        case o.ARROW_RIGHT:
                        case o.ARROW_DOWN:
                        case o.HOME:
                        case o.END:
                          return t.open
                            ? (e.keyCode === o.END
                                ? (t.selectedIdx = t.links.length - 1)
                                : (t.selectedIdx = 0),
                              C(t),
                              e.preventDefault(),
                              e.stopPropagation())
                            : (e.preventDefault(), e.stopPropagation());
                      }
                    };
                  })(a)
                ),
                a.el.on(
                  "keydown" + E,
                  (function (t) {
                    return function (e) {
                      if (t.open)
                        switch (
                          ((t.selectedIdx = t.links.index(
                            document.activeElement
                          )),
                          e.keyCode)
                        ) {
                          case o.HOME:
                          case o.END:
                            return (
                              e.keyCode === o.END
                                ? (t.selectedIdx = t.links.length - 1)
                                : (t.selectedIdx = 0),
                              C(t),
                              e.preventDefault(),
                              e.stopPropagation()
                            );
                          case o.ESCAPE:
                            return (
                              U(t),
                              t.button.focus(),
                              e.preventDefault(),
                              e.stopPropagation()
                            );
                          case o.ARROW_LEFT:
                          case o.ARROW_UP:
                            return (
                              (t.selectedIdx = Math.max(-1, t.selectedIdx - 1)),
                              C(t),
                              e.preventDefault(),
                              e.stopPropagation()
                            );
                          case o.ARROW_RIGHT:
                          case o.ARROW_DOWN:
                            return (
                              (t.selectedIdx = Math.min(
                                t.links.length - 1,
                                t.selectedIdx + 1
                              )),
                              C(t),
                              e.preventDefault(),
                              e.stopPropagation()
                            );
                        }
                    };
                  })(a)
                )),
            M(n, r);
        }
        function S(e, n) {
          var r = t.data(n, E);
          r && (x(r), t.removeData(n, E));
        }
        function x(t) {
          t.overlay && (U(t, !0), t.overlay.remove(), (t.overlay = null));
        }
        function R(t) {
          var n = {},
            r = t.config || {},
            i = (n.animation = t.el.attr("data-animation") || "default");
          (n.animOver = /^over/.test(i)),
            (n.animDirect = /left$/.test(i) ? -1 : 1),
            r.animation !== i && t.open && e.defer(N, t),
            (n.easing = t.el.attr("data-easing") || "ease"),
            (n.easing2 = t.el.attr("data-easing2") || "ease");
          var o = t.el.attr("data-duration");
          (n.duration = null != o ? Number(o) : 400),
            (n.docHeight = t.el.attr("data-doc-height")),
            (t.config = n);
        }
        function C(t) {
          if (t.links[t.selectedIdx]) {
            var e = t.links[t.selectedIdx];
            e.focus(), D(e);
          }
        }
        function N(t) {
          t.open && (U(t, !0), G(t, !0));
        }
        function L(t) {
          return p(function () {
            t.open ? U(t) : G(t);
          });
        }
        function D(e) {
          return function (n) {
            var i = t(this).attr("href");
            r.validClick(n.currentTarget)
              ? i && 0 === i.indexOf("#") && e.open && U(e)
              : n.preventDefault();
          };
        }
        (s.ready =
          s.design =
          s.preview =
            function () {
              if (
                ((u = v && r.env("design")),
                (c = r.env("editor")),
                (n = t(document.body)),
                !(a = d.find(E)).length)
              )
                return;
              a.each(A), O(), r.resize.on(w);
            }),
          (s.destroy = function () {
            (T = t()), O(), a && a.length && a.each(S);
          });
        var P = p(function (t, e) {
          if (t.open) {
            var n = e.closest(".w-nav-menu");
            t.menu.is(n) || U(t);
          }
        });
        function M(e, n) {
          var r = t.data(n, E),
            i = (r.collapsed = "none" !== r.button.css("display"));
          if ((!r.open || i || u || U(r, !0), r.container.length)) {
            var o = (function (e) {
              var n = e.container.css(j);
              "none" === n && (n = "");
              return function (e, r) {
                (r = t(r)).css(j, ""), "none" === r.css(j) && r.css(j, n);
              };
            })(r);
            r.links.each(o), r.dropdowns.each(o);
          }
          r.open && X(r);
        }
        var j = "max-width";
        function F(t, e) {
          e.setAttribute("data-nav-menu-open", "");
        }
        function k(t, e) {
          e.removeAttribute("data-nav-menu-open");
        }
        function G(t, e) {
          if (!t.open) {
            (t.open = !0),
              t.menu.each(F),
              t.links.addClass(I),
              t.dropdowns.addClass(_),
              t.dropdownToggle.addClass(y),
              t.dropdownList.addClass(m),
              t.button.addClass(g);
            var n = t.config;
            ("none" !== n.animation && f.support.transform) || (e = !0);
            var i = X(t),
              o = t.menu.outerHeight(!0),
              a = t.menu.outerWidth(!0),
              c = t.el.height(),
              s = t.el[0];
            if (
              (M(0, s),
              b.intro(0, s),
              r.redraw.up(),
              u || d.on("click" + E, t.outside),
              e)
            )
              v();
            else {
              var l = "transform " + n.duration + "ms " + n.easing;
              if (
                (t.overlay &&
                  ((T = t.menu.prev()), t.overlay.show().append(t.menu)),
                n.animOver)
              )
                return (
                  f(t.menu)
                    .add(l)
                    .set({ x: n.animDirect * a, height: i })
                    .start({ x: 0 })
                    .then(v),
                  void (t.overlay && t.overlay.width(a))
                );
              var p = c + o;
              f(t.menu).add(l).set({ y: -p }).start({ y: 0 }).then(v);
            }
          }
          function v() {
            t.button.attr("aria-expanded", "true");
          }
        }
        function X(t) {
          var e = t.config,
            r = e.docHeight ? d.height() : n.height();
          return (
            e.animOver
              ? t.menu.height(r)
              : "fixed" !== t.el.css("position") && (r -= t.el.outerHeight(!0)),
            t.overlay && t.overlay.height(r),
            r
          );
        }
        function U(t, e) {
          if (t.open) {
            (t.open = !1), t.button.removeClass(g);
            var n = t.config;
            if (
              (("none" === n.animation ||
                !f.support.transform ||
                n.duration <= 0) &&
                (e = !0),
              b.outro(0, t.el[0]),
              d.off("click" + E, t.outside),
              e)
            )
              return f(t.menu).stop(), void c();
            var r = "transform " + n.duration + "ms " + n.easing2,
              i = t.menu.outerHeight(!0),
              o = t.menu.outerWidth(!0),
              a = t.el.height();
            if (n.animOver)
              f(t.menu)
                .add(r)
                .start({ x: o * n.animDirect })
                .then(c);
            else {
              var u = a + i;
              f(t.menu).add(r).start({ y: -u }).then(c);
            }
          }
          function c() {
            t.menu.height(""),
              f(t.menu).set({ x: 0, y: 0 }),
              t.menu.each(k),
              t.links.removeClass(I),
              t.dropdowns.removeClass(_),
              t.dropdownToggle.removeClass(y),
              t.dropdownList.removeClass(m),
              t.overlay &&
                t.overlay.children().length &&
                (T.length ? t.menu.insertAfter(T) : t.menu.prependTo(t.parent),
                t.overlay.attr("style", "").hide()),
              t.el.triggerHandler("w-close"),
              t.button.attr("aria-expanded", "false");
          }
        }
        return s;
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = n(3),
      i = n(13),
      o = {
        ARROW_LEFT: 37,
        ARROW_UP: 38,
        ARROW_RIGHT: 39,
        ARROW_DOWN: 40,
        SPACE: 32,
        ENTER: 13,
        HOME: 36,
        END: 35,
      },
      a =
        'a[href], area[href], [role="button"], input, select, textarea, button, iframe, object, embed, *[tabindex], *[contenteditable]';
    r.define(
      "slider",
      (t.exports = function (t, e) {
        var n,
          u,
          c,
          s,
          f = {},
          l = t.tram,
          d = t(document),
          p = r.env(),
          v = ".w-slider",
          h = '<div class="w-slider-dot" data-wf-ignore />',
          E =
            '<div aria-live="off" aria-atomic="true" class="w-slider-aria-label" data-wf-ignore />',
          g = i.triggers;
        function _() {
          (n = d.find(v)).length &&
            (n.each(I),
            (s = null),
            c || (y(), r.resize.on(m), r.redraw.on(f.redraw)));
        }
        function y() {
          r.resize.off(m), r.redraw.off(f.redraw);
        }
        function m() {
          n.filter(":visible").each(D);
        }
        function I(e, n) {
          var r = t(n),
            i = t.data(n, v);
          i ||
            (i = t.data(n, v, {
              index: 0,
              depth: 1,
              hasFocus: { keyboard: !1, mouse: !1 },
              el: r,
              config: {},
            })),
            (i.mask = r.children(".w-slider-mask")),
            (i.left = r.children(".w-slider-arrow-left")),
            (i.right = r.children(".w-slider-arrow-right")),
            (i.nav = r.children(".w-slider-nav")),
            (i.slides = i.mask.children(".w-slide")),
            i.slides.each(g.reset),
            s && (i.maskWidth = 0),
            void 0 === r.attr("role") && r.attr("role", "region"),
            void 0 === r.attr("aria-label") && r.attr("aria-label", "carousel");
          var o = i.mask.attr("id");
          if (
            (o || ((o = "w-slider-mask-" + e), i.mask.attr("id", o)),
            (i.ariaLiveLabel = t(E).appendTo(i.mask)),
            i.left.attr("role", "button"),
            i.left.attr("tabindex", "0"),
            i.left.attr("aria-controls", o),
            void 0 === i.left.attr("aria-label") &&
              i.left.attr("aria-label", "previous slide"),
            i.right.attr("role", "button"),
            i.right.attr("tabindex", "0"),
            i.right.attr("aria-controls", o),
            void 0 === i.right.attr("aria-label") &&
              i.right.attr("aria-label", "next slide"),
            !l.support.transform)
          )
            return i.left.hide(), i.right.hide(), i.nav.hide(), void (c = !0);
          i.el.off(v),
            i.left.off(v),
            i.right.off(v),
            i.nav.off(v),
            b(i),
            u
              ? (i.el.on("setting" + v, C(i)), R(i), (i.hasTimer = !1))
              : (i.el.on("swipe" + v, C(i)),
                i.left.on("click" + v, A(i)),
                i.right.on("click" + v, S(i)),
                i.left.on("keydown" + v, w(i, A)),
                i.right.on("keydown" + v, w(i, S)),
                i.nav.on("keydown" + v, "> div", C(i)),
                i.config.autoplay &&
                  !i.hasTimer &&
                  ((i.hasTimer = !0), (i.timerCount = 1), x(i)),
                i.el.on("mouseenter" + v, O(i, !0, "mouse")),
                i.el.on("focusin" + v, O(i, !0, "keyboard")),
                i.el.on("mouseleave" + v, O(i, !1, "mouse")),
                i.el.on("focusout" + v, O(i, !1, "keyboard"))),
            i.nav.on("click" + v, "> div", C(i)),
            p ||
              i.mask
                .contents()
                .filter(function () {
                  return 3 === this.nodeType;
                })
                .remove();
          var a = r.filter(":hidden");
          a.show();
          var f = r.parents(":hidden");
          f.show(), D(e, n), a.css("display", ""), f.css("display", "");
        }
        function b(t) {
          var e = { crossOver: 0 };
          (e.animation = t.el.attr("data-animation") || "slide"),
            "outin" === e.animation &&
              ((e.animation = "cross"), (e.crossOver = 0.5)),
            (e.easing = t.el.attr("data-easing") || "ease");
          var n = t.el.attr("data-duration");
          if (
            ((e.duration = null != n ? parseInt(n, 10) : 500),
            T(t.el.attr("data-infinite")) && (e.infinite = !0),
            T(t.el.attr("data-disable-swipe")) && (e.disableSwipe = !0),
            T(t.el.attr("data-hide-arrows"))
              ? (e.hideArrows = !0)
              : t.config.hideArrows && (t.left.show(), t.right.show()),
            T(t.el.attr("data-autoplay")))
          ) {
            (e.autoplay = !0),
              (e.delay = parseInt(t.el.attr("data-delay"), 10) || 2e3),
              (e.timerMax = parseInt(t.el.attr("data-autoplay-limit"), 10));
            var r = "mousedown" + v + " touchstart" + v;
            u ||
              t.el.off(r).one(r, function () {
                R(t);
              });
          }
          var i = t.right.width();
          (e.edge = i ? i + 40 : 100), (t.config = e);
        }
        function T(t) {
          return "1" === t || "true" === t;
        }
        function O(e, n, r) {
          return function (i) {
            if (n) e.hasFocus[r] = n;
            else {
              if (t.contains(e.el.get(0), i.relatedTarget)) return;
              if (
                ((e.hasFocus[r] = n),
                (e.hasFocus.mouse && "keyboard" === r) ||
                  (e.hasFocus.keyboard && "mouse" === r))
              )
                return;
            }
            n
              ? (e.ariaLiveLabel.attr("aria-live", "polite"),
                e.hasTimer && R(e))
              : (e.ariaLiveLabel.attr("aria-live", "off"), e.hasTimer && x(e));
          };
        }
        function w(t, e) {
          return function (n) {
            switch (n.keyCode) {
              case o.SPACE:
              case o.ENTER:
                return e(t)(), n.preventDefault(), n.stopPropagation();
            }
          };
        }
        function A(t) {
          return function () {
            L(t, { index: t.index - 1, vector: -1 });
          };
        }
        function S(t) {
          return function () {
            L(t, { index: t.index + 1, vector: 1 });
          };
        }
        function x(t) {
          R(t);
          var e = t.config,
            n = e.timerMax;
          (n && t.timerCount++ > n) ||
            (t.timerId = window.setTimeout(function () {
              null == t.timerId || u || (S(t)(), x(t));
            }, e.delay));
        }
        function R(t) {
          window.clearTimeout(t.timerId), (t.timerId = null);
        }
        function C(n) {
          return function (i, a) {
            a = a || {};
            var c = n.config;
            if (u && "setting" === i.type) {
              if ("prev" === a.select) return A(n)();
              if ("next" === a.select) return S(n)();
              if ((b(n), P(n), null == a.select)) return;
              !(function (n, r) {
                var i = null;
                r === n.slides.length && (_(), P(n)),
                  e.each(n.anchors, function (e, n) {
                    t(e.els).each(function (e, o) {
                      t(o).index() === r && (i = n);
                    });
                  }),
                  null != i && L(n, { index: i, immediate: !0 });
              })(n, a.select);
            } else {
              if ("swipe" === i.type) {
                if (c.disableSwipe) return;
                if (r.env("editor")) return;
                return "left" === a.direction
                  ? S(n)()
                  : "right" === a.direction
                  ? A(n)()
                  : void 0;
              }
              if (n.nav.has(i.target).length) {
                var s = t(i.target).index();
                if (
                  ("click" === i.type && L(n, { index: s }),
                  "keydown" === i.type)
                )
                  switch (i.keyCode) {
                    case o.ENTER:
                    case o.SPACE:
                      L(n, { index: s }), i.preventDefault();
                      break;
                    case o.ARROW_LEFT:
                    case o.ARROW_UP:
                      N(n.nav, Math.max(s - 1, 0)), i.preventDefault();
                      break;
                    case o.ARROW_RIGHT:
                    case o.ARROW_DOWN:
                      N(n.nav, Math.min(s + 1, n.pages)), i.preventDefault();
                      break;
                    case o.HOME:
                      N(n.nav, 0), i.preventDefault();
                      break;
                    case o.END:
                      N(n.nav, n.pages), i.preventDefault();
                      break;
                    default:
                      return;
                  }
              }
            }
          };
        }
        function N(t, e) {
          var n = t.children().eq(e).focus();
          t.children().not(n);
        }
        function L(e, n) {
          n = n || {};
          var r = e.config,
            i = e.anchors;
          e.previous = e.index;
          var o = n.index,
            c = {};
          o < 0
            ? ((o = i.length - 1),
              r.infinite &&
                ((c.x = -e.endX), (c.from = 0), (c.to = i[0].width)))
            : o >= i.length &&
              ((o = 0),
              r.infinite &&
                ((c.x = i[i.length - 1].width),
                (c.from = -i[i.length - 1].x),
                (c.to = c.from - c.x))),
            (e.index = o);
          var f = e.nav
            .children()
            .eq(o)
            .addClass("w-active")
            .attr("aria-selected", "true")
            .attr("tabindex", "0");
          e.nav
            .children()
            .not(f)
            .removeClass("w-active")
            .attr("aria-selected", "false")
            .attr("tabindex", "-1"),
            r.hideArrows &&
              (e.index === i.length - 1 ? e.right.hide() : e.right.show(),
              0 === e.index ? e.left.hide() : e.left.show());
          var d = e.offsetX || 0,
            p = (e.offsetX = -i[e.index].x),
            v = { x: p, opacity: 1, visibility: "" },
            h = t(i[e.index].els),
            E = t(i[e.previous] && i[e.previous].els),
            _ = e.slides.not(h),
            y = r.animation,
            m = r.easing,
            I = Math.round(r.duration),
            b = n.vector || (e.index > e.previous ? 1 : -1),
            T = "opacity " + I + "ms " + m,
            O = "transform " + I + "ms " + m;
          if (
            (h.find(a).removeAttr("tabindex"),
            h.removeAttr("aria-hidden"),
            h.find("*").removeAttr("aria-hidden"),
            _.find(a).attr("tabindex", "-1"),
            _.attr("aria-hidden", "true"),
            _.find("*").attr("aria-hidden", "true"),
            u || (h.each(g.intro), _.each(g.outro)),
            n.immediate && !s)
          )
            return l(h).set(v), void S();
          if (e.index !== e.previous) {
            if (
              (e.ariaLiveLabel.text(
                "Slide ".concat(o + 1, " of ").concat(i.length, ".")
              ),
              "cross" === y)
            ) {
              var w = Math.round(I - I * r.crossOver),
                A = Math.round(I - w);
              return (
                (T = "opacity " + w + "ms " + m),
                l(E).set({ visibility: "" }).add(T).start({ opacity: 0 }),
                void l(h)
                  .set({ visibility: "", x: p, opacity: 0, zIndex: e.depth++ })
                  .add(T)
                  .wait(A)
                  .then({ opacity: 1 })
                  .then(S)
              );
            }
            if ("fade" === y)
              return (
                l(E).set({ visibility: "" }).stop(),
                void l(h)
                  .set({ visibility: "", x: p, opacity: 0, zIndex: e.depth++ })
                  .add(T)
                  .start({ opacity: 1 })
                  .then(S)
              );
            if ("over" === y)
              return (
                (v = { x: e.endX }),
                l(E).set({ visibility: "" }).stop(),
                void l(h)
                  .set({
                    visibility: "",
                    zIndex: e.depth++,
                    x: p + i[e.index].width * b,
                  })
                  .add(O)
                  .start({ x: p })
                  .then(S)
              );
            r.infinite && c.x
              ? (l(e.slides.not(E))
                  .set({ visibility: "", x: c.x })
                  .add(O)
                  .start({ x: p }),
                l(E)
                  .set({ visibility: "", x: c.from })
                  .add(O)
                  .start({ x: c.to }),
                (e.shifted = E))
              : (r.infinite &&
                  e.shifted &&
                  (l(e.shifted).set({ visibility: "", x: d }),
                  (e.shifted = null)),
                l(e.slides).set({ visibility: "" }).add(O).start({ x: p }));
          }
          function S() {
            (h = t(i[e.index].els)),
              (_ = e.slides.not(h)),
              "slide" !== y && (v.visibility = "hidden"),
              l(_).set(v);
          }
        }
        function D(e, n) {
          var r = t.data(n, v);
          if (r)
            return (function (t) {
              var e = t.mask.width();
              if (t.maskWidth !== e) return (t.maskWidth = e), !0;
              return !1;
            })(r)
              ? P(r)
              : void (
                  u &&
                  (function (e) {
                    var n = 0;
                    if (
                      (e.slides.each(function (e, r) {
                        n += t(r).outerWidth(!0);
                      }),
                      e.slidesWidth !== n)
                    )
                      return (e.slidesWidth = n), !0;
                    return !1;
                  })(r) &&
                  P(r)
                );
        }
        function P(e) {
          var n = 1,
            r = 0,
            i = 0,
            o = 0,
            a = e.maskWidth,
            c = a - e.config.edge;
          c < 0 && (c = 0),
            (e.anchors = [{ els: [], x: 0, width: 0 }]),
            e.slides.each(function (u, s) {
              i - r > c &&
                (n++,
                (r += a),
                (e.anchors[n - 1] = { els: [], x: i, width: 0 })),
                (o = t(s).outerWidth(!0)),
                (i += o),
                (e.anchors[n - 1].width += o),
                e.anchors[n - 1].els.push(s);
              var f = u + 1 + " of " + e.slides.length;
              t(s).attr("aria-label", f), t(s).attr("role", "group");
            }),
            (e.endX = i),
            u && (e.pages = null),
            e.nav.length &&
              e.pages !== n &&
              ((e.pages = n),
              (function (e) {
                var n,
                  r = [],
                  i = e.el.attr("data-nav-spacing");
                i && (i = parseFloat(i) + "px");
                for (var o = 0, a = e.pages; o < a; o++)
                  (n = t(h))
                    .attr("aria-label", "Show slide " + (o + 1) + " of " + a)
                    .attr("aria-selected", "false")
                    .attr("role", "button")
                    .attr("tabindex", "-1"),
                    e.nav.hasClass("w-num") && n.text(o + 1),
                    null != i && n.css({ "margin-left": i, "margin-right": i }),
                    r.push(n);
                e.nav.empty().append(r);
              })(e));
          var s = e.index;
          s >= n && (s = n - 1), L(e, { immediate: !0, index: s });
        }
        return (
          (f.ready = function () {
            (u = r.env("design")), _();
          }),
          (f.design = function () {
            (u = !0), _();
          }),
          (f.preview = function () {
            (u = !1), _();
          }),
          (f.redraw = function () {
            (s = !0), _();
          }),
          (f.destroy = y),
          f
        );
      })
    );
  },
  function (t, e, n) {
    "use strict";
    var r = n(3),
      i = n(13);
    r.define(
      "tabs",
      (t.exports = function (t) {
        var e,
          n,
          o = {},
          a = t.tram,
          u = t(document),
          c = r.env,
          s = c.safari,
          f = c(),
          l = "data-w-tab",
          d = "data-w-pane",
          p = ".w-tabs",
          v = "w--current",
          h = "w--tab-active",
          E = i.triggers,
          g = !1;
        function _() {
          (n = f && r.env("design")),
            (e = u.find(p)).length &&
              (e.each(I),
              r.env("preview") && !g && e.each(m),
              y(),
              r.redraw.on(o.redraw));
        }
        function y() {
          r.redraw.off(o.redraw);
        }
        function m(e, n) {
          var r = t.data(n, p);
          r &&
            (r.links && r.links.each(E.reset),
            r.panes && r.panes.each(E.reset));
        }
        function I(e, r) {
          var i = p.substr(1) + "-" + e,
            o = t(r),
            a = t.data(r, p);
          if (
            (a || (a = t.data(r, p, { el: o, config: {} })),
            (a.current = null),
            (a.tabIdentifier = i + "-" + l),
            (a.paneIdentifier = i + "-" + d),
            (a.menu = o.children(".w-tab-menu")),
            (a.links = a.menu.children(".w-tab-link")),
            (a.content = o.children(".w-tab-content")),
            (a.panes = a.content.children(".w-tab-pane")),
            a.el.off(p),
            a.links.off(p),
            a.menu.attr("role", "tablist"),
            a.links.attr("tabindex", "-1"),
            (function (t) {
              var e = {};
              e.easing = t.el.attr("data-easing") || "ease";
              var n = parseInt(t.el.attr("data-duration-in"), 10);
              n = e.intro = n == n ? n : 0;
              var r = parseInt(t.el.attr("data-duration-out"), 10);
              (r = e.outro = r == r ? r : 0),
                (e.immediate = !n && !r),
                (t.config = e);
            })(a),
            !n)
          ) {
            a.links.on(
              "click" + p,
              (function (t) {
                return function (e) {
                  e.preventDefault();
                  var n = e.currentTarget.getAttribute(l);
                  n && b(t, { tab: n });
                };
              })(a)
            ),
              a.links.on(
                "keydown" + p,
                (function (t) {
                  return function (e) {
                    var n = (function (t) {
                        var e = t.current;
                        return Array.prototype.findIndex.call(
                          t.links,
                          function (t) {
                            return t.getAttribute(l) === e;
                          },
                          null
                        );
                      })(t),
                      r = e.key,
                      i = {
                        ArrowLeft: n - 1,
                        ArrowUp: n - 1,
                        ArrowRight: n + 1,
                        ArrowDown: n + 1,
                        End: t.links.length - 1,
                        Home: 0,
                      };
                    if (r in i) {
                      e.preventDefault();
                      var o = i[r];
                      -1 === o && (o = t.links.length - 1),
                        o === t.links.length && (o = 0);
                      var a = t.links[o],
                        u = a.getAttribute(l);
                      u && b(t, { tab: u });
                    }
                  };
                })(a)
              );
            var u = a.links.filter("." + v).attr(l);
            u && b(a, { tab: u, immediate: !0 });
          }
        }
        function b(e, n) {
          n = n || {};
          var i = e.config,
            o = i.easing,
            u = n.tab;
          if (u !== e.current) {
            var c;
            (e.current = u),
              e.links.each(function (r, o) {
                var a = t(o);
                if (n.immediate || i.immediate) {
                  var s = e.panes[r];
                  o.id || (o.id = e.tabIdentifier + "-" + r),
                    s.id || (s.id = e.paneIdentifier + "-" + r),
                    (o.href = "#" + s.id),
                    o.setAttribute("role", "tab"),
                    o.setAttribute("aria-controls", s.id),
                    o.setAttribute("aria-selected", "false"),
                    s.setAttribute("role", "tabpanel"),
                    s.setAttribute("aria-labelledby", o.id);
                }
                o.getAttribute(l) === u
                  ? ((c = o),
                    a
                      .addClass(v)
                      .removeAttr("tabindex")
                      .attr({ "aria-selected": "true" })
                      .each(E.intro))
                  : a.hasClass(v) &&
                    a
                      .removeClass(v)
                      .attr({ tabindex: "-1", "aria-selected": "false" })
                      .each(E.outro);
              });
            var f = [],
              d = [];
            e.panes.each(function (e, n) {
              var r = t(n);
              n.getAttribute(l) === u ? f.push(n) : r.hasClass(h) && d.push(n);
            });
            var p = t(f),
              _ = t(d);
            if (n.immediate || i.immediate)
              return (
                p.addClass(h).each(E.intro),
                _.removeClass(h),
                void (g || r.redraw.up())
              );
            var y = window.scrollX,
              m = window.scrollY;
            c.focus(),
              window.scrollTo(y, m),
              _.length && i.outro
                ? (_.each(E.outro),
                  a(_)
                    .add("opacity " + i.outro + "ms " + o, { fallback: s })
                    .start({ opacity: 0 })
                    .then(function () {
                      return T(i, _, p);
                    }))
                : T(i, _, p);
          }
        }
        function T(t, e, n) {
          if (
            (e.removeClass(h).css({
              opacity: "",
              transition: "",
              transform: "",
              width: "",
              height: "",
            }),
            n.addClass(h).each(E.intro),
            r.redraw.up(),
            !t.intro)
          )
            return a(n).set({ opacity: 1 });
          a(n)
            .set({ opacity: 0 })
            .redraw()
            .add("opacity " + t.intro + "ms " + t.easing, { fallback: s })
            .start({ opacity: 1 });
        }
        return (
          (o.ready = o.design = o.preview = _),
          (o.redraw = function () {
            (g = !0), _(), (g = !1);
          }),
          (o.destroy = function () {
            (e = u.find(p)).length && (e.each(m), y());
          }),
          o
        );
      })
    );
  },
]);
/**
 * ----------------------------------------------------------------------
 * Webflow: Interactions 2.0: Init
 */
Webflow.require("ix2").init({
  events: {
    "e-5": {
      id: "e-5",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-3",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-6",
        },
      },
      mediaQueries: ["medium", "small", "tiny"],
      target: {
        id: "359ea5ff-c819-bee4-2599-c509a23ce009",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "359ea5ff-c819-bee4-2599-c509a23ce009",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593514398024,
    },
    "e-6": {
      id: "e-6",
      animationType: "custom",
      eventTypeId: "MOUSE_SECOND_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-4",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-5",
        },
      },
      mediaQueries: ["medium", "small", "tiny"],
      target: {
        id: "359ea5ff-c819-bee4-2599-c509a23ce009",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "359ea5ff-c819-bee4-2599-c509a23ce009",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593514398025,
    },
    "e-13": {
      id: "e-13",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-14",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593609221776,
    },
    "e-14": {
      id: "e-14",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-13",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593609221777,
    },
    "e-25": {
      id: "e-25",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-26" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "ddcc72d2-4b55-71e4-2812-1a15d2d42a4c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "ddcc72d2-4b55-71e4-2812-1a15d2d42a4c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593613588171,
    },
    "e-27": {
      id: "e-27",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-28" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "4b143938-9350-8b03-cfa1-061ca03eac26",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "4b143938-9350-8b03-cfa1-061ca03eac26",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593613619516,
    },
    "e-29": {
      id: "e-29",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-15", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "03b11ccd-1f1f-9f0a-f990-b6177d369bfb",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "03b11ccd-1f1f-9f0a-f990-b6177d369bfb",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-15-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593613771216,
    },
    "e-30": {
      id: "e-30",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-15", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "14282949-4872-f3fb-9e69-d75a4f350ace",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "14282949-4872-f3fb-9e69-d75a4f350ace",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-15-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593613922517,
    },
    "e-31": {
      id: "e-31",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-32",
        },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6344b5addf8c3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593614048217,
    },
    "e-32": {
      id: "e-32",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-31",
        },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6344b5addf8c3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593614048219,
    },
    "e-33": {
      id: "e-33",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-34",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343b42ddf8f2",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343b42ddf8f2",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593696853839,
    },
    "e-34": {
      id: "e-34",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-33",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343b42ddf8f2",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343b42ddf8f2",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593696853840,
    },
    "e-35": {
      id: "e-35",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-36" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341280ddf91a|bf38e42f-ff11-d2c9-befb-a9bce884a3ab",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341280ddf91a|bf38e42f-ff11-d2c9-befb-a9bce884a3ab",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593698874042,
    },
    "e-39": {
      id: "e-39",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-40" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "22d0db8c-e42c-43e8-ac15-cd8b4496ac70",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "22d0db8c-e42c-43e8-ac15-cd8b4496ac70",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593702259248,
    },
    "e-41": {
      id: "e-41",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-42" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "22d0db8c-e42c-43e8-ac15-cd8b4496ac7c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "22d0db8c-e42c-43e8-ac15-cd8b4496ac7c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593702268989,
    },
    "e-43": {
      id: "e-43",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-44" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "22d0db8c-e42c-43e8-ac15-cd8b4496ac8c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "22d0db8c-e42c-43e8-ac15-cd8b4496ac8c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593702280229,
    },
    "e-51": {
      id: "e-51",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-18", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "36df7adb-f1f6-5dcd-9d86-5374daa1a659",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "36df7adb-f1f6-5dcd-9d86-5374daa1a659",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-18-p",
          smoothing: 75,
          startsEntering: false,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593723307635,
    },
    "e-52": {
      id: "e-52",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-53",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63474b1ddf907",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63474b1ddf907",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593723634672,
    },
    "e-53": {
      id: "e-53",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-52",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63474b1ddf907",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63474b1ddf907",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593723634674,
    },
    "e-54": {
      id: "e-54",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63474b1ddf907",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63474b1ddf907",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593723990538,
    },
    "e-55": {
      id: "e-55",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-56",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63474b1ddf907",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63474b1ddf907",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593724057720,
    },
    "e-56": {
      id: "e-56",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-55",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63474b1ddf907",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63474b1ddf907",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593724057753,
    },
    "e-57": {
      id: "e-57",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6345d3addf909",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593724378077,
    },
    "e-58": {
      id: "e-58",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-59",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593724392273,
    },
    "e-59": {
      id: "e-59",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-58",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593724392274,
    },
    "e-60": {
      id: "e-60",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-61",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593726875492,
    },
    "e-61": {
      id: "e-61",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-60",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593726875494,
    },
    "e-62": {
      id: "e-62",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634379addf8ff",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593726900590,
    },
    "e-63": {
      id: "e-63",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-64",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593726913589,
    },
    "e-64": {
      id: "e-64",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-63",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593726913623,
    },
    "e-68": {
      id: "e-68",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-69" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "b396a38e-b3e6-6d00-b0ae-4fabf29cd3ac",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "b396a38e-b3e6-6d00-b0ae-4fabf29cd3ac",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593872885052,
    },
    "e-70": {
      id: "e-70",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-71" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "46871e84-cbeb-46aa-ab7c-daf8ec62f93f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "46871e84-cbeb-46aa-ab7c-daf8ec62f93f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593872937657,
    },
    "e-72": {
      id: "e-72",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-73" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "75d4fd92-57dd-15ca-dfd3-52291ea897a5",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "75d4fd92-57dd-15ca-dfd3-52291ea897a5",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593872950548,
    },
    "e-74": {
      id: "e-74",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-75" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "be98d41e-a07b-1df5-6a41-38fef8296dec",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "be98d41e-a07b-1df5-6a41-38fef8296dec",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593872963249,
    },
    "e-76": {
      id: "e-76",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-77" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "57e5d3a1-0f3e-89ce-52a6-44f3d75917cc",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "57e5d3a1-0f3e-89ce-52a6-44f3d75917cc",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593872976246,
    },
    "e-78": {
      id: "e-78",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-79" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "171042a9-0ae5-03bb-6f94-f3d50dbaca1b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "171042a9-0ae5-03bb-6f94-f3d50dbaca1b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593873012797,
    },
    "e-80": {
      id: "e-80",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-81" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "9507eada-df76-57dc-f404-2471a109533f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "9507eada-df76-57dc-f404-2471a109533f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593873021748,
    },
    "e-84": {
      id: "e-84",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-85" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "ba4eda54-7734-1e49-9430-56dee8b8379c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "ba4eda54-7734-1e49-9430-56dee8b8379c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593873043198,
    },
    "e-88": {
      id: "e-88",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6346772ddf8f3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1593875799652,
    },
    "e-89": {
      id: "e-89",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-90",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346772ddf8f3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593876938565,
    },
    "e-90": {
      id: "e-90",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-89",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346772ddf8f3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593876938570,
    },
    "e-91": {
      id: "e-91",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-92",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346772ddf8f3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593877894395,
    },
    "e-92": {
      id: "e-92",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-91",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346772ddf8f3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593877894400,
    },
    "e-93": {
      id: "e-93",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-94" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346772ddf8f3|75e5d58d-76b2-f91d-d40b-d5f296caa71c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3|75e5d58d-76b2-f91d-d40b-d5f296caa71c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593878022965,
    },
    "e-95": {
      id: "e-95",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-96" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346772ddf8f3|86b89abf-b1f3-28fc-0889-bf1a3100d774",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346772ddf8f3|86b89abf-b1f3-28fc-0889-bf1a3100d774",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593878034634,
    },
    "e-97": {
      id: "e-97",
      animationType: "custom",
      eventTypeId: "MOUSE_MOVE",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-33", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "36df7adb-f1f6-5dcd-9d86-5374daa1a64e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "36df7adb-f1f6-5dcd-9d86-5374daa1a64e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-33-p",
          selectedAxis: "X_AXIS",
          basedOn: "ELEMENT",
          reverse: false,
          smoothing: 90,
          restingState: 50,
        },
        {
          continuousParameterGroupId: "a-33-p-2",
          selectedAxis: "Y_AXIS",
          basedOn: "ELEMENT",
          reverse: false,
          smoothing: 90,
          restingState: 50,
        },
      ],
      createdOn: 1593879140319,
    },
    "e-98": {
      id: "e-98",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-99" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|c641cc3c-0a4d-093d-ba46-699f049e9878",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|c641cc3c-0a4d-093d-ba46-699f049e9878",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593943947271,
    },
    "e-102": {
      id: "e-102",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-103" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|62993965-5d71-bd43-c595-ab49012b7806",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|62993965-5d71-bd43-c595-ab49012b7806",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593944605274,
    },
    "e-104": {
      id: "e-104",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-105" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|abf7748e-2b16-399c-f6f5-666d70d93f0f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|abf7748e-2b16-399c-f6f5-666d70d93f0f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593944637248,
    },
    "e-106": {
      id: "e-106",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-107" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|271abfee-5f54-fcaa-96b8-98e9bd56867d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|271abfee-5f54-fcaa-96b8-98e9bd56867d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593945836659,
    },
    "e-108": {
      id: "e-108",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-25",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-109",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|6c7e29af-31fd-2529-3eea-1de8a65f4e02",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|6c7e29af-31fd-2529-3eea-1de8a65f4e02",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593946821398,
    },
    "e-114": {
      id: "e-114",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-26",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-115",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|a4352011-6e7c-1b24-5d73-875a05fdb800",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|a4352011-6e7c-1b24-5d73-875a05fdb800",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593948286872,
    },
    "e-116": {
      id: "e-116",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-27",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-117",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|7989ba60-c618-e906-2c66-effca0fbbcfe",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|7989ba60-c618-e906-2c66-effca0fbbcfe",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1593948321336,
    },
    "e-118": {
      id: "e-118",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-119" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "f6c7cb5d-9190-55e5-56b1-f2432a52b104",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "f6c7cb5d-9190-55e5-56b1-f2432a52b104",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593948701720,
    },
    "e-120": {
      id: "e-120",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-121" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "f6c7cb5d-9190-55e5-56b1-f2432a52b10a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "f6c7cb5d-9190-55e5-56b1-f2432a52b10a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593948701720,
    },
    "e-122": {
      id: "e-122",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-123" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "f6c7cb5d-9190-55e5-56b1-f2432a52b110",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "f6c7cb5d-9190-55e5-56b1-f2432a52b110",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593948701720,
    },
    "e-124": {
      id: "e-124",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-125" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "f6c7cb5d-9190-55e5-56b1-f2432a52b116",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "f6c7cb5d-9190-55e5-56b1-f2432a52b116",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 300,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593948701720,
    },
    "e-126": {
      id: "e-126",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-127" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9|813bfc85-0f8e-93da-6d7b-a2ade09d8a7f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9|813bfc85-0f8e-93da-6d7b-a2ade09d8a7f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593949801429,
    },
    "e-128": {
      id: "e-128",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-129" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|666418d0-b5d3-3ebb-4080-cb9a7d0705c7",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|666418d0-b5d3-3ebb-4080-cb9a7d0705c7",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593951241698,
    },
    "e-130": {
      id: "e-130",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-131" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|666418d0-b5d3-3ebb-4080-cb9a7d0705da",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|666418d0-b5d3-3ebb-4080-cb9a7d0705da",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593951241698,
    },
    "e-132": {
      id: "e-132",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-133" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6340347ddf8d6|be059c05-09d5-ad88-ccce-5c1b95eb2395",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6340347ddf8d6|be059c05-09d5-ad88-ccce-5c1b95eb2395",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1593956615987,
    },
    "e-148": {
      id: "e-148",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-149" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "89dc19f4-a1e9-08e0-19ca-aa27c3d882c6",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "89dc19f4-a1e9-08e0-19ca-aa27c3d882c6",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594033204672,
    },
    "e-150": {
      id: "e-150",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-151",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594033469683,
    },
    "e-151": {
      id: "e-151",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-150",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594033469691,
    },
    "e-152": {
      id: "e-152",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-153",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594033488080,
    },
    "e-153": {
      id: "e-153",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-152",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594033488089,
    },
    "e-198": {
      id: "e-198",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-199" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "e9d1b0b8-eebb-da42-1e3f-b83dbca90a70",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "e9d1b0b8-eebb-da42-1e3f-b83dbca90a70",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594052246011,
    },
    "e-201": {
      id: "e-201",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-202" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "0f435bcd-f5f1-024f-316c-804c2e7bdb48",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "0f435bcd-f5f1-024f-316c-804c2e7bdb48",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594052479502,
    },
    "e-204": {
      id: "e-204",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-205" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|eba275ac-db2b-fa49-71f3-1836fe6845f4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|eba275ac-db2b-fa49-71f3-1836fe6845f4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053250034,
    },
    "e-207": {
      id: "e-207",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-208" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f88",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f88",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053519325,
    },
    "e-209": {
      id: "e-209",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-210" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f8e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f8e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053519325,
    },
    "e-211": {
      id: "e-211",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-212" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f94",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f94",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053519325,
    },
    "e-213": {
      id: "e-213",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-214" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f9a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|941f7a63-10bb-7564-7758-c948cb911f9a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053519325,
    },
    "e-215": {
      id: "e-215",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-216" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|c40e19c0-1a7a-375b-39a5-c4aa9574f44b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|c40e19c0-1a7a-375b-39a5-c4aa9574f44b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053603678,
    },
    "e-217": {
      id: "e-217",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-218" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|3cbe6028-f078-86b4-89be-699706f25224",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|3cbe6028-f078-86b4-89be-699706f25224",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594053665224,
    },
    "e-219": {
      id: "e-219",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-220",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594053953743,
    },
    "e-220": {
      id: "e-220",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-219",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594053953753,
    },
    "e-221": {
      id: "e-221",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-222" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "e9d1b0b8-eebb-da42-1e3f-b83dbca90a77",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "e9d1b0b8-eebb-da42-1e3f-b83dbca90a77",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594055876931,
    },
    "e-223": {
      id: "e-223",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-224" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "0f435bcd-f5f1-024f-316c-804c2e7bdb51",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "0f435bcd-f5f1-024f-316c-804c2e7bdb51",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594055894048,
    },
    "e-227": {
      id: "e-227",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-228" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|eba275ac-db2b-fa49-71f3-1836fe6845ff",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|eba275ac-db2b-fa49-71f3-1836fe6845ff",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594055962206,
    },
    "e-229": {
      id: "e-229",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-230",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594055993970,
    },
    "e-230": {
      id: "e-230",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-229",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594055994011,
    },
    "e-231": {
      id: "e-231",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-232",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ff7dddf89c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ff7dddf89c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594056913134,
    },
    "e-232": {
      id: "e-232",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-231",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ff7dddf89c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ff7dddf89c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594056913146,
    },
    "e-233": {
      id: "e-233",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-234",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ff7dddf89c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ff7dddf89c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594056931983,
    },
    "e-234": {
      id: "e-234",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-233",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ff7dddf89c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ff7dddf89c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594056932029,
    },
    "e-235": {
      id: "e-235",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-52", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634ff7dddf89c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ff7dddf89c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-52-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594056969302,
    },
    "e-236": {
      id: "e-236",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594057076146,
    },
    "e-237": {
      id: "e-237",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-238",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594057084047,
    },
    "e-238": {
      id: "e-238",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-237",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594057084055,
    },
    "e-242": {
      id: "e-242",
      animationType: "custom",
      eventTypeId: "MOUSE_MOVE",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-33", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "aa1d3547-dd18-a72c-1325-d596e704808d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "aa1d3547-dd18-a72c-1325-d596e704808d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-33-p",
          selectedAxis: "X_AXIS",
          basedOn: "ELEMENT",
          reverse: false,
          smoothing: 90,
          restingState: 50,
        },
        {
          continuousParameterGroupId: "a-33-p-2",
          selectedAxis: "Y_AXIS",
          basedOn: "ELEMENT",
          reverse: false,
          smoothing: 90,
          restingState: 50,
        },
      ],
      createdOn: 1594124883267,
    },
    "e-243": {
      id: "e-243",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-244",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634af35ddf901",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125625498,
    },
    "e-244": {
      id: "e-244",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-243",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634af35ddf901",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125625510,
    },
    "e-245": {
      id: "e-245",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-246",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634af35ddf901",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125644488,
    },
    "e-246": {
      id: "e-246",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-245",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634af35ddf901",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125644502,
    },
    "e-247": {
      id: "e-247",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634af35ddf901",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594125666889,
    },
    "e-248": {
      id: "e-248",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-249",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634653addf900",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125710344,
    },
    "e-249": {
      id: "e-249",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-248",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634653addf900",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125710353,
    },
    "e-250": {
      id: "e-250",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-251",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634653addf900",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125758443,
    },
    "e-251": {
      id: "e-251",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-250",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634653addf900",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125758455,
    },
    "e-252": {
      id: "e-252",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634653addf900",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594125783885,
    },
    "e-253": {
      id: "e-253",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-254",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634341addf902",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125802836,
    },
    "e-254": {
      id: "e-254",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-253",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634341addf902",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125802845,
    },
    "e-255": {
      id: "e-255",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-256",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634341addf902",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125817360,
    },
    "e-256": {
      id: "e-256",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-255",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634341addf902",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125817371,
    },
    "e-257": {
      id: "e-257",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634341addf902",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594125835797,
    },
    "e-258": {
      id: "e-258",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-259",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d85ddf903",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125866688,
    },
    "e-259": {
      id: "e-259",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-258",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d85ddf903",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125866697,
    },
    "e-260": {
      id: "e-260",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-261",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d85ddf903",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125882136,
    },
    "e-261": {
      id: "e-261",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-260",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d85ddf903",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125882183,
    },
    "e-262": {
      id: "e-262",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6345d85ddf903",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594125895033,
    },
    "e-263": {
      id: "e-263",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-264",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d0f0ddf904",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125922636,
    },
    "e-264": {
      id: "e-264",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-263",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d0f0ddf904",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125922646,
    },
    "e-265": {
      id: "e-265",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-266",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d0f0ddf904",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125941090,
    },
    "e-266": {
      id: "e-266",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-265",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d0f0ddf904",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125941136,
    },
    "e-267": {
      id: "e-267",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634d0f0ddf904",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594125977836,
    },
    "e-268": {
      id: "e-268",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-269",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63423f2ddf905",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125995847,
    },
    "e-269": {
      id: "e-269",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-268",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63423f2ddf905",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594125995857,
    },
    "e-270": {
      id: "e-270",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-271",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63423f2ddf905",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594126015589,
    },
    "e-271": {
      id: "e-271",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-270",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63423f2ddf905",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594126015600,
    },
    "e-272": {
      id: "e-272",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63423f2ddf905",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594126032391,
    },
    "e-273": {
      id: "e-273",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-274" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff|4dfc3536-ec0d-f110-c8d0-1c31c2daae00",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff|4dfc3536-ec0d-f110-c8d0-1c31c2daae00",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594130515916,
    },
    "e-275": {
      id: "e-275",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-276" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff|616c18c1-33e9-c93c-6843-2acbbf39b6dd",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff|616c18c1-33e9-c93c-6843-2acbbf39b6dd",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594130527442,
    },
    "e-279": {
      id: "e-279",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-280" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff|95283217-a8a6-5bb9-8d88-39f38fdf9ee4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff|95283217-a8a6-5bb9-8d88-39f38fdf9ee4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594131646196,
    },
    "e-281": {
      id: "e-281",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-282" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".sticky-bulletpoint",
        originalId:
          "5f1efa24abe634379addf8ff|b6900d14-3fb4-4c6e-3894-7c4cfbc6e363",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".sticky-bulletpoint",
          originalId:
            "5f1efa24abe634379addf8ff|b6900d14-3fb4-4c6e-3894-7c4cfbc6e363",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594137785480,
    },
    "e-289": {
      id: "e-289",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-290" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "95197ec6-4c0c-50fa-ebbb-1f8f7202632c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "95197ec6-4c0c-50fa-ebbb-1f8f7202632c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594138822830,
    },
    "e-291": {
      id: "e-291",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-292" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "95197ec6-4c0c-50fa-ebbb-1f8f7202632d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "95197ec6-4c0c-50fa-ebbb-1f8f7202632d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594139515518,
    },
    "e-293": {
      id: "e-293",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-294" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "d033450d-f6ef-6588-8a8d-2ba48cba9ad2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "d033450d-f6ef-6588-8a8d-2ba48cba9ad2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594139850689,
    },
    "e-299": {
      id: "e-299",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-300" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "d033450d-f6ef-6588-8a8d-2ba48cba9ade",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "d033450d-f6ef-6588-8a8d-2ba48cba9ade",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594139850689,
    },
    "e-301": {
      id: "e-301",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-302" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "d033450d-f6ef-6588-8a8d-2ba48cba9ad8",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "d033450d-f6ef-6588-8a8d-2ba48cba9ad8",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594140039668,
    },
    "e-303": {
      id: "e-303",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-304" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31b2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31b2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594148186060,
    },
    "e-305": {
      id: "e-305",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-306" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31b8",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31b8",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594148186060,
    },
    "e-307": {
      id: "e-307",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-308" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31be",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31be",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594148186060,
    },
    "e-309": {
      id: "e-309",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-310" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31c4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31c4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594148186060,
    },
    "e-311": {
      id: "e-311",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-312" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31ca",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31ca",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594148186060,
    },
    "e-313": {
      id: "e-313",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-314" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31d0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "64dfd1e1-7fb4-27d3-af1e-6afb3c8e31d0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594148186060,
    },
    "e-315": {
      id: "e-315",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-316" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "bcb37482-b962-95c6-d566-60384c9568eb",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "bcb37482-b962-95c6-d566-60384c9568eb",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594149421565,
    },
    "e-317": {
      id: "e-317",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-318" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "383c9621-44f6-cbc8-86b3-2d7aff44df77",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "383c9621-44f6-cbc8-86b3-2d7aff44df77",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594149449321,
    },
    "e-319": {
      id: "e-319",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-320" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "986a9318-da68-9953-0693-4c41c9cc9035",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "986a9318-da68-9953-0693-4c41c9cc9035",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594149471964,
    },
    "e-321": {
      id: "e-321",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-322" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "6f7eb10c-b2b8-f9af-5235-c768911937ec",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "6f7eb10c-b2b8-f9af-5235-c768911937ec",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594149483015,
    },
    "e-325": {
      id: "e-325",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-326" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "d68d125b-c121-cf53-7481-5c19a22776c2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "d68d125b-c121-cf53-7481-5c19a22776c2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594149509418,
    },
    "e-327": {
      id: "e-327",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-328" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "fe389301-c3f8-415f-1719-d605976b595d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "fe389301-c3f8-415f-1719-d605976b595d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594149523414,
    },
    "e-329": {
      id: "e-329",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-330" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "aa1d3547-dd18-a72c-1325-d596e7048099",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "aa1d3547-dd18-a72c-1325-d596e7048099",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594150451239,
    },
    "e-335": {
      id: "e-335",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-336" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "8ef25637-2e56-11fa-e3c3-40e844055ab5",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "8ef25637-2e56-11fa-e3c3-40e844055ab5",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594198239748,
    },
    "e-337": {
      id: "e-337",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-338",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343bbeddf90a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343bbeddf90a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594199836837,
    },
    "e-338": {
      id: "e-338",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-337",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343bbeddf90a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343bbeddf90a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594199836852,
    },
    "e-339": {
      id: "e-339",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-340",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343bbeddf90a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343bbeddf90a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594199850632,
    },
    "e-340": {
      id: "e-340",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-339",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343bbeddf90a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343bbeddf90a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594199850676,
    },
    "e-341": {
      id: "e-341",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6343bbeddf90a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343bbeddf90a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594199869334,
    },
    "e-342": {
      id: "e-342",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-343",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634245bddf910",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634245bddf910",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202851953,
    },
    "e-343": {
      id: "e-343",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-342",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634245bddf910",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634245bddf910",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202851967,
    },
    "e-344": {
      id: "e-344",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-345",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634245bddf910",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634245bddf910",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202870397,
    },
    "e-345": {
      id: "e-345",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-344",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634245bddf910",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634245bddf910",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202870445,
    },
    "e-346": {
      id: "e-346",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634245bddf910",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634245bddf910",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594202887449,
    },
    "e-347": {
      id: "e-347",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-348",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202908449,
    },
    "e-348": {
      id: "e-348",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-347",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202908495,
    },
    "e-349": {
      id: "e-349",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-350",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202928759,
    },
    "e-350": {
      id: "e-350",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-349",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202928808,
    },
    "e-351": {
      id: "e-351",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63435f8ddf917",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594202952347,
    },
    "e-352": {
      id: "e-352",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-353",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202965598,
    },
    "e-353": {
      id: "e-353",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-352",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202965612,
    },
    "e-354": {
      id: "e-354",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-355",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202984206,
    },
    "e-355": {
      id: "e-355",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-354",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594202984252,
    },
    "e-356": {
      id: "e-356",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6344b01ddf915",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203001648,
    },
    "e-357": {
      id: "e-357",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-358",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203014855,
    },
    "e-358": {
      id: "e-358",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-357",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203014868,
    },
    "e-359": {
      id: "e-359",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-360",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203039643,
    },
    "e-360": {
      id: "e-360",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-359",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203039656,
    },
    "e-361": {
      id: "e-361",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634cda6ddf908",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203053846,
    },
    "e-362": {
      id: "e-362",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-363",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203074694,
    },
    "e-363": {
      id: "e-363",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-362",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203074708,
    },
    "e-364": {
      id: "e-364",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6343956ddf912",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203088194,
    },
    "e-365": {
      id: "e-365",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-366",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203095744,
    },
    "e-366": {
      id: "e-366",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-365",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203095790,
    },
    "e-367": {
      id: "e-367",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-368",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203115408,
    },
    "e-368": {
      id: "e-368",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-367",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203115422,
    },
    "e-369": {
      id: "e-369",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-370",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203133601,
    },
    "e-370": {
      id: "e-370",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-369",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203133614,
    },
    "e-371": {
      id: "e-371",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634253addf90c",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203155346,
    },
    "e-372": {
      id: "e-372",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-373",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203168744,
    },
    "e-373": {
      id: "e-373",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-372",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203168756,
    },
    "e-374": {
      id: "e-374",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6341f75ddf916",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203182013,
    },
    "e-375": {
      id: "e-375",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-376",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203193767,
    },
    "e-376": {
      id: "e-376",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-375",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203193813,
    },
    "e-377": {
      id: "e-377",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-378",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203223597,
    },
    "e-378": {
      id: "e-378",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-377",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203223613,
    },
    "e-379": {
      id: "e-379",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-380",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203236595,
    },
    "e-380": {
      id: "e-380",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-379",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203236643,
    },
    "e-381": {
      id: "e-381",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634c85eddf90b",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203248307,
    },
    "e-382": {
      id: "e-382",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-383",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203263319,
    },
    "e-383": {
      id: "e-383",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-382",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203263332,
    },
    "e-384": {
      id: "e-384",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-385",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203275393,
    },
    "e-385": {
      id: "e-385",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-384",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203275410,
    },
    "e-386": {
      id: "e-386",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6347622ddf911",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203290744,
    },
    "e-387": {
      id: "e-387",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-388",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203306168,
    },
    "e-388": {
      id: "e-388",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-387",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203306182,
    },
    "e-389": {
      id: "e-389",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634349fddf914",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203319674,
    },
    "e-390": {
      id: "e-390",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-391",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203327117,
    },
    "e-391": {
      id: "e-391",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-390",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203327163,
    },
    "e-392": {
      id: "e-392",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-393",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203351054,
    },
    "e-393": {
      id: "e-393",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-392",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203351107,
    },
    "e-394": {
      id: "e-394",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634ee89ddf913",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203365553,
    },
    "e-395": {
      id: "e-395",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-396",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203375547,
    },
    "e-396": {
      id: "e-396",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-395",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203375596,
    },
    "e-397": {
      id: "e-397",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-398",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346194ddf90d",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346194ddf90d",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203394610,
    },
    "e-398": {
      id: "e-398",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-397",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346194ddf90d",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346194ddf90d",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203394626,
    },
    "e-399": {
      id: "e-399",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6346194ddf90d",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346194ddf90d",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594203408746,
    },
    "e-400": {
      id: "e-400",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-401",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346194ddf90d",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346194ddf90d",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203417714,
    },
    "e-401": {
      id: "e-401",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-400",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346194ddf90d",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346194ddf90d",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594203417762,
    },
    "e-402": {
      id: "e-402",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-403",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341280ddf91a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341280ddf91a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214753766,
    },
    "e-403": {
      id: "e-403",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-402",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341280ddf91a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341280ddf91a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214753786,
    },
    "e-404": {
      id: "e-404",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-405",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341280ddf91a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341280ddf91a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214770702,
    },
    "e-405": {
      id: "e-405",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-404",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341280ddf91a",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341280ddf91a",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214770747,
    },
    "e-406": {
      id: "e-406",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-407",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214798883,
    },
    "e-407": {
      id: "e-407",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-406",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214798898,
    },
    "e-408": {
      id: "e-408",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-409",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214812684,
    },
    "e-409": {
      id: "e-409",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-408",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6342f26ddf8e9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342f26ddf8e9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594214812699,
    },
    "e-410": {
      id: "e-410",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-411",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347431ddf8f0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347431ddf8f0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594217690235,
    },
    "e-411": {
      id: "e-411",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-410",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347431ddf8f0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347431ddf8f0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594217690257,
    },
    "e-412": {
      id: "e-412",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-413",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347431ddf8f0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347431ddf8f0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594217703684,
    },
    "e-413": {
      id: "e-413",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-412",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347431ddf8f0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347431ddf8f0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594217703736,
    },
    "e-414": {
      id: "e-414",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-415" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".blog-post",
        originalId:
          "5f1efa24abe6347431ddf8f0|d22e4753-73bd-ee6b-9fa7-e09d6bfd6c88",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".blog-post",
          originalId:
            "5f1efa24abe6347431ddf8f0|d22e4753-73bd-ee6b-9fa7-e09d6bfd6c88",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594217739936,
    },
    "e-416": {
      id: "e-416",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-417",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343b42ddf8f2",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343b42ddf8f2",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594217896933,
    },
    "e-417": {
      id: "e-417",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-416",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343b42ddf8f2",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343b42ddf8f2",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594217896955,
    },
    "e-418": {
      id: "e-418",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-419" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".blog-post-container",
        originalId:
          "5f1efa24abe6343b42ddf8f2|9c5dbd82-575f-b95a-b0ce-f78d37c74e07",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".blog-post-container",
          originalId:
            "5f1efa24abe6343b42ddf8f2|9c5dbd82-575f-b95a-b0ce-f78d37c74e07",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594217924234,
    },
    "e-420": {
      id: "e-420",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-421",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594218178849,
    },
    "e-421": {
      id: "e-421",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-420",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594218178872,
    },
    "e-424": {
      id: "e-424",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-425" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|eab31ab8-dd63-a5fc-2213-93b0b46f6386",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|eab31ab8-dd63-a5fc-2213-93b0b46f6386",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594227147029,
    },
    "e-426": {
      id: "e-426",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-427" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|0149bd23-5862-708b-f97f-27ef857b1ab2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|0149bd23-5862-708b-f97f-27ef857b1ab2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594227159905,
    },
    "e-428": {
      id: "e-428",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-429" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|56cc8b0b-0d8d-1fdf-464c-bd6c34f1e272",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|56cc8b0b-0d8d-1fdf-464c-bd6c34f1e272",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594227170931,
    },
    "e-430": {
      id: "e-430",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-431" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|b6290d9b-2d1a-2871-09f6-042fdb112a3d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|b6290d9b-2d1a-2871-09f6-042fdb112a3d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594227201811,
    },
    "e-432": {
      id: "e-432",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-433" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|62a0ac4a-a515-4c27-5db1-6cefa505cc74",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|62a0ac4a-a515-4c27-5db1-6cefa505cc74",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 500,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594227213197,
    },
    "e-434": {
      id: "e-434",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-435" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|e215841a-1a2d-28e9-e659-4b8003aa4508",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|e215841a-1a2d-28e9-e659-4b8003aa4508",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 600,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594227224052,
    },
    "e-436": {
      id: "e-436",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-437",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6340347ddf8d6",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6340347ddf8d6",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594227323255,
    },
    "e-437": {
      id: "e-437",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-436",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6340347ddf8d6",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6340347ddf8d6",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594227323272,
    },
    "e-439": {
      id: "e-439",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-440",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6340347ddf8d6",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6340347ddf8d6",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594227347200,
    },
    "e-440": {
      id: "e-440",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-439",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6340347ddf8d6",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6340347ddf8d6",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594227347248,
    },
    "e-441": {
      id: "e-441",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-442" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|d0a85990-fc24-180f-35b0-9259d6081bf1",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|d0a85990-fc24-180f-35b0-9259d6081bf1",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594287941558,
    },
    "e-443": {
      id: "e-443",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-444" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|d0a85990-fc24-180f-35b0-9259d6081bf2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|d0a85990-fc24-180f-35b0-9259d6081bf2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594287941558,
    },
    "e-447": {
      id: "e-447",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-448",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594288943202,
    },
    "e-448": {
      id: "e-448",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-447",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594288943224,
    },
    "e-449": {
      id: "e-449",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-450",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594288947107,
    },
    "e-450": {
      id: "e-450",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-449",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594288947129,
    },
    "e-451": {
      id: "e-451",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634189cddf8f7",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594288976746,
    },
    "e-456": {
      id: "e-456",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-457" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "4df5a742-ec91-b875-ad6d-772cbb29023c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "4df5a742-ec91-b875-ad6d-772cbb29023c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594293864439,
    },
    "e-458": {
      id: "e-458",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-459" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|e66ccba5-8547-3c23-f65f-8b3c71ae7519",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|e66ccba5-8547-3c23-f65f-8b3c71ae7519",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594294606947,
    },
    "e-460": {
      id: "e-460",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-461" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|e66ccba5-8547-3c23-f65f-8b3c71ae751a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|e66ccba5-8547-3c23-f65f-8b3c71ae751a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594294606947,
    },
    "e-462": {
      id: "e-462",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-463" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|fa53918c-5410-c166-b0a3-2bd844323651",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|fa53918c-5410-c166-b0a3-2bd844323651",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594294676452,
    },
    "e-464": {
      id: "e-464",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-465" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|fa53918c-5410-c166-b0a3-2bd844323652",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|fa53918c-5410-c166-b0a3-2bd844323652",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594294676452,
    },
    "e-466": {
      id: "e-466",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-467" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|a1edb6f7-aa3d-9fa6-9dee-8e95038b6e94",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|a1edb6f7-aa3d-9fa6-9dee-8e95038b6e94",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594294745697,
    },
    "e-468": {
      id: "e-468",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-469" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|a1edb6f7-aa3d-9fa6-9dee-8e95038b6e95",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|a1edb6f7-aa3d-9fa6-9dee-8e95038b6e95",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594294745697,
    },
    "e-470": {
      id: "e-470",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-471" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634189cddf8f7|be87ff49-cdd0-86fa-1530-77550b8a7239",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|be87ff49-cdd0-86fa-1530-77550b8a7239",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594294864470,
    },
    "e-472": {
      id: "e-472",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-473" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634189cddf8f7|b21e24bf-26eb-5b20-e762-cf219d2a7f9b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|b21e24bf-26eb-5b20-e762-cf219d2a7f9b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594294874074,
    },
    "e-474": {
      id: "e-474",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-475" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634189cddf8f7|ac61c561-e31f-aaa0-3e3d-031a18cbdaa3",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|ac61c561-e31f-aaa0-3e3d-031a18cbdaa3",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594294883973,
    },
    "e-476": {
      id: "e-476",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-477" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634189cddf8f7|49a91aca-a654-3c48-d3e2-212c7da23a16",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|49a91aca-a654-3c48-d3e2-212c7da23a16",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 300,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594294895777,
    },
    "e-478": {
      id: "e-478",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-479" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|e668186e-f156-d69b-a490-7020742fa0fd",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|e668186e-f156-d69b-a490-7020742fa0fd",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594294911575,
    },
    "e-488": {
      id: "e-488",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-489" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|4455918d-1e52-c1bb-b622-5a025b052d8c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|4455918d-1e52-c1bb-b622-5a025b052d8c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594295979570,
    },
    "e-506": {
      id: "e-506",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-507" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfa6",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfa6",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296286854,
    },
    "e-508": {
      id: "e-508",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-509" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfa8",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfa8",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296286854,
    },
    "e-510": {
      id: "e-510",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-511" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfaa",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfaa",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296286854,
    },
    "e-512": {
      id: "e-512",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-513" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfac",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|92705f5b-ba58-1262-3454-52e2d672dfac",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 300,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296286854,
    },
    "e-520": {
      id: "e-520",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-521" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fb0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fb0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594296429508,
    },
    "e-522": {
      id: "e-522",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-523" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fdc",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fdc",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296429508,
    },
    "e-524": {
      id: "e-524",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-525" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fde",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fde",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296429508,
    },
    "e-526": {
      id: "e-526",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-527" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fe0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fe0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296429508,
    },
    "e-528": {
      id: "e-528",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-529" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fe2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|1bf769d8-0d39-01b6-0bd4-73b5f4183fe2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 300,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296429508,
    },
    "e-530": {
      id: "e-530",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-531",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296494043,
    },
    "e-531": {
      id: "e-531",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-530",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296494067,
    },
    "e-532": {
      id: "e-532",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-533",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296508148,
    },
    "e-533": {
      id: "e-533",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-532",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296508204,
    },
    "e-534": {
      id: "e-534",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63412adddf8f9",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594296520557,
    },
    "e-539": {
      id: "e-539",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-540" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|a3d16723-0ccd-4c8e-d614-0f582cc2f27f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|a3d16723-0ccd-4c8e-d614-0f582cc2f27f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594296593611,
    },
    "e-541": {
      id: "e-541",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-542" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|a3d16723-0ccd-4c8e-d614-0f582cc2f280",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|a3d16723-0ccd-4c8e-d614-0f582cc2f280",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594296593611,
    },
    "e-557": {
      id: "e-557",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-558" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1bed",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1bed",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594296710425,
    },
    "e-559": {
      id: "e-559",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-560" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c19",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c19",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296710425,
    },
    "e-561": {
      id: "e-561",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-562" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c1b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c1b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296710425,
    },
    "e-563": {
      id: "e-563",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-564" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c1d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c1d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296710425,
    },
    "e-565": {
      id: "e-565",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-566" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c1f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|724af624-6710-19fb-1d62-75be5b4d1c1f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 300,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594296710425,
    },
    "e-567": {
      id: "e-567",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-568",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296750358,
    },
    "e-568": {
      id: "e-568",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-567",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296750387,
    },
    "e-569": {
      id: "e-569",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-570",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296761907,
    },
    "e-570": {
      id: "e-570",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-569",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594296761934,
    },
    "e-571": {
      id: "e-571",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594296774358,
    },
    "e-574": {
      id: "e-574",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-575" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|e61b2584-0f2f-da96-9408-9886294fa132",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|e61b2584-0f2f-da96-9408-9886294fa132",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594299121173,
    },
    "e-576": {
      id: "e-576",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-577" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|e61b2584-0f2f-da96-9408-9886294fa133",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|e61b2584-0f2f-da96-9408-9886294fa133",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594299121173,
    },
    "e-580": {
      id: "e-580",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-581" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9|c5f386fb-ee88-7e18-37be-c0c53272745a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|c5f386fb-ee88-7e18-37be-c0c53272745a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594299218625,
    },
    "e-584": {
      id: "e-584",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-585" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|f839557b-4c8c-b26d-be08-f3fb56fc9ef4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|f839557b-4c8c-b26d-be08-f3fb56fc9ef4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594299490689,
    },
    "e-586": {
      id: "e-586",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-587" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|39895c2d-7475-71e9-3ba3-68c900e6b312",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|39895c2d-7475-71e9-3ba3-68c900e6b312",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594299629175,
    },
    "e-588": {
      id: "e-588",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-589" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|39895c2d-7475-71e9-3ba3-68c900e6b313",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|39895c2d-7475-71e9-3ba3-68c900e6b313",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594299629175,
    },
    "e-590": {
      id: "e-590",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-591" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba1",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba1",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594300713980,
    },
    "e-592": {
      id: "e-592",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-593" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba3",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba3",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594300713980,
    },
    "e-594": {
      id: "e-594",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-595" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba5",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba5",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594300713980,
    },
    "e-596": {
      id: "e-596",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-597" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba7",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|aa303745-0519-3153-f593-9eae07e3fba7",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 300,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594300713980,
    },
    "e-598": {
      id: "e-598",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-34",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-599",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: true,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594301497428,
    },
    "e-600": {
      id: "e-600",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-601",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594302987984,
    },
    "e-601": {
      id: "e-601",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-600",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594302988012,
    },
    "e-602": {
      id: "e-602",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-603",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594303001694,
    },
    "e-603": {
      id: "e-603",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-602",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594303001722,
    },
    "e-604": {
      id: "e-604",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634d989ddf8fa",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594303018581,
    },
    "e-605": {
      id: "e-605",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-19",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-606",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594303747348,
    },
    "e-606": {
      id: "e-606",
      animationType: "custom",
      eventTypeId: "PAGE_FINISH",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-20",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-605",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594303747376,
    },
    "e-607": {
      id: "e-607",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-608",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594303758992,
    },
    "e-608": {
      id: "e-608",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-607",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594303759017,
    },
    "e-609": {
      id: "e-609",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594303785279,
    },
    "e-618": {
      id: "e-618",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-619" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917|82545871-f4da-fdd0-45a8-117265a121d4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917|82545871-f4da-fdd0-45a8-117265a121d4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305170786,
    },
    "e-620": {
      id: "e-620",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-621" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917|82545871-f4da-fdd0-45a8-117265a121dd",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917|82545871-f4da-fdd0-45a8-117265a121dd",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305170786,
    },
    "e-622": {
      id: "e-622",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-623" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915|06bdc530-23cc-f764-9b88-48def2426a1d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915|06bdc530-23cc-f764-9b88-48def2426a1d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305247997,
    },
    "e-624": {
      id: "e-624",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-625" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915|06bdc530-23cc-f764-9b88-48def2426a26",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915|06bdc530-23cc-f764-9b88-48def2426a26",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305247997,
    },
    "e-628": {
      id: "e-628",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-629" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915|dd410bfa-90b0-98ca-ce1e-b0b277cf16c3",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915|dd410bfa-90b0-98ca-ce1e-b0b277cf16c3",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305257301,
    },
    "e-630": {
      id: "e-630",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-631" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909|7c6470af-66b9-080f-ba98-55ff82486f55",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909|7c6470af-66b9-080f-ba98-55ff82486f55",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305323898,
    },
    "e-632": {
      id: "e-632",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-633" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909|7c6470af-66b9-080f-ba98-55ff82486f5e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909|7c6470af-66b9-080f-ba98-55ff82486f5e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305323898,
    },
    "e-636": {
      id: "e-636",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-637" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909|91b1e6a9-2832-cded-5df1-d19485a90509",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909|91b1e6a9-2832-cded-5df1-d19485a90509",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305334505,
    },
    "e-638": {
      id: "e-638",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-639" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909|b4a2182e-ffb9-43b0-c929-6e4058483c6c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909|b4a2182e-ffb9-43b0-c929-6e4058483c6c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305339908,
    },
    "e-640": {
      id: "e-640",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-641" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908|e66e0c7e-2c38-ebed-c4bf-c6adef2afff6",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908|e66e0c7e-2c38-ebed-c4bf-c6adef2afff6",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305412198,
    },
    "e-642": {
      id: "e-642",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-643" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908|e66e0c7e-2c38-ebed-c4bf-c6adef2affff",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908|e66e0c7e-2c38-ebed-c4bf-c6adef2affff",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305412198,
    },
    "e-644": {
      id: "e-644",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-645" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908|e66e0c7e-2c38-ebed-c4bf-c6adef2b000f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908|e66e0c7e-2c38-ebed-c4bf-c6adef2b000f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305412198,
    },
    "e-646": {
      id: "e-646",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-647" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916|edba2703-2265-6202-0ebc-e3a84ad53b63",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916|edba2703-2265-6202-0ebc-e3a84ad53b63",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305547010,
    },
    "e-648": {
      id: "e-648",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-649" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916|edba2703-2265-6202-0ebc-e3a84ad53b6c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916|edba2703-2265-6202-0ebc-e3a84ad53b6c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305547010,
    },
    "e-650": {
      id: "e-650",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-651" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b|f8424953-e667-34a9-3dc0-1a2d0b47e2c6",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b|f8424953-e667-34a9-3dc0-1a2d0b47e2c6",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305619250,
    },
    "e-652": {
      id: "e-652",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-653" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b|f8424953-e667-34a9-3dc0-1a2d0b47e2cf",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b|f8424953-e667-34a9-3dc0-1a2d0b47e2cf",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305619250,
    },
    "e-656": {
      id: "e-656",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-657" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b|01944826-44cd-c953-5af8-20b49265a60a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b|01944826-44cd-c953-5af8-20b49265a60a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305633503,
    },
    "e-658": {
      id: "e-658",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-659" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b|1330eb18-b997-f9b9-0563-3e1396ef48e5",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b|1330eb18-b997-f9b9-0563-3e1396ef48e5",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305634040,
    },
    "e-660": {
      id: "e-660",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-661" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912|08783e35-3b52-6a3c-1ea5-fc82122e3390",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912|08783e35-3b52-6a3c-1ea5-fc82122e3390",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305710299,
    },
    "e-662": {
      id: "e-662",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-663" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912|08783e35-3b52-6a3c-1ea5-fc82122e3399",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912|08783e35-3b52-6a3c-1ea5-fc82122e3399",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305710299,
    },
    "e-664": {
      id: "e-664",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-665" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c|574e3aac-6668-1ba5-ec23-ea251785b0cc",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c|574e3aac-6668-1ba5-ec23-ea251785b0cc",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305792299,
    },
    "e-666": {
      id: "e-666",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-667" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c|574e3aac-6668-1ba5-ec23-ea251785b0d5",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c|574e3aac-6668-1ba5-ec23-ea251785b0d5",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305792299,
    },
    "e-668": {
      id: "e-668",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-669" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911|750e26d1-d016-dae8-9a47-d84eba1ff663",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911|750e26d1-d016-dae8-9a47-d84eba1ff663",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305973959,
    },
    "e-670": {
      id: "e-670",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-671" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911|750e26d1-d016-dae8-9a47-d84eba1ff66c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911|750e26d1-d016-dae8-9a47-d84eba1ff66c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305973959,
    },
    "e-674": {
      id: "e-674",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-675" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911|862a5a33-ae02-3342-50c9-a63c0a85c8d0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911|862a5a33-ae02-3342-50c9-a63c0a85c8d0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594305986718,
    },
    "e-676": {
      id: "e-676",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-677" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914|34e8ee43-e93d-f778-234b-c542d6392b8f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914|34e8ee43-e93d-f778-234b-c542d6392b8f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306051162,
    },
    "e-678": {
      id: "e-678",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-679" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914|34e8ee43-e93d-f778-234b-c542d6392b98",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914|34e8ee43-e93d-f778-234b-c542d6392b98",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306051162,
    },
    "e-682": {
      id: "e-682",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-683" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914|f8017d89-c895-2110-84d8-fbde2316ccdc",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914|f8017d89-c895-2110-84d8-fbde2316ccdc",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306106156,
    },
    "e-684": {
      id: "e-684",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-685" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914|665bae4e-5f14-31d0-50fa-59d601514c9c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914|665bae4e-5f14-31d0-50fa-59d601514c9c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306111658,
    },
    "e-686": {
      id: "e-686",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-687" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913|fe7e23c7-397a-4010-9a63-b9a6f6394662",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913|fe7e23c7-397a-4010-9a63-b9a6f6394662",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306513455,
    },
    "e-688": {
      id: "e-688",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-689" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913|fe7e23c7-397a-4010-9a63-b9a6f639466b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913|fe7e23c7-397a-4010-9a63-b9a6f639466b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306513455,
    },
    "e-692": {
      id: "e-692",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-693" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913|04c872ca-fff5-ebd8-eb7f-f3aad7aecea7",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913|04c872ca-fff5-ebd8-eb7f-f3aad7aecea7",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306517310,
    },
    "e-694": {
      id: "e-694",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-695" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913|d83082bd-ac51-e9d3-88f8-bf267cf36b71",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913|d83082bd-ac51-e9d3-88f8-bf267cf36b71",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594306517677,
    },
    "e-702": {
      id: "e-702",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-703" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912|f83cbfd3-da8b-9907-f567-9acb9e0a325d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912|f83cbfd3-da8b-9907-f567-9acb9e0a325d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594322251430,
    },
    "e-704": {
      id: "e-704",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-705" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343956ddf912|f83cbfd3-da8b-9907-f567-9acb9e0a325e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343956ddf912|f83cbfd3-da8b-9907-f567-9acb9e0a325e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594322251430,
    },
    "e-706": {
      id: "e-706",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-707" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347c43ddf91c|cfc0daaa-d0de-0e78-2150-a3ffe07652a0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347c43ddf91c|cfc0daaa-d0de-0e78-2150-a3ffe07652a0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594370104808,
    },
    "e-708": {
      id: "e-708",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-709" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "4437c6ea-37b7-49b9-8abf-8c4573457853",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "4437c6ea-37b7-49b9-8abf-8c4573457853",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594372871996,
    },
    "e-710": {
      id: "e-710",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-711" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "4437c6ea-37b7-49b9-8abf-8c4573457854",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "4437c6ea-37b7-49b9-8abf-8c4573457854",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594372871996,
    },
    "e-712": {
      id: "e-712",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-713" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|95162d57-0a16-199a-40b9-fff32ba82a45",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|95162d57-0a16-199a-40b9-fff32ba82a45",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594373490651,
    },
    "e-714": {
      id: "e-714",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-715" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|95162d57-0a16-199a-40b9-fff32ba82a46",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|95162d57-0a16-199a-40b9-fff32ba82a46",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594373490651,
    },
    "e-716": {
      id: "e-716",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_UP",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-13",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-717",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634f267ddf919",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634f267ddf919",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594387722328,
    },
    "e-717": {
      id: "e-717",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL_DOWN",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-14",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-716",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634f267ddf919",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634f267ddf919",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594387722365,
    },
    "e-718": {
      id: "e-718",
      animationType: "custom",
      eventTypeId: "PAGE_SCROLL",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-22", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-22-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594392741190,
    },
    "e-719": {
      id: "e-719",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-720" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "6f0160ef-0ab0-bd6f-b617-46551b7b6ed0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "6f0160ef-0ab0-bd6f-b617-46551b7b6ed0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594472749326,
    },
    "e-721": {
      id: "e-721",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-722" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "6f0160ef-0ab0-bd6f-b617-46551b7b6ed7",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "6f0160ef-0ab0-bd6f-b617-46551b7b6ed7",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594472760023,
    },
    "e-723": {
      id: "e-723",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-724" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "6a46bf0e-526d-110e-4ce0-583bd57689f2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "6a46bf0e-526d-110e-4ce0-583bd57689f2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594584609279,
    },
    "e-725": {
      id: "e-725",
      animationType: "custom",
      eventTypeId: "MOUSE_OVER",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-35",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-726",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".floating-card",
        originalId: "b012fd75-93a1-c60e-1aa5-10a7147f6543",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".floating-card",
          originalId: "b012fd75-93a1-c60e-1aa5-10a7147f6543",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594587909186,
    },
    "e-726": {
      id: "e-726",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-36",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-725",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".floating-card",
        originalId: "b012fd75-93a1-c60e-1aa5-10a7147f6543",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".floating-card",
          originalId: "b012fd75-93a1-c60e-1aa5-10a7147f6543",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594587909216,
    },
    "e-735": {
      id: "e-735",
      animationType: "custom",
      eventTypeId: "PAGE_START",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-34",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-736",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f",
        appliesTo: "PAGE",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f",
          appliesTo: "PAGE",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: true,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594632479648,
    },
    "e-737": {
      id: "e-737",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-738" },
      },
      mediaQueries: ["main", "medium"],
      target: {
        id: "5f1efa24abe63479fbddf84f|9587d2b5-e4f9-7ed0-ad20-ca6b7b8f5589",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|9587d2b5-e4f9-7ed0-ad20-ca6b7b8f5589",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594634587360,
    },
    "e-739": {
      id: "e-739",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-740" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|9587d2b5-e4f9-7ed0-ad20-ca6b7b8f558f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|9587d2b5-e4f9-7ed0-ad20-ca6b7b8f558f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594635066975,
    },
    "e-741": {
      id: "e-741",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-742" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|9587d2b5-e4f9-7ed0-ad20-ca6b7b8f5595",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|9587d2b5-e4f9-7ed0-ad20-ca6b7b8f5595",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594635079057,
    },
    "e-743": {
      id: "e-743",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-744" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347d7eddf8f4|5a6e19ee-199d-2e26-0eca-139ba86e78e9",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347d7eddf8f4|5a6e19ee-199d-2e26-0eca-139ba86e78e9",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594655106330,
    },
    "e-745": {
      id: "e-745",
      animationType: "custom",
      eventTypeId: "MOUSE_OVER",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-37",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-746",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347d7eddf8f4|244c6cd0-0bab-9629-c591-f07443ea33a1",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347d7eddf8f4|244c6cd0-0bab-9629-c591-f07443ea33a1",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594658284097,
    },
    "e-746": {
      id: "e-746",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-38",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-745",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347d7eddf8f4|244c6cd0-0bab-9629-c591-f07443ea33a1",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347d7eddf8f4|244c6cd0-0bab-9629-c591-f07443ea33a1",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594658284137,
    },
    "e-747": {
      id: "e-747",
      animationType: "custom",
      eventTypeId: "MOUSE_OVER",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-37",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-748",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "3e156fb6-4f34-fe8f-49ef-8453f5948284",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "3e156fb6-4f34-fe8f-49ef-8453f5948284",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594658545701,
    },
    "e-748": {
      id: "e-748",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-38",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-747",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "3e156fb6-4f34-fe8f-49ef-8453f5948284",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "3e156fb6-4f34-fe8f-49ef-8453f5948284",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594658545739,
    },
    "e-775": {
      id: "e-775",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-43", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe63479fbddf84f|13467ade-4041-1fdc-4b9d-6ceca6daf0ce",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|13467ade-4041-1fdc-4b9d-6ceca6daf0ce",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-43-p",
          smoothing: 75,
          startsEntering: false,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1594719075143,
    },
    "e-778": {
      id: "e-778",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-42",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-777",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe683",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe683",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594719204754,
    },
    "e-780": {
      id: "e-780",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-42",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-779",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe68e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe68e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594719204754,
    },
    "e-782": {
      id: "e-782",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-42",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-781",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe69c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe69c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594719204754,
    },
    "e-784": {
      id: "e-784",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-42",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-783",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe6a7",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe6a7",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594719204754,
    },
    "e-786": {
      id: "e-786",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-42",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-785",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe6b2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|ddaf241e-8d26-4ca9-0042-66ba10efe6b2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594719204754,
    },
    "e-787": {
      id: "e-787",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-788" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|346da917-5870-a2ec-9112-aad979a3f383",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|346da917-5870-a2ec-9112-aad979a3f383",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594735060819,
    },
    "e-789": {
      id: "e-789",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-790" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|6cb35fa9-39cf-df32-1359-ce551b843867",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|6cb35fa9-39cf-df32-1359-ce551b843867",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594735096713,
    },
    "e-791": {
      id: "e-791",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-792" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634653addf900|4531bcfb-aacd-a20b-f8cf-0acf3fd2cb93",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900|4531bcfb-aacd-a20b-f8cf-0acf3fd2cb93",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594741410322,
    },
    "e-793": {
      id: "e-793",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-794" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634653addf900|4531bcfb-aacd-a20b-f8cf-0acf3fd2cb94",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634653addf900|4531bcfb-aacd-a20b-f8cf-0acf3fd2cb94",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594741410322,
    },
    "e-803": {
      id: "e-803",
      animationType: "custom",
      eventTypeId: "MOUSE_OVER",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-28",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-804",
        },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|86124aef-c39f-34fa-9044-91071ba90257",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|86124aef-c39f-34fa-9044-91071ba90257",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594831923465,
    },
    "e-804": {
      id: "e-804",
      animationType: "custom",
      eventTypeId: "MOUSE_OUT",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-29",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-803",
        },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|86124aef-c39f-34fa-9044-91071ba90257",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|86124aef-c39f-34fa-9044-91071ba90257",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594831923503,
    },
    "e-807": {
      id: "e-807",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-29",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-808",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|511b0e9f-f3e8-35f7-373a-fb033357eb11",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|511b0e9f-f3e8-35f7-373a-fb033357eb11",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594834557223,
    },
    "e-811": {
      id: "e-811",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-812" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 100,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594835113918,
    },
    "e-813": {
      id: "e-813",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-814" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594835113918,
    },
    "e-815": {
      id: "e-815",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-816" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594835113918,
    },
    "e-817": {
      id: "e-817",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-818" },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b6",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|469e8674-1dd0-74cb-6f19-2203c1fa21b6",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 20,
        scrollOffsetUnit: "%",
        delay: 500,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1594835113918,
    },
    "e-819": {
      id: "e-819",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-28",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-820",
        },
      },
      mediaQueries: ["medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|86124aef-c39f-34fa-9044-91071ba90257",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|86124aef-c39f-34fa-9044-91071ba90257",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594905004224,
    },
    "e-830": {
      id: "e-830",
      animationType: "custom",
      eventTypeId: "MOUSE_MOVE",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-45", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff|8a4db051-54ed-df38-1a5f-3de9fd1fd372",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff|8a4db051-54ed-df38-1a5f-3de9fd1fd372",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-45-p",
          selectedAxis: "X_AXIS",
          basedOn: "ELEMENT",
          reverse: false,
          smoothing: 90,
          restingState: 50,
        },
        {
          continuousParameterGroupId: "a-45-p-2",
          selectedAxis: "Y_AXIS",
          basedOn: "ELEMENT",
          reverse: false,
          smoothing: 90,
          restingState: 50,
        },
      ],
      createdOn: 1594908675375,
    },
    "e-833": {
      id: "e-833",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-834" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634189cddf8f7|0340c600-1d6f-9958-e3a7-b1bb1e7c7d12",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634189cddf8f7|0340c600-1d6f-9958-e3a7-b1bb1e7c7d12",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910578884,
    },
    "e-835": {
      id: "e-835",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-836" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d9d4ddf8f8|8ba06d8a-2aca-b4a1-65a3-2feafbf19c38",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d9d4ddf8f8|8ba06d8a-2aca-b4a1-65a3-2feafbf19c38",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910646662,
    },
    "e-837": {
      id: "e-837",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-838" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63412adddf8f9|53593be5-f662-8ff8-59a0-48cd34959f6a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63412adddf8f9|53593be5-f662-8ff8-59a0-48cd34959f6a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910678311,
    },
    "e-839": {
      id: "e-839",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-840" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d989ddf8fa|7e0bc261-d7b6-12d8-6693-c08fe48a459b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d989ddf8fa|7e0bc261-d7b6-12d8-6693-c08fe48a459b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910703712,
    },
    "e-841": {
      id: "e-841",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-842" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344d49ddf8fb|a3838b24-c271-f727-0019-9cd31d8a0f52",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344d49ddf8fb|a3838b24-c271-f727-0019-9cd31d8a0f52",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910724872,
    },
    "e-843": {
      id: "e-843",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-844" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634379addf8ff|f4ddbe9a-aab4-b2d5-4bca-6737a42a2326",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634379addf8ff|f4ddbe9a-aab4-b2d5-4bca-6737a42a2326",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910746378,
    },
    "e-845": {
      id: "e-845",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-846" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "aa1d3547-dd18-a72c-1325-d596e7048091",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "aa1d3547-dd18-a72c-1325-d596e7048091",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: null,
        effectIn: true,
      },
      createdOn: 1594910773164,
    },
    "e-847": {
      id: "e-847",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-48",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-848",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|2b1b5647-d677-8a76-d8bd-934915ce3a27",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|2b1b5647-d677-8a76-d8bd-934915ce3a27",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1594989229019,
    },
    "e-891": {
      id: "e-891",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-892" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634af35ddf901|0996d568-52f7-23c3-39c6-327ce3604933",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901|0996d568-52f7-23c3-39c6-327ce3604933",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595098975127,
    },
    "e-893": {
      id: "e-893",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-894" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634af35ddf901|0996d568-52f7-23c3-39c6-327ce3604934",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634af35ddf901|0996d568-52f7-23c3-39c6-327ce3604934",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595098975127,
    },
    "e-895": {
      id: "e-895",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-896" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634341addf902|8f15bda0-5e3c-5d94-81b2-477f60a1f66c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902|8f15bda0-5e3c-5d94-81b2-477f60a1f66c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099058061,
    },
    "e-897": {
      id: "e-897",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-898" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634341addf902|8f15bda0-5e3c-5d94-81b2-477f60a1f66d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634341addf902|8f15bda0-5e3c-5d94-81b2-477f60a1f66d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099058061,
    },
    "e-899": {
      id: "e-899",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-900" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d85ddf903|c8264ae8-ef12-b3ba-a704-76533b40b538",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903|c8264ae8-ef12-b3ba-a704-76533b40b538",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099077172,
    },
    "e-901": {
      id: "e-901",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-902" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d85ddf903|c8264ae8-ef12-b3ba-a704-76533b40b539",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d85ddf903|c8264ae8-ef12-b3ba-a704-76533b40b539",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099077172,
    },
    "e-903": {
      id: "e-903",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-904" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63423f2ddf905|e85771a6-b134-4a6f-078b-ffb31fa4bee2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905|e85771a6-b134-4a6f-078b-ffb31fa4bee2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099100948,
    },
    "e-905": {
      id: "e-905",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-906" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63423f2ddf905|e85771a6-b134-4a6f-078b-ffb31fa4bee3",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63423f2ddf905|e85771a6-b134-4a6f-078b-ffb31fa4bee3",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099100948,
    },
    "e-907": {
      id: "e-907",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-908" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d0f0ddf904|06a3e9b5-6687-dc75-edf2-b5c8a0241561",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904|06a3e9b5-6687-dc75-edf2-b5c8a0241561",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099168211,
    },
    "e-909": {
      id: "e-909",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-910" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634d0f0ddf904|06a3e9b5-6687-dc75-edf2-b5c8a0241562",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634d0f0ddf904|06a3e9b5-6687-dc75-edf2-b5c8a0241562",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099168211,
    },
    "e-911": {
      id: "e-911",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-912" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|d4ba87b5-2e94-02be-96e3-ee60bbed0340",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|d4ba87b5-2e94-02be-96e3-ee60bbed0340",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099238583,
    },
    "e-914": {
      id: "e-914",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-915" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|61178cd2-1112-6156-c955-8d43bd3e82b4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|61178cd2-1112-6156-c955-8d43bd3e82b4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099249373,
    },
    "e-919": {
      id: "e-919",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-920" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|61178cd2-1112-6156-c955-8d43bd3e82bb",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|61178cd2-1112-6156-c955-8d43bd3e82bb",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099249373,
    },
    "e-921": {
      id: "e-921",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-922" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|b2f251ac-6784-0618-d0e6-24198553febd",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|b2f251ac-6784-0618-d0e6-24198553febd",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099254988,
    },
    "e-924": {
      id: "e-924",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-925" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|6816278c-26cd-3297-fd59-03a3a905a89b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|6816278c-26cd-3297-fd59-03a3a905a89b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099259226,
    },
    "e-929": {
      id: "e-929",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-930" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|6816278c-26cd-3297-fd59-03a3a905a8a2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|6816278c-26cd-3297-fd59-03a3a905a8a2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099259226,
    },
    "e-931": {
      id: "e-931",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-932" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|35e21025-fd7a-7bf1-69a1-065bd1528518",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|35e21025-fd7a-7bf1-69a1-065bd1528518",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099264261,
    },
    "e-934": {
      id: "e-934",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-935" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|0e3c977c-970e-4d44-414c-2ebfc469d051",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|0e3c977c-970e-4d44-414c-2ebfc469d051",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099269223,
    },
    "e-939": {
      id: "e-939",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-940" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|0e3c977c-970e-4d44-414c-2ebfc469d058",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|0e3c977c-970e-4d44-414c-2ebfc469d058",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099269223,
    },
    "e-941": {
      id: "e-941",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-942" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|ecbed3e1-aad5-ac9e-1e5a-1f89c78330e8",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|ecbed3e1-aad5-ac9e-1e5a-1f89c78330e8",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099279930,
    },
    "e-944": {
      id: "e-944",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-945" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|45ea2f9f-1776-efec-6a69-a811e2600d73",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|45ea2f9f-1776-efec-6a69-a811e2600d73",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099285016,
    },
    "e-949": {
      id: "e-949",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-950" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|45ea2f9f-1776-efec-6a69-a811e2600d7a",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|45ea2f9f-1776-efec-6a69-a811e2600d7a",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099285016,
    },
    "e-959": {
      id: "e-959",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-960" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|f8c39d1e-df33-5712-357b-067ffe606649",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|f8c39d1e-df33-5712-357b-067ffe606649",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099296297,
    },
    "e-961": {
      id: "e-961",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-962" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|9c6ef251-a85c-b78c-aab3-5ca53ef6bd15",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|9c6ef251-a85c-b78c-aab3-5ca53ef6bd15",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099300693,
    },
    "e-964": {
      id: "e-964",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-965" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|ae1317dd-139d-1861-5b68-d7b209d7210d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|ae1317dd-139d-1861-5b68-d7b209d7210d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099304469,
    },
    "e-969": {
      id: "e-969",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-970" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|ae1317dd-139d-1861-5b68-d7b209d72114",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|ae1317dd-139d-1861-5b68-d7b209d72114",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099304469,
    },
    "e-971": {
      id: "e-971",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-972" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|df67436e-10c4-8f4e-bf99-96de9b48712f",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|df67436e-10c4-8f4e-bf99-96de9b48712f",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595099312339,
    },
    "e-974": {
      id: "e-974",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-975" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63474b1ddf907|3900c190-aa52-4fd5-748f-9ac9faa3c360",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63474b1ddf907|3900c190-aa52-4fd5-748f-9ac9faa3c360",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099654425,
    },
    "e-976": {
      id: "e-976",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-977" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6343bbeddf90a|5855ba57-f871-dfbb-0c39-ce9824d6583d",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6343bbeddf90a|5855ba57-f871-dfbb-0c39-ce9824d6583d",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099685133,
    },
    "e-978": {
      id: "e-978",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-979" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634245bddf910|ad982476-8973-bae0-4ff6-86e4aa9f3567",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634245bddf910|ad982476-8973-bae0-4ff6-86e4aa9f3567",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099699381,
    },
    "e-980": {
      id: "e-980",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-981" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63435f8ddf917|3ed74e87-9293-bb65-bb43-99d288d7c046",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63435f8ddf917|3ed74e87-9293-bb65-bb43-99d288d7c046",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099714554,
    },
    "e-982": {
      id: "e-982",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-983" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b01ddf915|d59fd004-dae6-7d8c-0dac-1592848f6ff3",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b01ddf915|d59fd004-dae6-7d8c-0dac-1592848f6ff3",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099734507,
    },
    "e-984": {
      id: "e-984",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-985" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634cda6ddf908|8f441087-dd0a-3088-9fe0-35b550e0ae53",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634cda6ddf908|8f441087-dd0a-3088-9fe0-35b550e0ae53",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099747250,
    },
    "e-986": {
      id: "e-986",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-987" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6345d3addf909|5121d25a-5bc5-3798-4183-858c61ac6c4e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6345d3addf909|5121d25a-5bc5-3798-4183-858c61ac6c4e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099760180,
    },
    "e-988": {
      id: "e-988",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-989" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634253addf90c|930b796f-2f46-c8b0-d2b2-538f797d389b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634253addf90c|930b796f-2f46-c8b0-d2b2-538f797d389b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099800691,
    },
    "e-990": {
      id: "e-990",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-991" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6341f75ddf916|87b354a3-389f-03c0-c502-00dce2540fb2",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6341f75ddf916|87b354a3-389f-03c0-c502-00dce2540fb2",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099816668,
    },
    "e-992": {
      id: "e-992",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-993" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634c85eddf90b|b7d36a62-cf87-8aed-f42d-2af15af70e79",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634c85eddf90b|b7d36a62-cf87-8aed-f42d-2af15af70e79",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099836848,
    },
    "e-994": {
      id: "e-994",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-995" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6347622ddf911|3087578a-0c7f-f796-9b84-45b835d3d158",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6347622ddf911|3087578a-0c7f-f796-9b84-45b835d3d158",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099881416,
    },
    "e-996": {
      id: "e-996",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-997" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634349fddf914|92d8c852-8d4d-c555-da7c-b65d80dfc88b",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634349fddf914|92d8c852-8d4d-c555-da7c-b65d80dfc88b",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099898196,
    },
    "e-998": {
      id: "e-998",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-999" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6346194ddf90d|08ccfe9e-008c-7652-1384-d77b7e000511",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6346194ddf90d|08ccfe9e-008c-7652-1384-d77b7e000511",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099918783,
    },
    "e-1000": {
      id: "e-1000",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-1001" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe634ee89ddf913|d4eeeddc-41cb-b7ef-e010-9c6202ee92a0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ee89ddf913|d4eeeddc-41cb-b7ef-e010-9c6202ee92a0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595099933992,
    },
    "e-1002": {
      id: "e-1002",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-15", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "small"],
      target: {
        selector: ".animated-image-container",
        originalId:
          "5f1efa24abe6344b5addf8c3|d4ba87b5-2e94-02be-96e3-ee60bbed0347",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".animated-image-container",
          originalId:
            "5f1efa24abe6344b5addf8c3|d4ba87b5-2e94-02be-96e3-ee60bbed0347",
          appliesTo: "CLASS",
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-15-p",
          smoothing: 50,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1595244949708,
    },
    "e-1003": {
      id: "e-1003",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-1004" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6344b5addf8c3|f7ba1225-9311-afd7-c536-d6546d13d91e",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6344b5addf8c3|f7ba1225-9311-afd7-c536-d6546d13d91e",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595244962691,
    },
    "e-1008": {
      id: "e-1008",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-1009" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".animated-image._1",
        originalId:
          "5f1efa24abe6344b5addf8c3|f7ba1225-9311-afd7-c536-d6546d13d927",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".animated-image._1",
          originalId:
            "5f1efa24abe6344b5addf8c3|f7ba1225-9311-afd7-c536-d6546d13d927",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595245115286,
    },
    "e-1010": {
      id: "e-1010",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "SLIDE_EFFECT",
        instant: false,
        config: { actionListId: "slideInBottom", autoStopEventId: "e-1011" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: ".animated-image._2",
        originalId:
          "5f1efa24abe6344b5addf8c3|f7ba1225-9311-afd7-c536-d6546d13d926",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: ".animated-image._2",
          originalId:
            "5f1efa24abe6344b5addf8c3|f7ba1225-9311-afd7-c536-d6546d13d926",
          appliesTo: "CLASS",
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 200,
        direction: "BOTTOM",
        effectIn: true,
      },
      createdOn: 1595245139987,
    },
    "e-1012": {
      id: "e-1012",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-53", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        selector: "._3d-element",
        originalId: "72e9b945-1cd7-ec90-a216-c227118483d8",
        appliesTo: "CLASS",
      },
      targets: [
        {
          selector: "._3d-element",
          originalId: "72e9b945-1cd7-ec90-a216-c227118483d8",
          appliesTo: "CLASS",
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-53-p",
          smoothing: 35,
          startsEntering: true,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1595252301718,
    },
    "e-1013": {
      id: "e-1013",
      animationType: "custom",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-51",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-1014",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|e526a472-625e-0f4e-cf49-e7ce75d56ba0",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|e526a472-625e-0f4e-cf49-e7ce75d56ba0",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: true,
        playInReverse: false,
        scrollOffsetValue: 0,
        scrollOffsetUnit: "%",
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1595258907444,
    },
    "e-1017": {
      id: "e-1017",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-43", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe634ff7dddf89c|ba54d4da-a507-09f5-1382-3b560eaa8e3c",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe634ff7dddf89c|ba54d4da-a507-09f5-1382-3b560eaa8e3c",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-43-p",
          smoothing: 75,
          startsEntering: false,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1595347935859,
    },
    "e-1018": {
      id: "e-1018",
      animationType: "custom",
      eventTypeId: "SCROLLING_IN_VIEW",
      action: {
        id: "",
        actionTypeId: "GENERAL_CONTINUOUS_ACTION",
        config: { actionListId: "a-43", affectedElements: {}, duration: 0 },
      },
      mediaQueries: ["main"],
      target: {
        id: "5f1efa24abe6342b5dddf8b0|2a561894-f957-4b23-f145-22c27a16cadd",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6342b5dddf8b0|2a561894-f957-4b23-f145-22c27a16cadd",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: [
        {
          continuousParameterGroupId: "a-43-p",
          smoothing: 75,
          startsEntering: false,
          addStartOffset: false,
          addOffsetValue: 50,
          startsExiting: false,
          addEndOffset: false,
          endOffsetValue: 50,
        },
      ],
      createdOn: 1595499953454,
    },
    "e-1019": {
      id: "e-1019",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-1020" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "3e156fb6-4f34-fe8f-49ef-8453f5948284",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "3e156fb6-4f34-fe8f-49ef-8453f5948284",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 400,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595613061584,
    },
    "e-1023": {
      id: "e-1023",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-1024" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "3525af71-c677-faec-afb7-61883b9028ae",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "3525af71-c677-faec-afb7-61883b9028ae",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 600,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595673304813,
    },
    "e-1025": {
      id: "e-1025",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-1026" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "3525af71-c677-faec-afb7-61883b9028b7",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "3525af71-c677-faec-afb7-61883b9028b7",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 800,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595673358717,
    },
    "e-1027": {
      id: "e-1027",
      animationType: "preset",
      eventTypeId: "SCROLL_INTO_VIEW",
      action: {
        id: "",
        actionTypeId: "FADE_EFFECT",
        instant: false,
        config: { actionListId: "fadeIn", autoStopEventId: "e-1028" },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe6348265ddf8f1|b7eedea7-0d19-ee2e-cc8a-4e63d62373b8",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe6348265ddf8f1|b7eedea7-0d19-ee2e-cc8a-4e63d62373b8",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: 40,
        scrollOffsetUnit: "%",
        delay: 0,
        direction: null,
        effectIn: true,
      },
      createdOn: 1595674221451,
    },
    "e-1029": {
      id: "e-1029",
      animationType: "custom",
      eventTypeId: "MOUSE_CLICK",
      action: {
        id: "",
        actionTypeId: "GENERAL_START_ACTION",
        config: {
          delay: 0,
          easing: "",
          duration: 0,
          actionListId: "a-54",
          affectedElements: {},
          playInReverse: false,
          autoStopEventId: "e-1030",
        },
      },
      mediaQueries: ["main", "medium", "small", "tiny"],
      target: {
        id: "5f1efa24abe63479fbddf84f|2659e8ba-2800-5d2d-6321-7f9d05d642e4",
        appliesTo: "ELEMENT",
        styleBlockIds: [],
      },
      targets: [
        {
          id: "5f1efa24abe63479fbddf84f|2659e8ba-2800-5d2d-6321-7f9d05d642e4",
          appliesTo: "ELEMENT",
          styleBlockIds: [],
        },
      ],
      config: {
        loop: false,
        playInReverse: false,
        scrollOffsetValue: null,
        scrollOffsetUnit: null,
        delay: null,
        direction: null,
        effectIn: null,
      },
      createdOn: 1595687290773,
    },
  },
  actionLists: {
    "a-3": {
      id: "a-3",
      title: "open mobile menu",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-3-n-3",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".burger-icon",
                  selectorGuids: ["15ec93d3-2580-ba30-5213-bce84ac06603"],
                },
                value: "none",
              },
            },
            {
              id: "a-3-n-2",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".close-button",
                  selectorGuids: ["15ec93d3-2580-ba30-5213-bce84ac06604"],
                },
                value: "block",
              },
            },
            {
              id: "a-3-n-4",
              actionTypeId: "STYLE_TEXT_COLOR",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  selector: ".nav-link",
                  selectorGuids: ["5f98e719-97c1-cc73-c664-b5baedf22b87"],
                },
                globalSwatchId: "ecb01f2f",
                rValue: 148,
                bValue: 148,
                gValue: 148,
                aValue: 1,
              },
            },
            {
              id: "a-3-n-5",
              actionTypeId: "STYLE_BACKGROUND_COLOR",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "PARENT",
                  selector: ".navbar",
                  selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                },
                globalSwatchId: "",
                rValue: 255,
                bValue: 255,
                gValue: 255,
                aValue: 1,
              },
            },
            {
              id: "a-3-n-6",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  selector: ".logo.white",
                  selectorGuids: [
                    "3df1cdd8-289c-8a28-a593-422aad05aec4",
                    "dbe7e6cc-d679-595c-828e-9b51b8d956cc",
                  ],
                },
                value: 0,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1568815244301,
    },
    "a-4": {
      id: "a-4",
      title: "close-kit-nav-close",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-4-n-4",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".close-button",
                  selectorGuids: ["15ec93d3-2580-ba30-5213-bce84ac06604"],
                },
                value: "none",
              },
            },
            {
              id: "a-4-n-5",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".burger-icon",
                  selectorGuids: ["15ec93d3-2580-ba30-5213-bce84ac06603"],
                },
                value: "block",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1568816186123,
    },
    "a-13": {
      id: "a-13",
      title: "Show navbar",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-13-n",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "outSine",
                duration: 200,
                target: {
                  selector: ".navbar",
                  selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                },
                yValue: 0,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1593609235180,
    },
    "a-14": {
      id: "a-14",
      title: "hide navbar",
      actionItemGroups: [],
      useFirstGroupAsInitialState: false,
      createdOn: 1593609261902,
    },
    "a-15": {
      id: "a-15",
      title: "Parallax effect",
      continuousParameterGroups: [
        {
          id: "a-15-p",
          type: "SCROLL_PROGRESS",
          parameterLabel: "Scroll",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-15-n",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: true,
                      id: "03b11ccd-1f1f-9f0a-f990-b6177d369bfb",
                    },
                    yValue: 200,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-15-n-2",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: true,
                      id: "03b11ccd-1f1f-9f0a-f990-b6177d369bfb",
                    },
                    yValue: -200,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593613779218,
    },
    "a-18": {
      id: "a-18",
      title: "Hero scolling animation",
      continuousParameterGroups: [
        {
          id: "a-18-p",
          type: "SCROLL_PROGRESS",
          parameterLabel: "Scroll",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-18-n",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: 0,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-18-n-2",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: -222,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593723291281,
    },
    "a-19": {
      id: "a-19",
      title: "hide page",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-19-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".wrapper",
                  selectorGuids: ["5d62204f-78bf-28b5-f907-c7d89f24833d"],
                },
                value: 0,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: true,
      createdOn: 1593723688790,
    },
    "a-20": {
      id: "a-20",
      title: "show page",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-20-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".wrapper",
                  selectorGuids: ["5d62204f-78bf-28b5-f907-c7d89f24833d"],
                },
                value: 1,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1593723716771,
    },
    "a-22": {
      id: "a-22",
      title: "make navbar transparent",
      continuousParameterGroups: [
        {
          id: "a-22-p",
          type: "SCROLL_PROGRESS",
          parameterLabel: "Scroll",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-22-n",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "outSine",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 0,
                    bValue: 0,
                    gValue: 0,
                    aValue: 0,
                  },
                },
                {
                  id: "a-22-n-5",
                  actionTypeId: "STYLE_TEXT_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".nav-link",
                      selectorGuids: ["5f98e719-97c1-cc73-c664-b5baedf22b87"],
                    },
                    globalSwatchId: "",
                    rValue: 255,
                    bValue: 255,
                    gValue: 255,
                    aValue: 1,
                  },
                },
                {
                  id: "a-22-n-8",
                  actionTypeId: "STYLE_OPACITY",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: { id: "5b0d79ab-491a-54b8-c06e-42151ebab595" },
                    value: 1,
                    unit: "",
                  },
                },
                {
                  id: "a-22-n-11",
                  actionTypeId: "STYLE_OPACITY",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: { id: "6af427a1-f606-3256-0dce-86f7e7e6bce8" },
                    value: 1,
                    unit: "",
                  },
                },
              ],
            },
            {
              keyframe: 8,
              actionItems: [
                {
                  id: "a-22-n-4",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 0,
                    bValue: 0,
                    gValue: 0,
                    aValue: 0,
                  },
                },
                {
                  id: "a-22-n-6",
                  actionTypeId: "STYLE_TEXT_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".nav-link",
                      selectorGuids: ["5f98e719-97c1-cc73-c664-b5baedf22b87"],
                    },
                    globalSwatchId: "",
                    rValue: 255,
                    bValue: 255,
                    gValue: 255,
                    aValue: 1,
                  },
                },
                {
                  id: "a-22-n-10",
                  actionTypeId: "STYLE_OPACITY",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: { id: "5b0d79ab-491a-54b8-c06e-42151ebab595" },
                    value: 1,
                    unit: "",
                  },
                },
                {
                  id: "a-22-n-12",
                  actionTypeId: "STYLE_OPACITY",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: { id: "6af427a1-f606-3256-0dce-86f7e7e6bce8" },
                    value: 1,
                    unit: "",
                  },
                },
              ],
            },
            {
              keyframe: 10,
              actionItems: [
                {
                  id: "a-22-n-3",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 255,
                    bValue: 255,
                    gValue: 255,
                    aValue: 1,
                  },
                },
                {
                  id: "a-22-n-7",
                  actionTypeId: "STYLE_TEXT_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".nav-link",
                      selectorGuids: ["5f98e719-97c1-cc73-c664-b5baedf22b87"],
                    },
                    globalSwatchId: "672a5062",
                    rValue: 5,
                    bValue: 35,
                    gValue: 5,
                    aValue: 1,
                  },
                },
                {
                  id: "a-22-n-9",
                  actionTypeId: "STYLE_OPACITY",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: { id: "5b0d79ab-491a-54b8-c06e-42151ebab595" },
                    value: 0,
                    unit: "",
                  },
                },
                {
                  id: "a-22-n-13",
                  actionTypeId: "STYLE_OPACITY",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: { id: "6af427a1-f606-3256-0dce-86f7e7e6bce8" },
                    value: 0,
                    unit: "",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-22-n-2",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 255,
                    bValue: 255,
                    gValue: 255,
                    aValue: 1,
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593723996585,
    },
    "a-33": {
      id: "a-33",
      title: "New Mouse Animation 2",
      continuousParameterGroups: [
        {
          id: "a-33-p",
          type: "MOUSE_X",
          parameterLabel: "Mouse X",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-33-n",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    xValue: 20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-33-n-2",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    xValue: -20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
        {
          id: "a-33-p-2",
          type: "MOUSE_Y",
          parameterLabel: "Mouse Y",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-33-n-3",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: 20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-33-n-4",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: -20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593878966046,
    },
    "a-25": {
      id: "a-25",
      title: "Show Superior plan",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-25-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".pricing-plan-diagram",
                  selectorGuids: ["85c90256-5703-e6c7-aa07-839c2b92d24f"],
                },
                value: 0,
                unit: "",
              },
            },
            {
              id: "a-25-n-2",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".pricing-plan-diagram.superior",
                  selectorGuids: [
                    "85c90256-5703-e6c7-aa07-839c2b92d24f",
                    "776ac0f6-1b72-2593-a1d4-19eec8c91ff7",
                  ],
                },
                value: 1,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1593946825124,
    },
    "a-26": {
      id: "a-26",
      title: "Show Genuius plan",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-26-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".pricing-plan-diagram",
                  selectorGuids: ["85c90256-5703-e6c7-aa07-839c2b92d24f"],
                },
                value: 0,
                unit: "",
              },
            },
            {
              id: "a-26-n-2",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".pricing-plan-diagram.genius",
                  selectorGuids: [
                    "85c90256-5703-e6c7-aa07-839c2b92d24f",
                    "dc81b5b1-a3f1-0ab4-3dfc-06627c7efd96",
                  ],
                },
                value: 1,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1593946825124,
    },
    "a-27": {
      id: "a-27",
      title: "Show Ultra plan",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-27-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".pricing-plan-diagram",
                  selectorGuids: ["85c90256-5703-e6c7-aa07-839c2b92d24f"],
                },
                value: 0,
                unit: "",
              },
            },
            {
              id: "a-27-n-2",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 500,
                target: {
                  selector: ".pricing-plan-diagram.ultra",
                  selectorGuids: [
                    "85c90256-5703-e6c7-aa07-839c2b92d24f",
                    "84696f80-2bf4-452c-3efc-377a3a4fe6f9",
                  ],
                },
                value: 1,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1593948324983,
    },
    "a-52": {
      id: "a-52",
      title: "make navbar transparent - why commerce ai",
      continuousParameterGroups: [
        {
          id: "a-52-p",
          type: "SCROLL_PROGRESS",
          parameterLabel: "Scroll",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-52-n",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "outSine",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 0,
                    bValue: 0,
                    gValue: 0,
                    aValue: 0,
                  },
                },
              ],
            },
            {
              keyframe: 8,
              actionItems: [
                {
                  id: "a-52-n-5",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 0,
                    bValue: 0,
                    gValue: 0,
                    aValue: 0,
                  },
                },
              ],
            },
            {
              keyframe: 10,
              actionItems: [
                {
                  id: "a-52-n-9",
                  actionTypeId: "STYLE_BACKGROUND_COLOR",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".navbar",
                      selectorGuids: ["5af0c484-4a20-1cd0-994e-6e8adf2c701d"],
                    },
                    globalSwatchId: "",
                    rValue: 255,
                    bValue: 255,
                    gValue: 255,
                    aValue: 1,
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593723996585,
    },
    "a-34": {
      id: "a-34",
      title: "Marquee",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-34-n",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "",
                duration: 30000,
                target: {
                  selector: ".marquee",
                  selectorGuids: ["6876158a-5755-5063-445b-56f45b238fc0"],
                },
                xValue: -50,
                xUnit: "%",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
        {
          actionItems: [
            {
              id: "a-34-n-2",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  selector: ".marquee",
                  selectorGuids: ["6876158a-5755-5063-445b-56f45b238fc0"],
                },
                xValue: 0,
                xUnit: "%",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594301500636,
    },
    "a-35": {
      id: "a-35",
      title: "related solutions rollover",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-35-n",
              actionTypeId: "TRANSFORM_SCALE",
              config: {
                delay: 0,
                easing: "",
                duration: 200,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".related-solutions-thumbnail",
                  selectorGuids: ["8718a6b1-c071-3b10-ae2e-2a678e6a6106"],
                },
                xValue: 1.05,
                yValue: 1.05,
                locked: true,
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594587926836,
    },
    "a-36": {
      id: "a-36",
      title: "related solution rollover out",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-36-n",
              actionTypeId: "TRANSFORM_SCALE",
              config: {
                delay: 0,
                easing: "",
                duration: 200,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".related-solutions-thumbnail",
                  selectorGuids: ["8718a6b1-c071-3b10-ae2e-2a678e6a6106"],
                },
                xValue: 1,
                yValue: 1,
                locked: true,
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594588015989,
    },
    "a-37": {
      id: "a-37",
      title: "button chevron animation",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-37-n",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "easeInOut",
                duration: 100,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".chevron",
                  selectorGuids: ["72e495b4-b4bb-3dce-42a1-a6684f9edf0f"],
                },
                xValue: 8,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594658287916,
    },
    "a-38": {
      id: "a-38",
      title: "button chevron rollover out",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-38-n",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "easeInOut",
                duration: 100,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".chevron",
                  selectorGuids: ["72e495b4-b4bb-3dce-42a1-a6684f9edf0f"],
                },
                xValue: 0,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594658370768,
    },
    "a-43": {
      id: "a-43",
      title: "Hero scolling animation 2",
      continuousParameterGroups: [
        {
          id: "a-43-p",
          type: "SCROLL_PROGRESS",
          parameterLabel: "Scroll",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-43-n",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: 0,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-43-n-2",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: -222,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593723291281,
    },
    "a-42": {
      id: "a-42",
      title: "button chevron rollover out 2",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-42-n",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "easeInOut",
                duration: 100,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".chevron",
                  selectorGuids: ["72e495b4-b4bb-3dce-42a1-a6684f9edf0f"],
                },
                xValue: 0,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594658370768,
    },
    "a-28": {
      id: "a-28",
      title: "show bio",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-28-n-2",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".bio",
                  selectorGuids: ["b5a926ad-4e58-d39c-fc43-9aa4cadf1390"],
                },
                value: "flex",
              },
            },
            {
              id: "a-28-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 200,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".bio",
                  selectorGuids: ["b5a926ad-4e58-d39c-fc43-9aa4cadf1390"],
                },
                value: 1,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594027991426,
    },
    "a-29": {
      id: "a-29",
      title: "hide bio",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-29-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "",
                duration: 200,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".bio",
                  selectorGuids: ["b5a926ad-4e58-d39c-fc43-9aa4cadf1390"],
                },
                value: 0,
                unit: "",
              },
            },
            {
              id: "a-29-n-2",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "CHILDREN",
                  selector: ".bio",
                  selectorGuids: ["b5a926ad-4e58-d39c-fc43-9aa4cadf1390"],
                },
                value: "none",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594028484987,
    },
    "a-45": {
      id: "a-45",
      title: "New Mouse Animation 4",
      continuousParameterGroups: [
        {
          id: "a-45-p",
          type: "MOUSE_X",
          parameterLabel: "Mouse X",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-45-n",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    xValue: 20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-45-n-2",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    xValue: -20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
        {
          id: "a-45-p-2",
          type: "MOUSE_Y",
          parameterLabel: "Mouse Y",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-45-n-3",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: 20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-45-n-4",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      useEventTarget: "CHILDREN",
                      selector: ".hero-image",
                      selectorGuids: ["31640e13-f68a-4cc3-0e24-7eebc7246ce0"],
                    },
                    yValue: -20,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593878966046,
    },
    "a-48": {
      id: "a-48",
      title: "Close announcement",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-48-n",
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "outQuart",
                duration: 200,
                target: {
                  useEventTarget: "PARENT",
                  selector: ".section.announcement",
                  selectorGuids: [
                    "5d62204f-78bf-28b5-f907-c7d89f248335",
                    "5aba8b55-6619-d770-a4ab-171f83af6286",
                  ],
                },
                yValue: 33,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1594989232757,
    },
    "a-53": {
      id: "a-53",
      title: "Parallax effect / 3d elements + rotation",
      continuousParameterGroups: [
        {
          id: "a-53-p",
          type: "SCROLL_PROGRESS",
          parameterLabel: "Scroll",
          continuousActionGroups: [
            {
              keyframe: 0,
              actionItems: [
                {
                  id: "a-53-n",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "easeInOut",
                    duration: 500,
                    target: {
                      selector: "._3d-element.customers-3",
                      selectorGuids: [
                        "1aad394a-c8b6-7c13-841b-8297411ae8d9",
                        "3123eb2d-4866-53b4-9fdc-641eb1c57f4d",
                      ],
                    },
                    yValue: 25,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
                {
                  id: "a-53-n-3",
                  actionTypeId: "TRANSFORM_ROTATE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: "._3d-element.customers-3",
                      selectorGuids: [
                        "1aad394a-c8b6-7c13-841b-8297411ae8d9",
                        "3123eb2d-4866-53b4-9fdc-641eb1c57f4d",
                      ],
                    },
                    zValue: 0,
                    xUnit: "DEG",
                    yUnit: "DEG",
                    zUnit: "DEG",
                  },
                },
              ],
            },
            {
              keyframe: 100,
              actionItems: [
                {
                  id: "a-53-n-2",
                  actionTypeId: "TRANSFORM_MOVE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: "._3d-element.customers-3",
                      selectorGuids: [
                        "1aad394a-c8b6-7c13-841b-8297411ae8d9",
                        "3123eb2d-4866-53b4-9fdc-641eb1c57f4d",
                      ],
                    },
                    yValue: -25,
                    xUnit: "PX",
                    yUnit: "PX",
                    zUnit: "PX",
                  },
                },
                {
                  id: "a-53-n-4",
                  actionTypeId: "TRANSFORM_ROTATE",
                  config: {
                    delay: 0,
                    easing: "",
                    duration: 500,
                    target: {
                      selector: "._3d-element.customers-3",
                      selectorGuids: [
                        "1aad394a-c8b6-7c13-841b-8297411ae8d9",
                        "3123eb2d-4866-53b4-9fdc-641eb1c57f4d",
                      ],
                    },
                    zValue: 45,
                    xUnit: "DEG",
                    yUnit: "DEG",
                    zUnit: "DEG",
                  },
                },
              ],
            },
          ],
        },
      ],
      createdOn: 1593613779218,
    },
    "a-51": {
      id: "a-51",
      title: "cursor blinking",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-51-n",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "easeInOut",
                duration: 500,
                target: {
                  useEventTarget: true,
                  id: "5f1efa24abe63479fbddf84f|e526a472-625e-0f4e-cf49-e7ce75d56ba0",
                },
                value: 1,
                unit: "",
              },
            },
          ],
        },
        {
          actionItems: [
            {
              id: "a-51-n-2",
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "easeInOut",
                duration: 500,
                target: {
                  useEventTarget: true,
                  id: "5f1efa24abe63479fbddf84f|e526a472-625e-0f4e-cf49-e7ce75d56ba0",
                },
                value: 0,
                unit: "",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1595258913460,
    },
    "a-54": {
      id: "a-54",
      title: "close cookies popup",
      actionItemGroups: [
        {
          actionItems: [
            {
              id: "a-54-n",
              actionTypeId: "GENERAL_DISPLAY",
              config: {
                delay: 0,
                easing: "",
                duration: 0,
                target: {
                  useEventTarget: "PARENT",
                  selector: ".popup-overlay",
                  selectorGuids: ["07f4dc19-f00a-af6f-8605-e61311286206"],
                },
                value: "none",
              },
            },
          ],
        },
      ],
      useFirstGroupAsInitialState: false,
      createdOn: 1595687293549,
    },
    fadeIn: {
      id: "fadeIn",
      useFirstGroupAsInitialState: true,
      actionItemGroups: [
        {
          actionItems: [
            {
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                duration: 0,
                target: {
                  id: "N/A",
                  appliesTo: "TRIGGER_ELEMENT",
                  useEventTarget: true,
                },
                value: 0,
              },
            },
          ],
        },
        {
          actionItems: [
            {
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "outQuart",
                duration: 1000,
                target: {
                  id: "N/A",
                  appliesTo: "TRIGGER_ELEMENT",
                  useEventTarget: true,
                },
                value: 1,
              },
            },
          ],
        },
      ],
    },
    slideInBottom: {
      id: "slideInBottom",
      useFirstGroupAsInitialState: true,
      actionItemGroups: [
        {
          actionItems: [
            {
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                duration: 0,
                target: {
                  id: "N/A",
                  appliesTo: "TRIGGER_ELEMENT",
                  useEventTarget: true,
                },
                value: 0,
              },
            },
          ],
        },
        {
          actionItems: [
            {
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                duration: 0,
                target: {
                  id: "N/A",
                  appliesTo: "TRIGGER_ELEMENT",
                  useEventTarget: true,
                },
                xValue: 0,
                yValue: 100,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
          ],
        },
        {
          actionItems: [
            {
              actionTypeId: "TRANSFORM_MOVE",
              config: {
                delay: 0,
                easing: "outQuart",
                duration: 1000,
                target: {
                  id: "N/A",
                  appliesTo: "TRIGGER_ELEMENT",
                  useEventTarget: true,
                },
                xValue: 0,
                yValue: 0,
                xUnit: "PX",
                yUnit: "PX",
                zUnit: "PX",
              },
            },
            {
              actionTypeId: "STYLE_OPACITY",
              config: {
                delay: 0,
                easing: "outQuart",
                duration: 1000,
                target: {
                  id: "N/A",
                  appliesTo: "TRIGGER_ELEMENT",
                  useEventTarget: true,
                },
                value: 1,
              },
            },
          ],
        },
      ],
    },
  },
  site: {
    mediaQueries: [
      { key: "main", min: 992, max: 10000 },
      { key: "medium", min: 768, max: 991 },
      { key: "small", min: 480, max: 767 },
      { key: "tiny", min: 0, max: 479 },
    ],
  },
});
