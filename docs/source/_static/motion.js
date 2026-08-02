/* ============================================================================
   ManipulaPy documentation — motion layer
   ----------------------------------------------------------------------------
   Powered by the vendored anime.js bundle (_static/anime.umd.min.js, MIT).

   Two rules govern everything here:

     1. Motion is a reading aid, not decoration. Elements rise 10px and fade
        in once, on first approach. Nothing loops, bounces, pulses, or moves
        after the reader has arrived — a docs page that keeps moving is a page
        you cannot read.

     2. The hidden start state is applied from JavaScript, never from CSS.
        If the bundle is blocked, fails to load, or the reader has
        prefers-reduced-motion set, this file returns early and the page is
        simply static. It can never leave content invisible.

   Only content below the fold is ever animated. Sphinx emits these scripts
   synchronously in <head>, so the DOM is not ready until DOMContentLoaded —
   by which point the first screenful has already painted. Animating it then
   would mean hiding visible content and fading it back in, a flicker on every
   page load. Anything on screen at load is therefore left alone, and motion
   is purely a scroll-reading aid.
   ========================================================================= */

(function () {
    "use strict";

    var api = window.anime;
    if (!api || typeof api.animate !== "function") return;
    if (!api.utils || typeof api.utils.set !== "function") return;

    // Honour the OS-level setting. Checked before any style is written, so a
    // reduced-motion reader gets an untouched document.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    var animate = api.animate;
    var stagger = typeof api.stagger === "function" ? api.stagger : null;
    var set = api.utils.set;

    // Decelerating curve, no overshoot. Built with cubicBezier rather than a
    // named ease so the exact curve is pinned here and not in the bundle.
    var EASE = api.cubicBezier(0.22, 1.0, 0.36, 1.0);
    var RISE = 10;    // px — reads as motion, not as travel
    var DUR = 460;    // ms
    var STEP = 55;    // ms between staggered siblings

    function hide(targets) {
        set(targets, { opacity: 0, translateY: RISE });
    }

    function reveal(targets, step) {
        animate(targets, {
            opacity: 1,
            translateY: 0,
            duration: DUR,
            delay: step && stagger ? stagger(step) : 0,
            ease: EASE
        });
    }

    function query(root, selector) {
        return Array.prototype.slice.call(root.querySelectorAll(selector));
    }

    function ready(fn) {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", fn, { once: true });
        } else {
            fn();
        }
    }

    ready(start);

    function start() {
    var main = document.querySelector("main") || document.body;
    if (!main) return;

    /* --- Scroll-revealed blocks -----------------------------------------
       Triggered with IntersectionObserver rather than the bundle's scroll
       observer: the trigger contract is part of the platform and behaves
       identically across themes, so only the motion itself depends on
       anime.js. Each group is revealed once and then unobserved. */
    var groups = [];

    // Containers whose children stagger in sequence.
    [
        { container: ".mp-grid", items: ".mp-card" },
        { container: ".mp-flow", items: ".mp-flow__stage" }
    ].forEach(function (spec) {
        query(main, spec.container).forEach(function (node) {
            var items = query(node, spec.items);
            if (items.length) groups.push({ node: node, items: items, step: STEP });
        });
    });

    // Standalone blocks — revealed whole, so tables and lists do not shimmer
    // row by row.
    [".mp-rail", ".mp-matrix"].forEach(function (sel) {
        query(main, sel).forEach(function (node) {
            groups.push({ node: node, items: [node], step: 0 });
        });
    });

    if (!groups.length) return;

    // Anything already on screen at load is shown without animation — a
    // reader should never watch content they are already looking at fade in.
    var pending = groups.filter(function (group) {
        return group.node.getBoundingClientRect().top > window.innerHeight;
    });
    if (!pending.length) return;

    pending.forEach(function (group) {
        hide(group.items);
    });

    if (typeof window.IntersectionObserver !== "function") {
        // No observer support: reveal everything at once rather than leaving
        // the hidden start state in place.
        pending.forEach(function (group) {
            reveal(group.items, 0);
        });
        return;
    }

    var observer = new IntersectionObserver(
        function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                observer.unobserve(entry.target);
                for (var i = 0; i < pending.length; i++) {
                    if (pending[i].node === entry.target) {
                        reveal(pending[i].items, pending[i].step);
                        break;
                    }
                }
            });
        },
        { rootMargin: "0px 0px -8% 0px", threshold: 0 }
    );

    pending.forEach(function (group) {
        observer.observe(group.node);
    });
    }
})();
