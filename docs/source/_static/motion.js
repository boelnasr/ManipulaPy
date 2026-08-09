(function () {
  "use strict";

  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reducedMotion.matches) return;

  function reveal(element) {
    var animation = element.animate(
      [
        { opacity: 0, transform: "translateY(18px)" },
        { opacity: 1, transform: "translateY(0)" }
      ],
      {
        duration: 520,
        easing: "cubic-bezier(0.16, 1, 0.3, 1)",
        fill: "both"
      }
    );
    animation.addEventListener("finish", function () {
      element.style.opacity = "";
      element.style.transform = "";
    }, { once: true });
  }

  function start() {
    var targets = Array.prototype.slice.call(
      document.querySelectorAll("[data-reveal]")
    );
    if (!targets.length) return;

    if (!("IntersectionObserver" in window)) {
      targets.forEach(reveal);
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        reveal(entry.target);
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.08 });

    targets.forEach(function (target) {
      if (target.getBoundingClientRect().top > window.innerHeight) {
        observer.observe(target);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
