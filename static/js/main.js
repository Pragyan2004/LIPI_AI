/**
 * Akshara AI Unified Micro-interactions v2.0
 * Optimized for 2025 Neural Interfaces
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Smooth Motion Engine (GSAP + AOS)
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 1000,
            easing: 'cubic-bezier(0.2, 1, 0.3, 1)',
            once: true,
            offset: 50
        });
    }

    // 2. Global Magnetic Cursor Effect
    const cursor = document.getElementById('cursor');
    if (cursor && window.innerWidth > 1024) {
        document.addEventListener('mousemove', (e) => {
            gsap.to(cursor, {
                x: e.clientX,
                y: e.clientY,
                duration: 0.2,
                ease: "power2.out"
            });
        });

        // Interactive states for bento items and buttons
        const interactables = document.querySelectorAll('.bento-item, .btn-premium, .nav-link, .dash-card');
        interactables.forEach(el => {
            el.addEventListener('mouseenter', () => {
                gsap.to(cursor, {
                    scale: 4,
                    backgroundColor: "rgba(99, 102, 241, 0.1)",
                    border: "1px solid rgba(99, 102, 241, 0.5)",
                    duration: 0.3
                });
            });
            el.addEventListener('mouseleave', () => {
                gsap.to(cursor, {
                    scale: 1,
                    backgroundColor: "#6366f1",
                    border: "none",
                    duration: 0.3
                });
            });
        });
    }

    // 3. Floating Navigation Transition
    const nav = document.getElementById('navbar');
    if (nav) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 30) {
                nav.classList.add('scrolled');
                gsap.to(nav, { y: -10, duration: 0.4 });
            } else {
                nav.classList.remove('scrolled');
                gsap.to(nav, { y: 0, duration: 0.4 });
            }
        });
    }

    // 4. Input Persistence & Micro-focus
    const inputs = document.querySelectorAll('.form-input-modern');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            gsap.to(input, { borderColor: "#6366f1", boxShadow: "0 0 20px rgba(99, 102, 241, 0.2)", duration: 0.3 });
        });
        input.addEventListener('blur', () => {
            gsap.to(input, { borderColor: "rgba(255,255,255,0.08)", boxShadow: "none", duration: 0.3 });
        });
    });

    // 5. Global Counter Utility (High Performance)
    window.animateValue = function (id, start, end, duration) {
        let obj = document.getElementById(id);
        if (!obj) return;
        let range = end - start;
        let minTimer = 50;
        let stepTime = Math.abs(Math.floor(duration / range));
        stepTime = Math.max(stepTime, minTimer);
        let startTime = new Date().getTime();
        let endTime = startTime + duration;
        let timer;

        function run() {
            let now = new Date().getTime();
            let remaining = Math.max((endTime - now) / duration, 0);
            let value = Math.round(end - (remaining * range));
            obj.innerHTML = value.toLocaleString();
            if (value == end) {
                clearInterval(timer);
            }
        }

        timer = setInterval(run, stepTime);
        run();
    };
});
