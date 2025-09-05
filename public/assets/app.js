// Additional JavaScript for EquiLens GitHub Pages

class EquiLensApp {
	constructor() {
		this.init();
	}

	init() {
		this.setupLazyLoading();
		this.setupProgressiveEnhancement();
		this.setupKeyboardNavigation();
		this.setupFormValidation();
		this.setupAdvancedAnimations();
	}

	// Lazy loading for images and heavy content
	setupLazyLoading() {
		if ("IntersectionObserver" in window) {
			const imageObserver = new IntersectionObserver((entries) => {
				entries.forEach((entry) => {
					if (entry.isIntersecting) {
						const img = entry.target;
						img.src = img.dataset.src;
						img.classList.remove("lazy");
						imageObserver.unobserve(img);
					}
				});
			});

			document.querySelectorAll("img[data-src]").forEach((img) => {
				imageObserver.observe(img);
			});
		}
	}

	// Progressive enhancement for advanced features
	setupProgressiveEnhancement() {
		// Check for required APIs
		const hasIntersectionObserver = "IntersectionObserver" in window;
		const hasLocalStorage = typeof Storage !== "undefined";
		const hasClipboard =
			navigator.clipboard && navigator.clipboard.writeText;

		// Add classes based on support
		document.documentElement.classList.toggle(
			"has-intersection-observer",
			hasIntersectionObserver
		);
		document.documentElement.classList.toggle(
			"has-local-storage",
			hasLocalStorage
		);
		document.documentElement.classList.toggle(
			"has-clipboard",
			hasClipboard
		);

		// Fallbacks
		if (!hasClipboard) {
			document.querySelectorAll(".copy-btn").forEach((btn) => {
				btn.style.display = "none";
			});
		}
	}

	// Enhanced keyboard navigation
	setupKeyboardNavigation() {
		// Skip to content link
		const skipLink = document.createElement("a");
		skipLink.href = "#main-content";
		skipLink.textContent = "Skip to main content";
		skipLink.className = "skip-link sr-only";
		skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            background: var(--primary-color);
            color: white;
            padding: 8px;
            border-radius: 4px;
            text-decoration: none;
            z-index: 9999;
            transition: top 0.3s ease;
        `;

		skipLink.addEventListener("focus", () => {
			skipLink.style.top = "6px";
		});

		skipLink.addEventListener("blur", () => {
			skipLink.style.top = "-40px";
		});

		document.body.insertBefore(skipLink, document.body.firstChild);

		// Add main content ID
		const hero = document.querySelector(".hero");
		if (hero && !hero.id) {
			hero.id = "main-content";
		}

		// Escape key to close modals/menus
		document.addEventListener("keydown", (e) => {
			if (e.key === "Escape") {
				// Close any open modals or menus
				document
					.querySelectorAll(".modal.open, .menu.open")
					.forEach((el) => {
						el.classList.remove("open");
					});
			}
		});

		// Arrow key navigation for card grids
		this.setupGridNavigation();
	}

	setupGridNavigation() {
		const grids = document.querySelectorAll(
			".features-grid, .steps-grid, .stats-grid"
		);

		grids.forEach((grid) => {
			const cards = grid.querySelectorAll(
				".feature-card, .step-card, .stat-card"
			);
			cards.forEach((card, index) => {
				card.tabIndex = 0;

				card.addEventListener("keydown", (e) => {
					let newIndex = index;
					const columns =
						getComputedStyle(grid).gridTemplateColumns.split(
							" "
						).length;

					switch (e.key) {
						case "ArrowRight":
							newIndex = Math.min(index + 1, cards.length - 1);
							break;
						case "ArrowLeft":
							newIndex = Math.max(index - 1, 0);
							break;
						case "ArrowDown":
							newIndex = Math.min(
								index + columns,
								cards.length - 1
							);
							break;
						case "ArrowUp":
							newIndex = Math.max(index - columns, 0);
							break;
						default:
							return;
					}

					if (newIndex !== index) {
						e.preventDefault();
						cards[newIndex].focus();
					}
				});
			});
		});
	}

	// Form validation (for future contact forms)
	setupFormValidation() {
		document.querySelectorAll("form").forEach((form) => {
			form.addEventListener("submit", (e) => {
				const isValid = this.validateForm(form);
				if (!isValid) {
					e.preventDefault();
				}
			});
		});
	}

	validateForm(form) {
		let isValid = true;
		const fields = form.querySelectorAll("input, textarea, select");

		fields.forEach((field) => {
			const value = field.value.trim();
			const isRequired = field.hasAttribute("required");
			const type = field.type;

			// Remove previous error styling
			field.classList.remove("error");

			// Check if required field is empty
			if (isRequired && !value) {
				this.showFieldError(field, "This field is required");
				isValid = false;
				return;
			}

			// Validate email
			if (type === "email" && value && !this.isValidEmail(value)) {
				this.showFieldError(
					field,
					"Please enter a valid email address"
				);
				isValid = false;
				return;
			}

			// Validate URL
			if (type === "url" && value && !this.isValidUrl(value)) {
				this.showFieldError(field, "Please enter a valid URL");
				isValid = false;
				return;
			}
		});

		return isValid;
	}

	showFieldError(field, message) {
		field.classList.add("error");

		// Remove existing error message
		const existingError = field.parentNode.querySelector(".error-message");
		if (existingError) {
			existingError.remove();
		}

		// Add new error message
		const errorEl = document.createElement("div");
		errorEl.className = "error-message";
		errorEl.textContent = message;
		errorEl.style.cssText = `
            color: #ef4444;
            font-size: 0.875rem;
            margin-top: 0.25rem;
        `;

		field.parentNode.appendChild(errorEl);
	}

	isValidEmail(email) {
		const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
		return emailRegex.test(email);
	}

	isValidUrl(url) {
		try {
			new URL(url);
			return true;
		} catch {
			return false;
		}
	}

	// Advanced animations and interactions
	setupAdvancedAnimations() {
		// Parallax effect for hero section
		if (window.innerWidth > 768) {
			this.setupParallax();
		}

		// Typing animation for demo
		this.setupTypingAnimation();

		// Particle background animation
		this.setupParticleAnimation();

		// Interactive hover effects
		this.setupInteractiveEffects();
	}

	setupParallax() {
		const hero = document.querySelector(".hero");
		if (!hero) return;

		let ticking = false;

		const updateParallax = () => {
			const scrolled = window.pageYOffset;
			const parallax = hero.querySelector(".hero-content");

			if (parallax) {
				const speed = scrolled * 0.5;
				parallax.style.transform = `translateY(${speed}px)`;
			}

			ticking = false;
		};

		const requestTick = () => {
			if (!ticking) {
				requestAnimationFrame(updateParallax);
				ticking = true;
			}
		};

		window.addEventListener("scroll", requestTick);
	}

	setupTypingAnimation() {
		const typeElements = document.querySelectorAll(".typing-cursor");

		typeElements.forEach((element) => {
			const text = element.textContent;
			element.textContent = "";

			let index = 0;
			const typeInterval = setInterval(() => {
				element.textContent += text[index];
				index++;

				if (index >= text.length) {
					clearInterval(typeInterval);
					element.classList.add("typing-complete");
				}
			}, 100);
		});
	}

	setupParticleAnimation() {
		// Simple particle system for hero background
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");
		const hero = document.querySelector(".hero");

		if (!hero || window.innerWidth < 768) return;

		canvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0.1;
            z-index: 0;
        `;

		hero.appendChild(canvas);

		const particles = [];
		const particleCount = 50;

		// Resize canvas
		const resizeCanvas = () => {
			canvas.width = hero.offsetWidth;
			canvas.height = hero.offsetHeight;
		};

		resizeCanvas();
		window.addEventListener("resize", resizeCanvas);

		// Create particles
		for (let i = 0; i < particleCount; i++) {
			particles.push({
				x: Math.random() * canvas.width,
				y: Math.random() * canvas.height,
				size: Math.random() * 3 + 1,
				speedX: (Math.random() - 0.5) * 0.5,
				speedY: (Math.random() - 0.5) * 0.5,
				opacity: Math.random() * 0.5 + 0.2,
			});
		}

		// Animate particles
		const animateParticles = () => {
			ctx.clearRect(0, 0, canvas.width, canvas.height);

			particles.forEach((particle) => {
				particle.x += particle.speedX;
				particle.y += particle.speedY;

				// Wrap around edges
				if (particle.x < 0) particle.x = canvas.width;
				if (particle.x > canvas.width) particle.x = 0;
				if (particle.y < 0) particle.y = canvas.height;
				if (particle.y > canvas.height) particle.y = 0;

				// Draw particle
				ctx.beginPath();
				ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
				ctx.fillStyle = `rgba(99, 102, 241, ${particle.opacity})`;
				ctx.fill();
			});

			requestAnimationFrame(animateParticles);
		};

		animateParticles();
	}

	setupInteractiveEffects() {
		// Add magnetic effect to buttons
		document.querySelectorAll(".btn").forEach((btn) => {
			btn.addEventListener("mousemove", (e) => {
				const rect = btn.getBoundingClientRect();
				const x = e.clientX - rect.left - rect.width / 2;
				const y = e.clientY - rect.top - rect.height / 2;

				btn.style.transform = `translate(${x * 0.1}px, ${y * 0.1}px)`;
			});

			btn.addEventListener("mouseleave", () => {
				btn.style.transform = "";
			});
		});

		// Add tilt effect to cards
		document
			.querySelectorAll(".feature-card, .step-card")
			.forEach((card) => {
				card.addEventListener("mousemove", (e) => {
					const rect = card.getBoundingClientRect();
					const x = e.clientX - rect.left;
					const y = e.clientY - rect.top;

					const centerX = rect.width / 2;
					const centerY = rect.height / 2;

					const rotateX = (y - centerY) / 10;
					const rotateY = (centerX - x) / 10;

					card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
				});

				card.addEventListener("mouseleave", () => {
					card.style.transform = "";
				});
			});
	}
}

// Service Worker registration for PWA capabilities
if ("serviceWorker" in navigator) {
	window.addEventListener("load", () => {
		navigator.serviceWorker
			.register("/EquiLens/sw.js")
			.then((registration) => {
				console.log("SW registered: ", registration);
			})
			.catch((registrationError) => {
				console.log("SW registration failed: ", registrationError);
			});
	});
}

// Performance monitoring
class PerformanceMonitor {
	constructor() {
		this.metrics = {};
		this.init();
	}

	init() {
		// Measure page load time
		window.addEventListener("load", () => {
			const loadTime =
				performance.timing.loadEventEnd -
				performance.timing.navigationStart;
			this.metrics.pageLoadTime = loadTime;

			// Log to console in development
			if (window.location.hostname === "localhost") {
				console.log(`Page load time: ${loadTime}ms`);
			}
		});

		// Measure Core Web Vitals
		this.measureCoreWebVitals();
	}

	measureCoreWebVitals() {
		// Largest Contentful Paint
		new PerformanceObserver((entryList) => {
			const entries = entryList.getEntries();
			const lastEntry = entries[entries.length - 1];
			this.metrics.lcp = lastEntry.startTime;
		}).observe({ entryTypes: ["largest-contentful-paint"] });

		// First Input Delay
		new PerformanceObserver((entryList) => {
			const firstInput = entryList.getEntries()[0];
			this.metrics.fid =
				firstInput.processingStart - firstInput.startTime;
		}).observe({ entryTypes: ["first-input"], buffered: true });

		// Cumulative Layout Shift
		let clsValue = 0;
		new PerformanceObserver((entryList) => {
			for (const entry of entryList.getEntries()) {
				if (!entry.hadRecentInput) {
					clsValue += entry.value;
				}
			}
			this.metrics.cls = clsValue;
		}).observe({ entryTypes: ["layout-shift"] });
	}
}

// Initialize app when DOM is ready
if (document.readyState === "loading") {
	document.addEventListener("DOMContentLoaded", () => {
		new EquiLensApp();
		new PerformanceMonitor();
	});
} else {
	new EquiLensApp();
	new PerformanceMonitor();
}

// Export for testing
if (typeof module !== "undefined" && module.exports) {
	module.exports = { EquiLensApp, PerformanceMonitor };
}
