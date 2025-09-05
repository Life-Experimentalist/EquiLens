# EquiLens GitHub Pages

This directory contains the GitHub Pages website for the EquiLens project.

## 🌐 Live Site

Visit the live site at: [https://life-experimentalists.github.io/EquiLens/](https://life-experimentalists.github.io/EquiLens/)

## 📁 Structure

```
public/
├── index.html          # Main landing page
├── assets/            # Static assets (images, icons, etc.)
├── css/               # Additional stylesheets (if needed)
├── js/                # Additional JavaScript files (if needed)
└── README.md          # This file
```

## ✨ Features

- **Responsive Design**: Optimized for all devices and screen sizes
- **Dark/Light Theme**: Toggle between themes with persistent storage
- **Dark Reader Compatible**: Native dark mode that works seamlessly with Dark Reader extension
- **Interactive Animations**: Smooth scrolling, fade-ins, and counters
- **Live Demo**: Terminal simulation showing EquiLens in action
- **Copy-to-Clipboard**: Easy code copying from examples
- **SEO Optimized**: Meta tags, Open Graph, and Twitter cards
- **Accessibility**: ARIA labels and keyboard navigation support

## 🚀 Development

To work on the site locally:

1. Clone the repository
2. Navigate to the `public` directory
3. Serve the files using any static file server:
   ```bash
   # Using Python
   python -m http.server 8000

   # Using Node.js
   npx serve .

   # Using PHP
   php -S localhost:8000
   ```
4. Open `http://localhost:8000` in your browser

## 📱 Mobile Responsiveness

The site is fully responsive and tested on:
- Desktop (1920px+)
- Laptop (1024px - 1919px)
- Tablet (768px - 1023px)
- Mobile (320px - 767px)

## 🎨 Design System

### Colors
- Primary: `#6366f1` (Indigo)
- Secondary: `#0f172a` (Slate Dark)
- Accent: `#f59e0b` (Amber)

### Typography
- Headings: `Inter` (Google Fonts)
- Code: `JetBrains Mono` (Google Fonts)

### Components
- Cards with subtle shadows and hover effects
- Gradient buttons and icons
- Progress bars and loading animations
- Toast notifications for user feedback

## 🔧 Customization

To customize the site:

1. **Colors**: Modify CSS custom properties in `:root`
2. **Content**: Update HTML content sections
3. **Features**: Add new feature cards in the features grid
4. **Statistics**: Update data-target attributes for counters
5. **Release Links**: Update GitHub release URLs and DOI links as needed

## 📊 Performance

The site is optimized for performance:
- Minimal external dependencies
- Efficient CSS and JavaScript
- Optimized images and assets
- Fast loading times

## 🌟 Browser Support

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📝 License

This website is part of the EquiLens project and is licensed under the Apache 2.0 License.
