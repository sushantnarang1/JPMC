/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0a101b',
        panel: '#111a28',
        line: '#243247',
        accent: '#39c8c3',
        blue: '#6d9eff',
      },
    },
  },
  plugins: [],
}
