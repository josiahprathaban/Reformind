/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#fdf8f0',
          100: '#f9efe0',
          200: '#f3dcb9',
          300: '#ebc893',
          400: '#e3b46d',
          500: '#d99d47',
          600: '#be8020',
          700: '#9d6a1c',
          800: '#7c5519',
          900: '#5e4116',
        },
        dark: {
          100: '#3e3833',
          200: '#35302b',
          300: '#2c2823',
          400: '#25221e',
          500: '#1f1c19',
          600: '#191714',
          700: '#13110f',
          800: '#0c0a09',
          900: '#060504',
        },
        accent: {
          100: '#fff8cc',
          200: '#fff1b3',
          300: '#ffeb99',
          400: '#ffe480',
          500: '#ffde66',
          600: '#ffc926',
          700: '#e6b520',
          800: '#cca01b',
          900: '#b38c17',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
      boxShadow: {
        'glass': '0 4px 30px rgba(0, 0, 0, 0.1)',
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.gray.300'),
            a: {
              color: theme('colors.primary.400'),
              '&:hover': {
                color: theme('colors.primary.300'),
              },
            },
          },
        },
        invert: {
          css: {
            color: theme('colors.gray.300'),
            a: {
              color: theme('colors.primary.400'),
              '&:hover': {
                color: theme('colors.primary.300'),
              },
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
