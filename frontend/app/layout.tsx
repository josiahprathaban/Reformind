import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Reformind - Your Reformed AI Pastor',
  description: 'An AI assistant trained on Scripture and historic Reformed confessions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className} suppressHydrationWarning={true}>
        <main className="flex flex-col min-h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}
