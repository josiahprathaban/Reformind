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
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-dark-500 text-gray-200`} suppressHydrationWarning={true}>
        <main className="flex flex-col min-h-screen relative overflow-hidden">
          {/* Full page background glow effect */}
          <div aria-hidden="true" className="absolute inset-0 -z-10 transform-gpu overflow-hidden blur-3xl">
            <div 
              style={{
                clipPath: 'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)'
              }} 
              className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#d99d47] to-[#ffde66] opacity-25 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
            ></div>
            
            {/* Additional right-side gradient */}
            <div 
              style={{
                clipPath: 'polygon(64.1% 54.1%, 90% 71.6%, 87.5% 36.9%, 75.5% 10.1%, 70.7% 12%, 62.5% 42.5%, 50.2% 72.4%, 42.4% 78.1%, 37.5% 68.3%, 35.2% 44.5%, 17.5% 86.7%, 10.1% 74.9%, 27.9% 90%, 37.6% 66.8%, 66.1% 87.7%, 64.1% 54.1%)'
              }} 
              className="relative left-[calc(50%+15rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/3 rotate-[150deg] bg-gradient-to-r from-[#be8020] via-[#ffb74d] to-[#ebc893] opacity-20 sm:left-[calc(50%+20rem)] sm:w-[72.1875rem]"
            ></div>
            
            {/* Center gradient */}
            <div 
              style={{
                clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)'
              }} 
              className="relative left-[calc(50%)] aspect-square w-[20rem] rotate-[45deg] bg-gradient-conic from-[#d99d47] via-[#ffde66] to-[#be8020] opacity-10 sm:w-[40rem]"
            ></div>
          </div>
          
          {children}
          
          {/* Bottom background glow effect */}
          <div aria-hidden="true" className="absolute inset-x-0 bottom-0 -z-10 transform-gpu overflow-hidden blur-3xl">
            <div 
              style={{
                clipPath: 'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)'
              }} 
              className="relative left-[calc(50%+11rem)] aspect-[1155/678] w-[36.125rem] translate-x-1/2 rotate-[210deg] bg-gradient-to-tr from-[#be8020] to-[#ebc893] opacity-25 sm:left-[calc(50%+30rem)] sm:w-[72.1875rem]"
            ></div>
            
            {/* Additional bottom-left gradient */}
            <div 
              style={{
                clipPath: 'polygon(44.1% 74.1%, 61.6% 100%, 26.9% 97.5%, 0.1% 85.5%, 2% 80.7%, 32.5% 72.5%, 62.4% 60.2%, 68.1% 52.4%, 58.3% 47.5%, 34.5% 45.2%, 76.7% 27.5%, 64.9% 0.1%, 100% 17.9%, 76.8% 27.6%, 97.7% 76.1%, 44.1% 74.1%)'
              }} 
              className="relative left-[calc(30%)] bottom-[5rem] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[120deg] bg-gradient-to-br from-[#ffb74d] via-[#d99d47] to-[#7c5519] opacity-20 sm:w-[60rem]"
            ></div>
          </div>
        </main>
      </body>
    </html>
  )
}
