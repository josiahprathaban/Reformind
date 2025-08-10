'use client'

import { useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!question.trim()) {
      setError('Please enter a question')
      return
    }

    setIsLoading(true)
    setError('')
    
    try {
      // Query the backend
      const response = await axios.post('/api/query', { question })
      setAnswer(response.data.answer)
    } catch (err: any) {
      console.error('Error querying Reformind:', err)
      setError(err.response?.data?.detail || 'An error occurred while processing your question')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8 flex flex-col flex-grow">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-primary-800 mb-2">Reformind</h1>
        <p className="text-xl text-gray-600">Your Reformed AI Pastor</p>
      </header>

      <div className="max-w-3xl mx-auto w-full flex-grow flex flex-col">
        <form onSubmit={handleSubmit} className="mb-6">
          <div className="mb-4">
            <label htmlFor="question" className="block text-gray-700 mb-2 font-medium">
              Ask a theological question:
            </label>
            <textarea
              id="question"
              rows={4}
              className="input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What does the Bible say about salvation?"
            />
          </div>
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={isLoading}
          >
            {isLoading ? 'Asking...' : 'Ask Reformind'}
          </button>
        </form>

        {error && (
          <div className="p-4 mb-4 bg-red-100 border-l-4 border-red-500 text-red-700">
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="text-center py-8">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-primary-600 border-r-transparent"></div>
            <p className="mt-2 text-gray-600">Consulting Scripture...</p>
          </div>
        )}

        {answer && !isLoading && (
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-primary-800">Answer:</h2>
            <div className="prose">
              {answer.split('\n').map((paragraph, i) => (
                <p key={i} className="mb-4">{paragraph}</p>
              ))}
            </div>
          </div>
        )}
      </div>

      <footer className="mt-auto pt-8 text-center text-gray-500 text-sm">
        <p>Reformind - Biblical wisdom, Reformed truth.</p>
      </footer>
    </div>
  )
}
