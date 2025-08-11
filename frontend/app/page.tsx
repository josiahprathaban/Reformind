'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'

export default function Home() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [requestId, setRequestId] = useState<string | null>(null)
  const [pollingCount, setPollingCount] = useState(0)

  // Add a polling effect to check for response
  useEffect(() => {
    // Only poll if we have a requestId and are still loading
    if (requestId && isLoading && pollingCount < 30) { // Increased to 30 attempts (2.5 minutes at 5s intervals)
      const pollTimer = setTimeout(async () => {
        try {
          console.log(`Polling for response (attempt ${pollingCount + 1})...`);
          const response = await axios.get(`/api/status/${requestId}`, {
            timeout: 10000
          });
          
          if (response.data.status === 'completed') {
            setAnswer(response.data.answer);
            setIsLoading(false);
            setRequestId(null);
            setPollingCount(0);
          } else if (response.data.status === 'failed') {
            setError(response.data.error || 'Processing failed. Please try again.');
            setIsLoading(false);
            setRequestId(null);
            setPollingCount(0);
          } else {
            // Still processing, increment polling count and continue
            setPollingCount(prev => prev + 1);
          }
        } catch (err) {
          console.error('Error polling for response:', err);
          // Continue polling even on error
          setPollingCount(prev => prev + 1);
        }
      }, 5000); // Poll every 5 seconds
      
      return () => clearTimeout(pollTimer);
    } else if (pollingCount >= 30) {
      // Give up after 30 attempts (2.5 minutes)
      setIsLoading(false);
      setError('The request is taking longer than expected. Your question will continue processing in the background. Please check back in a few minutes.');
      setRequestId(null);
      setPollingCount(0);
    }
  }, [requestId, isLoading, pollingCount]);

  const [lastRequestId, setLastRequestId] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!question.trim()) {
      setError('Please enter a question')
      return
    }

    setIsLoading(true)
    setError('')
    setAnswer('')
    setRequestId(null)
    setPollingCount(0)
    
    try {
      // Query the backend with increased timeout
      const response = await axios.post('/api/query', { question }, {
        timeout: 60000, // 1 minute timeout
      });
      
      if (response.data.answer) {
        setAnswer(response.data.answer);
      } else if (response.data.requestId) {
        // If we get a requestId, it means the request is being processed asynchronously
        const newRequestId = response.data.requestId;
        setRequestId(newRequestId);
        setLastRequestId(newRequestId);
        // Polling will be handled by the useEffect
      }
    } catch (err: any) {
      console.error('Error querying Reformind:', err)
      
      if (err.response?.data?.requestId) {
        // Even on timeout, if we have a requestId, we can poll for results
        const newRequestId = err.response.data.requestId;
        setRequestId(newRequestId);
        setLastRequestId(newRequestId);
      } else {
        // Provide more specific error messages
        if (err.code === 'ECONNABORTED') {
          setError('The request took too long to complete. The system is still processing your question. Please try again in a moment.');
        } else if (err.response?.status === 504) {
          setError('The server took too long to respond. Your question may be complex and require more processing time.');
        } else {
          setError(err.response?.data?.detail || 'An error occurred while processing your question');
          setIsLoading(false);
        }
      }
    }
  }

  const checkLastRequest = async () => {
    if (!lastRequestId) {
      setError('No previous request to check');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`/api/status/${lastRequestId}`, {
        timeout: 10000
      });
      
      if (response.data.status === 'completed') {
        setAnswer(response.data.answer);
        setIsLoading(false);
      } else if (response.data.status === 'failed') {
        setError(response.data.error || 'Processing failed. Please try again.');
        setIsLoading(false);
      } else {
        setError('Your request is still being processed. Please check back later.');
        setIsLoading(false);
      }
    } catch (err: any) {
      console.error('Error checking last request:', err);
      setError('Error checking request status. The request may have expired.');
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 flex flex-col flex-grow relative">     
      <header className="mb-8 text-center relative z-10">
        <h1 className="text-4xl font-bold text-primary-400 mb-2">Reformind</h1>
        <p className="text-xl text-gray-300">Your Reformed AI Pastor</p>
      </header>

      <div className="max-w-3xl mx-auto w-full flex-grow flex flex-col relative z-10">
        <form onSubmit={handleSubmit} className="mb-6">
          <div className="mb-4">
            <label htmlFor="question" className="block text-gray-300 mb-2 font-medium">
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
          
          {lastRequestId && !requestId && (
            <button 
              type="button"
              onClick={checkLastRequest}
              className="ml-4 px-4 py-2 rounded text-primary-400 border border-primary-500 hover:bg-dark-600 hover:bg-opacity-20 transition-all duration-300"
              disabled={isLoading}
            >
              Check Last Request
            </button>
          )}
        </form>

        {error && (
          <div className="error-card">
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="text-center py-8 glass-card rounded-lg p-6">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-primary-400 border-r-transparent"></div>
            <p className="mt-2 text-gray-300">Consulting Scripture and Reformed theology...</p>
            {pollingCount > 0 && (
              <div>
                <p className="mt-1 text-primary-400">
                  Still processing your question. Bible interpretation takes time...
                  {pollingCount > 3 && " This is a deep theological question!"}
                </p>
                <div className="w-full max-w-xs mx-auto bg-dark-400 rounded-full h-2.5 mt-3">
                  <div 
                    className="bg-primary-500 h-2.5 rounded-full" 
                    style={{ width: `${Math.min(100, (pollingCount / 30) * 100)}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  {pollingCount < 10 ? "Initial processing..." : 
                   pollingCount < 20 ? "Analyzing Scripture..." : 
                   "Finalizing response..."}
                </p>
              </div>
            )}
            <p className="mt-3 text-gray-400 text-sm italic">
              {requestId ? "Your request is being processed in the background." : "Reflecting on God's Word"}
            </p>
          </div>
        )}

        {answer && !isLoading && (
          <div className="glass-card rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-primary-400">Answer:</h2>
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  // Customize how different markdown elements are rendered
                  h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-6 mb-4 text-primary-400" {...props} />,
                  h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-5 mb-3 text-primary-400" {...props} />,
                  h3: ({node, ...props}) => <h3 className="text-lg font-bold mt-4 mb-2 text-primary-500" {...props} />,
                  p: ({node, ...props}) => <p className="mb-4 text-gray-300" {...props} />,
                  strong: ({node, ...props}) => <strong className="font-bold text-primary-400" {...props} />,
                  blockquote: ({node, ...props}) => (
                    <blockquote className="pl-4 border-l-4 border-gray-600 italic text-gray-400 my-4" {...props} />
                  ),
                  ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-4 text-gray-300" {...props} />,
                  ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-4 text-gray-300" {...props} />,
                  li: ({node, ...props}) => <li className="mb-1" {...props} />,
                }}
              >
                {answer}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      <footer className="mt-auto pt-8 text-center text-gray-500 text-sm relative z-10">
        <p>Reformind - Biblical wisdom, Reformed truth.</p>
      </footer>
    </div>
  )
}
