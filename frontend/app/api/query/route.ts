import { NextResponse } from 'next/server'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function POST(request: Request) {
  try {
    const body = await request.json()
    
    // Forward request to backend
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json(
        { detail: errorData.detail || 'Failed to process query' },
        { status: response.status }
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error: any) {
    console.error('Error processing query:', error)
    return NextResponse.json(
      { detail: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
