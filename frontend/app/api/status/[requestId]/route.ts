import { NextResponse } from 'next/server'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET(
  request: Request,
  { params }: { params: { requestId: string } }
) {
  try {
    const requestId = params.requestId
    
    // Check request status from backend
    const response = await fetch(`${API_URL}/status/${requestId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json(
        { detail: errorData.detail || 'Failed to check status' },
        { status: response.status }
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error: any) {
    console.error('Error checking status:', error)
    return NextResponse.json(
      { detail: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
