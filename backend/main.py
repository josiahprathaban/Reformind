# main.py
"""
This module contains the FastAPI server for the Reformind application.
It provides an API endpoint for querying the Bible using the vector index.
"""

import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core.settings import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

from indexer import load_or_create_index

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Reformind API",
    description="A Reformed AI Pastor powered by the KJV Bible",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the index
try:
    logger.info("Initializing the vector index...")
    
    # Load the index
    index = load_or_create_index()
    
    # Determine if we should use an LLM for response generation
    use_llm = os.environ.get("USE_LLM", "false").lower() == "true"
    
    if use_llm:
        # Configure a simple HuggingFace model for text generation
        try:
            llm = HuggingFaceLLM(
                model_name="microsoft/phi-2",
                tokenizer_name="microsoft/phi-2",
                context_window=2048,
                max_new_tokens=512,
                generate_kwargs={"temperature": 0.5, "do_sample": True, "top_p": 0.95},
                device_map="auto",
            )
            
            # Set the LLM in global settings
            Settings.llm = llm
            
            # Create a custom prompt for Reformed theological responses
            qa_template = PromptTemplate(
                """\
You are Reformind, a Reformed Christian AI pastor, trained to provide biblical answers based on Scripture and Reformed doctrine.

When answering questions, always:
1. Use Scripture as your primary authority
2. Reflect Reformed theological understanding
3. Provide clear reasoning
4. Cite relevant Bible verses
5. Be pastoral and edifying in tone

Consider these passages from Scripture that are relevant to the question:
{context_str}

USER QUESTION: {query_str}

Answer the question thoroughly from a Reformed perspective, using the scripture passages above as your primary source. 
Structure your response with clear reasoning, biblical evidence, and application. 
Be concise but complete in your explanation.

YOUR ANSWER:
"""
            )
            
            # Create query engine with LLM for synthesis
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",  # Better mode for synthesizing multiple sources
                similarity_top_k=7,  # Retrieve more relevant passages
                text_qa_template=qa_template,
                verbose=True
            )
            logger.info("Using LLM for response generation")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {str(e)}. Falling back to retrieval only mode.")
            use_llm = False
    
    if not use_llm:
        # Create a retrieval-only query engine
        Settings.llm = None
        query_engine = index.as_query_engine(
            response_mode="no_text",  # Just retrieve nodes without synthesizing 
            similarity_top_k=10
        )
        logger.info("Using retrieval-only mode (no LLM)")
    
    logger.info("Vector index initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector index: {str(e)}")
    raise

# Define request and response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_bible(request: QueryRequest) -> Dict[str, Any]:
    """
    Query the Bible using the vector index.
    
    Args:
        request: QueryRequest object containing the question
        
    Returns:
        Dict containing the answer
    """
    try:
        logger.info(f"Received question: {request.question}")
        
        # Check if question is empty
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Query the index
        response = query_engine.query(request.question)
        
        # Format the response
        if os.environ.get("USE_LLM", "false").lower() == "true":
            # When using an LLM, the response is already synthesized
            formatted_answer = str(response)
            
            # Add source references and organize them better
            try:
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    scripture_refs = {}
                    
                    # Group by book for better organization
                    for node in response.source_nodes:
                        if "reference" in node.metadata:
                            ref = node.metadata['reference']
                            book = ref.split()[0]  # Extract book name
                            if book not in scripture_refs:
                                scripture_refs[book] = []
                            scripture_refs[book].append(ref)
                    
                    if scripture_refs:
                        formatted_answer += "\n\n**Scripture References:**\n"
                        for book, refs in scripture_refs.items():
                            formatted_answer += f"\n{book}: {', '.join(refs)}\n"
                            
                        # Add a concluding statement
                        formatted_answer += "\n\nMay these scriptures guide your understanding according to Reformed doctrine."
            except Exception as e:
                logger.warning(f"Error formatting source references: {str(e)}")
        else:
            # In retrieval-only mode, format the response from the source nodes
            formatted_answer = f"Results for: {request.question}\n\n"
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    reference = node.metadata.get('reference', f"Source {i+1}")
                    formatted_answer += f"{reference}:\n{node.text}\n\n"
            else:
                formatted_answer += "No relevant scripture passages found."
        
        # Log the response
        logger.info(f"Generated answer for question: {request.question}")
        
        return {"answer": formatted_answer}
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dict containing the status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
