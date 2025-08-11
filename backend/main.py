# main.py
"""
This module contains the FastAPI server for the Reformind application.
It provides an API endpoint fo            # Configure a smaller model for faster download and inference
            llm = HuggingFaceLLM(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                context_window=2048,
                max_new_tokens=512,
                generate_kwargs={
                    "temperature": 0.5, 
                    "do_sample": True, 
                    "top_p": 0.95,
                    "repetition_penalty": 1.1,  # Prevent repetitive outputs
                    "max_length": 2048,  # Control maximum token length
                },
                device_map=device_map,
                model_kwargs=model_kwargs,
            )
            
            # Add performance tracking attributes
            llm._last_query_time = 0
            llm._last_token_count = 0
            
            # Monkey patch the complete method to track performance
            original_complete = llm.complete
            
            def complete_with_metrics(prompt, **kwargs):
                start_time = time.time()
                response = original_complete(prompt, **kwargs)
                end_time = time.time()
                
                # Track performance metrics
                llm._last_query_time = end_time - start_time
                llm._last_token_count = len(response.text.split())
                
                return response
                
            llm.complete = complete_with_metricsible using the vector index.
"""

import os
import logging
import time
import uuid
import threading
import torch
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Set Hugging Face environment variables to improve download experience
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars for cleaner logs

# Optimize PyTorch for the powerful CPU
os.environ["OMP_NUM_THREADS"] = "16"  # Use all CPU cores for OpenMP
os.environ["MKL_NUM_THREADS"] = "16"  # Use all CPU cores for MKL
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallelism in tokenizers

# Load environment variables from .env file
load_dotenv()

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

# Force enable LLM mode
os.environ["USE_LLM"] = "true"
logger.info("LLM mode forcibly enabled")

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

# Set maximum concurrent requests
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent processing to prevent overload

# Load the index
try:
    logger.info("Initializing the vector index...")
    
    # Check if CUDA is available and set up device
    if torch.cuda.is_available():
        logger.info(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        # Set optimizations for GPU
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("CUDA not available, checking for other acceleration options...")
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel XPU acceleration (for Intel Arc GPUs)
            logger.info(f"Intel XPU is available! Using Intel GPU")
            device = "xpu"
        else:
            logger.info("Using CPU with optimized settings")
            device = "cpu"
    
    # Load the index
    index = load_or_create_index()
    
    # Determine if we should use an LLM for response generation
    use_llm = os.environ.get("USE_LLM", "false").lower() == "true"
    logger.info(f"USE_LLM environment variable: {os.environ.get('USE_LLM', 'not set')}")
    logger.info(f"Using LLM mode: {use_llm}")
    
    if use_llm:
        # Configure a simple HuggingFace model for text generation
        try:
            # Suppress HuggingFace warnings
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            
            print("Attempting to load language model - this may take a few minutes the first time...")
            start_time = time.time()
            
            # Set a timeout for model loading (3 minutes)
            model_timeout = float(os.environ.get("MODEL_TIMEOUT", "180"))  # Default 3 minutes
            
            # Configure model settings based on available hardware
            if device == "cuda" or device == "xpu":
                # GPU/XPU configuration (Intel Arc)
                logger.info(f"Configuring model to use {device} acceleration")
                model_kwargs = {
                    "torch_dtype": torch.float16,  # Use half precision for GPU
                }
                # Don't pass device directly to avoid errors
                device_map = {"": 0} if device == "cuda" else "auto"
            else:
                # CPU configuration (powerful Intel CPU)
                logger.info("Configuring model for optimized CPU usage")
                model_kwargs = {
                    "torch_dtype": torch.float16,  # Still use half precision for speed
                    # Using native PyTorch threading instead of model-specific threading
                }
                device_map = "cpu"
            
            # Configure a smaller model for faster download and inference
            llm = HuggingFaceLLM(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                context_window=2048,
                max_new_tokens=512,
                generate_kwargs={
                    "temperature": 0.5, 
                    "do_sample": True, 
                    "top_p": 0.95,
                    "repetition_penalty": 1.1,  # Prevent repetitive outputs
                    "max_length": 2048,  # Control maximum token length
                },
                device_map=device_map,
                model_kwargs=model_kwargs,
            )
            
            elapsed = time.time() - start_time
            print(f"Model loaded successfully in {elapsed:.2f} seconds!")
            logger.info(f"Model loaded successfully in {elapsed:.2f} seconds using {device} device")
            
            # Log memory usage for diagnostics
            if device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(f"GPU Memory: Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB")
            elif device == "xpu":
                try:
                    memory_allocated = torch.xpu.memory_allocated() / 1024**2
                    memory_reserved = torch.xpu.memory_reserved() / 1024**2
                    logger.info(f"XPU Memory: Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB")
                except:
                    logger.info("Could not determine XPU memory usage")
            
            # Set the LLM in global settings
            Settings.llm = llm
            
            # Create a custom prompt for Reformed theological responses
            qa_template = PromptTemplate(
                """
You are Reformind, a Reformed Christian AI pastor. Your job is to answer questions in a style similar to ChatGPT: clear, conversational, structured, and deeply explanatory, always rooted in Scripture and Reformed doctrine.

When you answer:
- Start with a short, engaging summary of the biblical position in plain language.
- Break down your answer into clear, numbered sections with headings (e.g., "1. Faith is Trust in God", "2. Faith Leads to Obedience").
- For each section, explain the point in simple terms, as if teaching someone new to the topic.
- Quote 1-2 relevant Bible verses for each point, using **bold** for references and including the verse text.
- After the main points, add a section called "In Summary" that briefly recaps the answer.
- End with a practical application or encouragement for the reader.
- Use markdown formatting for structure and emphasis.
- Never just list verses or references—always explain and connect them to the question.
- Be warm, pastoral, and encouraging, but also clear and direct.
- Make your answer at least 300-500 words for depth.

Here are some relevant Scripture passages for this question:
{context_str}

USER QUESTION: {query_str}

Now, write a thorough, ChatGPT-style answer as described above.

YOUR ANSWER:
"""
            )
            
            # Create query engine with LLM for synthesis
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",  # Better mode for synthesizing multiple sources
                similarity_top_k=25,  # Retrieve more relevant passages for a comprehensive response
                text_qa_template=qa_template,
                verbose=True
            )
            logger.info("Using LLM for response generation")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {str(e)}. Falling back to retrieval only mode.")
            use_llm = False
            
            # Tell the user what happened
            print("="*80)
            print("Failed to initialize the language model. Using retrieval-only mode instead.")
            print("This means you'll get direct Bible verses without AI-generated summaries.")
            print("Error details:", str(e))
            print("="*80)
    
    if not use_llm:
        # Create a retrieval-only query engine
        Settings.llm = None
        query_engine = index.as_query_engine(
            response_mode="no_text",  # Just retrieve nodes without synthesizing 
            similarity_top_k=20
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
    answer: str = ""
    requestId: str = ""
    status: str = "completed"

class StatusResponse(BaseModel):
    status: str
    answer: str = ""
    error: str = "" 

# Store for tracking in-progress requests
request_store = {}

def process_query_async(request_id: str, question: str):
    """Process a query asynchronously and store the result."""
    try:
        logger.info(f"Processing async request {request_id} for question: {question}")
        
        # Extract key topics for better search
        original_question = question
        
        # Add explicit related terms for certain topics to improve retrieval
        topic_expansions = {
            "sexual": ["fornication", "adultery", "lust", "immorality", "purity"],
            "marriage": ["husband", "wife", "divorce", "marry", "matrimony"],
            "sin": ["transgression", "iniquity", "wickedness", "evil", "temptation"],
            "salvation": ["saved", "redeemed", "justified", "sanctified", "elect"],
            "faith": ["belief", "trust", "confidence", "assurance", "doctrine"],
        }
        
        # Check if we should expand the query
        expanded_question = original_question
        for topic, expansions in topic_expansions.items():
            if topic.lower() in original_question.lower():
                expanded_terms = " ".join(expansions)
                expanded_question = f"{original_question} {expanded_terms}"
                logger.info(f"Expanded question with terms for '{topic}': {expanded_question}")
                break
        
        # Query the index with potentially expanded question
        response = query_engine.query(expanded_question)
        
        # Format the response
        logger.info(f"USE_LLM environment variable in query: {os.environ.get('USE_LLM', 'not set')}")
        use_llm_for_response = os.environ.get("USE_LLM", "false").lower() == "true"
        logger.info(f"Using LLM for response formatting: {use_llm_for_response}")
        
        # Log retrieved passages for debugging
        if hasattr(response, 'source_nodes') and response.source_nodes:
            logger.info(f"Retrieved {len(response.source_nodes)} passages:")
            for i, node in enumerate(response.source_nodes[:5]):  # Log first 5 for brevity
                ref = node.metadata.get('reference', f"Source {i+1}")
                logger.info(f"  {i+1}. {ref}: {node.text[:100]}...")
        else:
            logger.info("No source nodes found in response")
        
        if use_llm_for_response:
            # When using an LLM, the response is already synthesized
            formatted_answer = str(response)
            
            # Check if response looks like just a list of references instead of a proper answer
            if formatted_answer.count("Book:") > 3 or formatted_answer.count("Reference:") > 3:
                logger.warning("Response appears to be just a list of references. Regenerating a structured answer.")
                # Force a more structured answer
                formatted_answer = "# Biblical Teaching on This Topic\n\n"
                
                # Group verses by topic/theme if possible
                topics = {
                    "God's Design": [],
                    "Biblical Prohibitions": [],
                    "Consequences": [],
                    "Grace and Redemption": []
                }
                
                # Add some verses to each topic
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    # Distribute verses among topics
                    for i, node in enumerate(response.source_nodes[:16]):
                        topic_idx = i % 4  # Simple distribution
                        topic_keys = list(topics.keys())
                        if topic_idx < len(topic_keys):
                            reference = node.metadata.get('reference', f"Source {i+1}")
                            topics[topic_keys[topic_idx]].append((reference, node.text))
                
                # Create structured response with the grouped verses
                for topic, verses in topics.items():
                    if verses:
                        formatted_answer += f"## {topic}\n\n"
                        for reference, text in verses:
                            formatted_answer += f"**{reference}**\n{text}\n\n"
                        formatted_answer += "---\n\n"
                
                formatted_answer += "## Summary\n\nThese passages from Scripture provide clear guidance on this topic. The Bible teaches us to honor God with our bodies and minds, to flee from temptation, and to seek purity in all aspects of life.\n\n## Application\n\nConsider how these biblical principles apply to your life and spiritual walk. Pray for God's strength to live according to His Word and for His grace when you fall short.\n"
            
            # Fix potential issues in the response
            elif formatted_answer.strip().startswith("None"):
                formatted_answer = formatted_answer.replace("None", "", 1).strip()
                logger.warning("Removed 'None' from the beginning of the response")
            
            # If the response is empty or just "None", generate a better response
            elif not formatted_answer or formatted_answer.strip() == "None":
                logger.warning("Empty or invalid LLM response, generating formatted response from source nodes")
                formatted_answer = "# Biblical Teaching on This Topic\n\n"
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for i, node in enumerate(response.source_nodes[:5]):
                        reference = node.metadata.get('reference', f"Source {i+1}")
                        formatted_answer += f"**{reference}**\n{node.text}\n\n"
                    formatted_answer += "## Application\n\nThese passages from Scripture provide clear guidance on this topic. Consider how they apply to your life and spiritual walk.\n"
                else:
                    formatted_answer = "I couldn't find a clear biblical answer to this question. Please try rephrasing or asking about a different topic."
            
            # Add source references and organize them better
            try:
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    scripture_refs = {}
                    
                    # Group by book for better organization
                    for node in response.source_nodes:
                        if "reference" in node.metadata:
                            ref = node.metadata['reference']
                            book = ref.split()[0] if " " in ref else ref  # Extract book name
                            if book not in scripture_refs:
                                scripture_refs[book] = []
                            if ref not in scripture_refs[book]:  # Avoid duplicates
                                scripture_refs[book].append(ref)
                    
                    if scripture_refs:
                        formatted_answer += "\n\n**Scripture References:**\n"
                        for book, refs in sorted(scripture_refs.items()):  # Sort by book name
                            formatted_answer += f"- **{book}**: {', '.join(sorted(refs))}\n"
                            
                        # Add a concluding statement
                        formatted_answer += "\n\nThese verses were used to inform this response. May they guide your understanding according to Reformed doctrine."
            except Exception as e:
                logger.warning(f"Error formatting source references: {str(e)}")
        else:
            # In retrieval-only mode, format the response from the source nodes
            formatted_answer = f"Scripture passages related to: {question}\n\n"
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    reference = node.metadata.get('reference', f"Source {i+1}")
                    formatted_answer += f"**{reference}**\n{node.text}\n\n"
                
                # Add a concluding statement
                formatted_answer += "These scripture passages are provided to help you understand this topic from a Biblical perspective. May they guide your understanding according to Reformed doctrine."
            else:
                formatted_answer += "No relevant scripture passages found."
        
        # Store the result
        request_store[request_id] = {
            "status": "completed",
            "answer": formatted_answer,
            "timestamp": time.time()
        }
        
        logger.info(f"Completed async request {request_id}")
        
    except Exception as e:
        logger.error(f"Error processing async query {request_id}: {str(e)}")
        request_store[request_id] = {
            "status": "failed",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/query", response_model=QueryResponse)
async def query_bible(request: QueryRequest) -> Dict[str, Any]:
    """
    Query the Bible using the vector index.
    
    Args:
        request: QueryRequest object containing the question
        
    Returns:
        Dict containing the answer or a request ID for async processing
    """
    try:
        logger.info(f"Received question: {request.question}")
        
        # Check if question is empty
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if we're not overloading the system
        active_requests = sum(1 for req in request_store.values() if req["status"] == "processing")
        
        if active_requests >= MAX_CONCURRENT_REQUESTS:
            raise HTTPException(
                status_code=503, 
                detail="System is currently processing too many requests. Please try again in a few minutes."
            )
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Start asynchronous processing
        request_store[request_id] = {
            "status": "processing",
            "timestamp": time.time()
        }
        
        # Process query in a separate thread
        thread = threading.Thread(
            target=process_query_async,
            args=(request_id, request.question)
        )
        thread.daemon = True
        thread.start()
        
        # Return request ID for client to poll
        return {"requestId": request_id, "status": "processing"}
    
    except Exception as e:
        logger.error(f"Error starting query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/status/{request_id}", response_model=StatusResponse)
async def get_query_status(request_id: str) -> Dict[str, Any]:
    """
    Check the status of an asynchronous query.
    
    Args:
        request_id: The ID of the request to check
        
    Returns:
        Dict containing the status and possibly the answer
    """
    try:
        if request_id not in request_store:
            raise HTTPException(status_code=404, detail="Request not found")
        
        request_data = request_store[request_id]
        
        # Clean up old requests (older than 2 hours)
        current_time = time.time()
        for rid in list(request_store.keys()):
            if current_time - request_store[rid]["timestamp"] > 7200:  # 2 hours (increased from 30 minutes)
                del request_store[rid]
        
        if request_data["status"] == "completed":
            return {
                "status": "completed",
                "answer": request_data["answer"]
            }
        elif request_data["status"] == "failed":
            return {
                "status": "failed",
                "error": request_data.get("error", "Unknown error occurred")
            }
        else:
            return {"status": "processing"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking query status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking query status: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/benchmark")
async def run_benchmark() -> Dict[str, Any]:
    """
    Run a performance benchmark.
    
    Returns:
        Dict containing benchmark results
    """
    try:
        import psutil
        
        start_time = time.time()
        
        # Get system information
        cpu_info = {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "CPU usage": f"{psutil.cpu_percent()}%",
        }
        
        memory_info = {
            "Total memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Available memory": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            "Used memory": f"{psutil.virtual_memory().used / (1024**3):.2f} GB",
            "Memory percent": f"{psutil.virtual_memory().percent}%"
        }
        
        # Test embedding performance
        test_text = "What does the Bible say about faith, hope, and love in the context of salvation?"
        embed_start = time.time()
        if hasattr(Settings.embed_model, "get_text_embedding"):
            embedding = Settings.embed_model.get_text_embedding(test_text)
            embed_time = time.time() - embed_start
        else:
            embed_time = 0
            embedding = None
        
        # Test LLM performance if available
        llm_info = {}
        if Settings.llm:
            llm_start = time.time()
            response = Settings.llm.complete(test_text)
            llm_time = time.time() - llm_start
            tokens_generated = len(response.text.split())
            tokens_per_second = tokens_generated / llm_time if llm_time > 0 else 0
            
            llm_info = {
                "Generation time": f"{llm_time:.2f} seconds",
                "Tokens generated": tokens_generated,
                "Tokens per second": f"{tokens_per_second:.2f}"
            }
        
        # Test index query performance
        query_start = time.time()
        nodes = index.as_retriever(similarity_top_k=5).retrieve(test_text)
        query_time = time.time() - query_start
        
        # Total benchmark time
        total_time = time.time() - start_time
        
        return {
            "benchmark_total_time": f"{total_time:.2f} seconds",
            "cpu": cpu_info,
            "memory": memory_info,
            "embedding": {
                "time": f"{embed_time:.2f} seconds",
                "dimensions": len(embedding) if embedding else "N/A"
            },
            "llm": llm_info,
            "index_query": {
                "time": f"{query_time:.2f} seconds",
                "nodes_retrieved": len(nodes)
            },
            "device_used": device
        }
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dict containing the status and system information
    """
    import psutil
    
    # Count active requests
    active_requests = sum(1 for req in request_store.values() if req["status"] == "processing")
    
    # Get system resource usage
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_percent = memory_info.percent
    available_memory_gb = memory_info.available / (1024**3)
    
    # GPU info if available
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            gpu_info[f"cuda:{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": f"{memory_allocated:.2f}",
                "memory_reserved_gb": f"{memory_reserved:.2f}"
            }
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            memory_allocated = torch.xpu.memory_allocated() / (1024**3)
            memory_reserved = torch.xpu.memory_reserved() / (1024**3)
            gpu_info["xpu"] = {
                "name": "Intel XPU (Arc Graphics)",
                "memory_allocated_gb": f"{memory_allocated:.2f}",
                "memory_reserved_gb": f"{memory_reserved:.2f}"
            }
        except:
            gpu_info["xpu"] = {"name": "Intel XPU (Arc Graphics)"}
    
    # Performance metrics
    if hasattr(llm, "_last_query_time") and hasattr(llm, "_last_token_count"):
        performance = {
            "last_query_time": llm._last_query_time,
            "last_token_count": llm._last_token_count,
            "tokens_per_second": llm._last_token_count / llm._last_query_time if llm._last_query_time > 0 else 0
        }
    else:
        performance = {"status": "No queries processed yet"}
    
    return {
        "status": "healthy",
        "active_requests": active_requests,
        "llm_mode": os.environ.get("USE_LLM", "false").lower() == "true",
        "total_completed_requests": sum(1 for req in request_store.values() if req["status"] == "completed"),
        "total_failed_requests": sum(1 for req in request_store.values() if req["status"] == "failed"),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "available_memory_gb": f"{available_memory_gb:.2f}",
            "device_used": device,
            "gpu": gpu_info
        },
        "performance": performance
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Check if --use-llm is provided as a command-line argument
    if "--use-llm" in sys.argv:
        os.environ["USE_LLM"] = "true"
        print("LLM mode enabled via command line")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
