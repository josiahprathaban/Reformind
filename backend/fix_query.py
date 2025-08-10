# fix_query.py

import os
import logging
from pathlib import Path

# Get the main.py file path
main_path = Path(__file__).parent / "main.py"

# Read the current content
with open(main_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the process_query_async function
start_str = "def process_query_async(request_id: str, question: str):"
start_idx = content.find(start_str)

if start_idx == -1:
    print("Could not find process_query_async function")
    exit(1)

# Find the next function definition to determine the end
end_str = "\n\ndef "
end_idx = content.find(end_str, start_idx)

if end_idx == -1:
    # If there's no next function, look for app.get pattern
    end_str = "\n\n@app."
    end_idx = content.find(end_str, start_idx)

if end_idx == -1:
    print("Could not find the end of process_query_async function")
    exit(1)

# Extract the function
old_function = content[start_idx:end_idx]

# Define the new function with the fix for "None" responses
new_function = """def process_query_async(request_id: str, question: str):
    \"\"\"Process a query asynchronously and store the result.\"\"\"
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
            
            # Fix potential issues in the response
            if formatted_answer.strip().startswith("None"):
                formatted_answer = formatted_answer.replace("None", "", 1).strip()
                logger.warning("Removed 'None' from the beginning of the response")
            
            # If the response is empty or just "None", generate a better response
            if not formatted_answer or formatted_answer.strip() == "None":
                logger.warning("Empty or invalid LLM response, generating formatted response from source nodes")
                formatted_answer = "# Biblical Teaching on This Topic\\n\\n"
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for i, node in enumerate(response.source_nodes[:5]):
                        reference = node.metadata.get('reference', f"Source {i+1}")
                        formatted_answer += f"**{reference}**\\n{node.text}\\n\\n"
                    formatted_answer += "## Application\\n\\nThese passages from Scripture provide clear guidance on this topic. Consider how they apply to your life and spiritual walk.\\n"
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
                        formatted_answer += "\\n\\n**Scripture References:**\\n"
                        for book, refs in sorted(scripture_refs.items()):  # Sort by book name
                            formatted_answer += f"- **{book}**: {', '.join(sorted(refs))}\\n"
                            
                        # Add a concluding statement
                        formatted_answer += "\\n\\nThese verses were used to inform this response. May they guide your understanding according to Reformed doctrine."
            except Exception as e:
                logger.warning(f"Error formatting source references: {str(e)}")
        else:
            # In retrieval-only mode, format the response from the source nodes
            formatted_answer = f"Scripture passages related to: {question}\\n\\n"
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    reference = node.metadata.get('reference', f"Source {i+1}")
                    formatted_answer += f"**{reference}**\\n{node.text}\\n\\n"
                
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
        }"""

# Replace the old function with the new one
new_content = content.replace(old_function, new_function)

# Write the new content back to the file
with open(main_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully fixed process_query_async function to handle 'None' responses")
