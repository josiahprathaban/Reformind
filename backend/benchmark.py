# benchmark.py
"""
Utility script to benchmark system performance for LLM and embedding operations.
"""

import os
import time
import torch
import platform
import psutil
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_system_info():
    """Get system information for benchmarking."""
    try:
        cpu_info = {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "CPU usage": f"{psutil.cpu_percent()}%",
            "CPU frequency": f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "Unknown"
        }
        
        memory_info = {
            "Total memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Available memory": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            "Used memory": f"{psutil.virtual_memory().used / (1024**3):.2f} GB",
            "Memory percent": f"{psutil.virtual_memory().percent}%"
        }
        
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info[f"GPU {i}"] = {
                    "Name": torch.cuda.get_device_name(i),
                    "Memory allocated": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB",
                    "Memory reserved": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB"
                }
        else:
            # Check for Intel XPU (Arc Graphics)
            try:
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    gpu_info["XPU"] = {
                        "Name": "Intel XPU (Arc Graphics)",
                        "Devices": torch.xpu.device_count()
                    }
                    if hasattr(torch.xpu, 'memory_allocated'):
                        gpu_info["XPU"]["Memory allocated"] = f"{torch.xpu.memory_allocated() / (1024**3):.2f} GB"
                        gpu_info["XPU"]["Memory reserved"] = f"{torch.xpu.memory_reserved() / (1024**3):.2f} GB"
            except:
                gpu_info["Status"] = "No GPU acceleration detected or XPU support not enabled"
            
        system_info = {
            "Platform": platform.system(),
            "Platform version": platform.version(),
            "Platform release": platform.release(),
            "Architecture": platform.machine(),
            "Python version": platform.python_version(),
            "CPU": cpu_info,
            "Memory": memory_info,
            "GPU": gpu_info
        }
        
        return system_info
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"Error": str(e)}

def benchmark_llm(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Benchmark LLM performance."""
    logger.info(f"Benchmarking LLM: {model_name}")
    
    try:
        # Measure model loading time
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_time = time.time() - start_time
        logger.info(f"Tokenizer loading time: {load_time:.2f} seconds")
        
        # Determine the device to use
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            # Check for Intel XPU
            try:
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    device = "xpu"
                    logger.info("Using Intel XPU device (Arc Graphics)")
                else:
                    device = "cpu"
                    logger.info("Using CPU device")
            except:
                device = "cpu"
                logger.info("Using CPU device (XPU check failed)")
        
        # Load model with appropriate settings
        start_time = time.time()
        try:
            # For Intel Arc Graphics, try to use BF16
            if device == "xpu":
                logger.info("Loading model with Intel XPU optimizations (BF16)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                # For other devices
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            
            model = model.to(device)
        except Exception as e:
            logger.warning(f"Error loading model with specialized settings: {e}. Trying generic loading.")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device)
        model_load_time = time.time() - start_time
        logger.info(f"Model loading time: {model_load_time:.2f} seconds")
        
        # Benchmark inference time
        test_prompts = [
            "What does the Bible say about faith?",
            "Explain the doctrine of justification.",
            "How should Christians view the relationship between faith and works?"
        ]
        
        inference_times = []
        token_generation_rates = []
        
        for prompt in test_prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            # Warm up
            with torch.no_grad():
                model.generate(input_ids, max_length=20)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(input_ids, max_length=100)
            inference_time = time.time() - start_time
            
            tokens_generated = len(output[0]) - len(input_ids[0])
            tokens_per_second = tokens_generated / inference_time
            
            inference_times.append(inference_time)
            token_generation_rates.append(tokens_per_second)
            
            logger.info(f"Prompt: '{prompt}'")
            logger.info(f"  Inference time: {inference_time:.2f} seconds")
            logger.info(f"  Tokens generated: {tokens_generated}")
            logger.info(f"  Tokens per second: {tokens_per_second:.2f}")
        
        results = {
            "Model": model_name,
            "Device": device,
            "Tokenizer loading time": f"{load_time:.2f} seconds",
            "Model loading time": f"{model_load_time:.2f} seconds",
            "Average inference time": f"{sum(inference_times)/len(inference_times):.2f} seconds",
            "Average tokens per second": f"{sum(token_generation_rates)/len(token_generation_rates):.2f}",
            "System info": get_system_info()
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error benchmarking LLM: {e}")
        return {"Error": str(e)}

if __name__ == "__main__":
    # Install required packages if necessary
    try:
        import psutil
    except ImportError:
        os.system("pip install psutil")
        import psutil
    
    # Run benchmarks
    system_info = get_system_info()
    logger.info("System Information:")
    for key, value in system_info.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    logger.info(f"  {k}:")
                    for k2, v2 in v.items():
                        logger.info(f"    {k2}: {v2}")
                else:
                    logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    
    # Run LLM benchmark
    results = benchmark_llm()
    logger.info("Benchmark Results:")
    for key, value in results.items():
        if key != "System info":
            logger.info(f"{key}: {value}")
