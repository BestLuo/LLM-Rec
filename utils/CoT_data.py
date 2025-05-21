import ast
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Initialize vLLM model 
model_path = "Qwen2.5-32B-Instruct"
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,  # Adjust based on the number of GPUs
    gpu_memory_utilization=0.9
)

# Prompt template
system = """You are a professional text analysis expert. Your task is to generate a detailed chain of thought analysis based on the item's textual description according to the five requirements listed below, and then summarize keywords that describe the item's characteristics. Each keyword must be should focus on the item's attributes. Please analyze objectively and rationally.
1. The characteristic 1 of the item and the original text source of characteristic 1
2. The characteristic 2 of the item and the original text source of characteristic 2
3. The characteristic 3 of the item and the original text source of characteristic 3
4. The characteristic 4 of the item and the original text source of characteristic 4
5. The characteristic 5 of the item and the original text source of characteristic 5
Provide response in strict JSON format:{'CoT':"Step-by-step analysis with source text references",'keywords':"Summarized keywords"}"""

def prepare_prompt(review_text):
    """Prepare single prompt"""
    input_text = f"reviewText：{review_text}"
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def parse_response(response):
    """Parse the model's JSON response"""
    try:
        # Clean the response text to ensure it's valid JSON format
        clean_response = response.strip()
        # Check for extra text
        json_start = clean_response.find('{')
        json_end = clean_response.rfind('}')
        if json_start >= 0 and json_end >= 0:
            clean_response = clean_response[json_start:json_end+1]
        
        # Attempt to parse JSON
        try:
            data_dict = json.loads(clean_response)
        except:
            # If parsing fails directly, use ast.literal_eval
            data_dict = ast.literal_eval(clean_response)
        
        cot_part = data_dict.get('CoT', '')
        keywords_part = data_dict.get('keywords', '')
        
        # If keywords are a list, convert them to a comma-separated string
        if isinstance(keywords_part, list):
            keywords_part = ", ".join(keywords_part)
            
        return cot_part.strip(), keywords_part, response
    except Exception as e:
        print(f"Failed to parse response: {e}")
        return "", "", response

def count_valid_json_lines(file_path):
    """Count valid JSON lines"""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("{") and line.endswith("}"):
                count += 1
    return count

def load_data_in_batches(file_path, batch_size=16):
    """Load data in batches"""
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{") or not line.endswith("}"):
                continue
            
            try:
                data = json.loads(line)
                asin = data.get("asin", "")
                style = data.get("style", {}).get("Format:", "")
                review_text = data.get("reviewText", "")
                
                if not review_text:
                    continue
                
                batch.append((asin, style, review_text))
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError:
                continue
    
    if batch:
        yield batch

def convert_dataset(input_path, output_path, batch_size=16):
    """Main processing function"""
    cot_dataset = []
    sampling_params = SamplingParams(
        max_tokens=512,  # Reduce max generation length to speed up inference
        temperature=0.1, # Low temperature to reduce randomness
        top_p=0.9,      # Nucleus sampling
        repetition_penalty=1.1,
    )
    
    # Calculate total lines to display progress
    total_lines = count_valid_json_lines(input_path)
    print(f"Detected {total_lines} valid entries")
    
    # Batch counter
    batch_counter = 0
    processed_counter = 0
    
    # Create progress bar
    progress_bar = tqdm(total=total_lines, desc="Processing Progress", unit="entries")
    
    for batch in load_data_in_batches(input_path, batch_size):
        # Prepare batch prompts
        prompts = []
        batch_data = []
        
        for asin, style, review_text in batch:
            prompt = prepare_prompt(style, review_text)
            prompts.append(prompt)
            batch_data.append((asin, style, review_text))
        
        # Batch generation
        outputs = llm.generate(prompts, sampling_params)
        
        # Process each output
        for i, output in enumerate(outputs):
            asin, style, review_text = batch_data[i]
            response = output.outputs[0].text
            
            # Parse response
            cot, keywords, response_text = parse_response(response)
            
            # Build output structure
            entry = {
                "asin": asin,
                "system_prompt": f"style：{style}\nreview_text：{review_text}",
                "response": response_text,
                "CoT": cot,
                "keywords": keywords
            }
            cot_dataset.append(entry)
            processed_counter += 1
        
        # Update progress bar
        progress_bar.update(len(batch))
        
        # Increment batch counter
        batch_counter += 1
        
    # Close progress bar
    progress_bar.close()
    
    # Save the final results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cot_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed! A total of {len(cot_dataset)} entries processed, results saved to {output_path}")

# Example usage
if __name__ == "__main__":
    convert_dataset(
        "Input_Path",
        "Output_Path",
        batch_size=8192  # Batch size, adjust based on GPU memory
    )
