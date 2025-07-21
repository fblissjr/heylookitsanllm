#!/usr/bin/env python3
"""
Examples of different ways to send images to heylookllm
"""
import requests
import base64
from PIL import Image
import io

def image_to_base64(image_path_or_pil):
    """Convert image to base64 string"""
    if isinstance(image_path_or_pil, str):
        with open(image_path_or_pil, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    else:
        # PIL Image
        buffered = io.BytesIO()
        image_path_or_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# Example 1: Single message with multiple images (comparison task)
def single_message_multiple_images():
    """One message containing multiple images for comparison/analysis"""
    
    # Create test images
    img1 = Image.new('RGB', (100, 100), color='red')
    img2 = Image.new('RGB', (100, 100), color='blue')
    img3 = Image.new('RGB', (100, 100), color='green')
    
    img1_b64 = image_to_base64(img1)
    img2_b64 = image_to_base64(img2)
    img3_b64 = image_to_base64(img3)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these three images and describe their colors"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img3_b64}"}}
            ]
        }
    ]
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen2.5-vl-72b-mlx",
            "messages": messages,
            "max_tokens": 200
        }
    )
    
    print("=== Single Message, Multiple Images ===")
    print("Model sees all 3 images in one context")
    print("Response:", response.json()['choices'][0]['message']['content'])
    print()

# Example 2: Multiple messages, each with one image (conversation history)
def conversation_with_images():
    """Standard OpenAI conversation where each message has its own image"""
    
    img1 = Image.new('RGB', (100, 100), color='red')
    img2 = Image.new('RGB', (100, 100), color='blue')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img1)}"}}
            ]
        },
        {
            "role": "assistant",
            "content": "This image is red."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "And what color is this one?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img2)}"}}
            ]
        }
    ]
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen2.5-vl-72b-mlx",
            "messages": messages,
            "max_tokens": 200
        }
    )
    
    print("=== Conversation with Images ===")
    print("Model sees conversation history with 2 images")
    print("Response:", response.json()['choices'][0]['message']['content'])
    print()

# Example 3: Sequential processing (with new batch API)
def sequential_image_processing():
    """Process multiple images independently (requires batch processing implementation)"""
    
    images = [
        Image.new('RGB', (100, 100), color='red'),
        Image.new('RGB', (100, 100), color='blue'),
        Image.new('RGB', (100, 100), color='green')
    ]
    
    messages = []
    for i, img in enumerate(images):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe the color of image {i+1}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"}}
            ]
        })
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen2.5-vl-72b-mlx",
            "messages": messages,
            "processing_mode": "sequential",  # New parameter
            "return_individual": True,
            "max_tokens": 200
        }
    )
    
    print("=== Sequential Processing (Batch Mode) ===")
    print("Each image processed independently")
    result = response.json()
    if "completions" in result:
        for i, completion in enumerate(result["completions"]):
            print(f"Image {i+1}: {completion['choices'][0]['message']['content']}")
    print()

# Example 4: Your ShrugPrompter use case - modified for batch
def shrug_prompter_batch_example():
    """How to modify ShrugPrompter to batch multiple images"""
    
    # Instead of calling the API 4 times, batch them:
    images = [create_test_image(i) for i in range(4)]  # Your 4 workflow images
    
    messages = []
    system_prompt = "You are an image analyzer"
    user_prompt = "Describe this image in detail"
    
    for img in images:
        messages.extend([
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"}}
                ]
            }
        ])
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "qwen2.5-vl-72b-mlx",
            "messages": messages,
            "processing_mode": "sequential",
            "return_individual": True,
            "max_tokens": 512
        }
    )
    
    print("=== Batched ShrugPrompter ===")
    print("All 4 images in one request, processed sequentially")

def create_test_image(index):
    """Create a test image with a number"""
    img = Image.new('RGB', (200, 200), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), f"Image {index}", fill='black')
    return img

if __name__ == "__main__":
    print("heylookllm Multi-Image Examples\n")
    
    # These work with current implementation
    single_message_multiple_images()
    conversation_with_images()
    
    # These require the batch processing implementation
    print("\n--- Following examples require batch processing implementation ---\n")
    sequential_image_processing()
    shrug_prompter_batch_example()