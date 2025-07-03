#!/usr/bin/env python3
"""
Debug script to test image loading and base64 handling.
Run this to diagnose image-related issues.

Usage: python debug_images.py
"""

import base64
import io
import requests
from PIL import Image

def create_test_image():
    """Create a simple test image and return its base64 representation."""
    # Create a simple 10x10 red square
    img = Image.new('RGB', (10, 10), color='red')

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"

def test_base64_image(data_url):
    """Test loading a base64 image."""
    print(f"Testing base64 image (length: {len(data_url)})...")

    try:
        # Split the data URL
        if not data_url.startswith("data:image"):
            raise ValueError("Not a data URL")

        header, encoded = data_url.split(",", 1)
        print(f"Header: {header}")
        print(f"Base64 length: {len(encoded)}")

        # Check if base64 looks truncated
        if len(encoded) < 20:
            print("⚠️  Base64 data seems very short - possibly truncated")

        # Decode base64
        try:
            image_data = base64.b64decode(encoded)
            print(f"Decoded data length: {len(image_data)} bytes")
        except Exception as e:
            print(f"❌ Base64 decode failed: {e}")
            return False

        # Try to load as image
        try:
            img = Image.open(io.BytesIO(image_data))
            print(f"✅ Image loaded successfully: {img.size}, mode: {img.mode}")
            return True
        except Exception as e:
            print(f"❌ Image loading failed: {e}")
            return False

    except Exception as e:
        print(f"❌ General error: {e}")
        return False

def test_url_image(url):
    """Test loading an image from URL."""
    print(f"Testing URL image: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(response.raw)
        print(f"✅ URL image loaded: {img.size}, mode: {img.mode}")
        return True
    except Exception as e:
        print(f"❌ URL image failed: {e}")
        return False

def main():
    """Run image loading tests."""
    print("🖼️  Image Loading Debug Script")
    print("=" * 50)

    # Test 1: Create and test a good base64 image
    print("1. Testing good base64 image...")
    good_image = create_test_image()
    test_base64_image(good_image)
    print()

    # Test 2: Test the problematic base64 from the error log
    print("2. Testing problematic base64 from error log...")
    bad_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/x8AAoAB/1b/nFUAAA..."
    test_base64_image(bad_image)
    print()

    # Test 3: Test a complete 1x1 transparent pixel
    print("3. Testing 1x1 transparent pixel...")
    tiny_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQIHWNgAAIAAAUAAY27m/MAAAAASUVORK5CYII="
    test_base64_image(tiny_image)
    print()

    # Test 4: Test 1x1 red pixel (complete)
    print("4. Testing 1x1 red pixel...")
    red_pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/x8AAoAB/1b/nFUAAAAASUVORK5CYII="
    test_base64_image(red_pixel)
    print()

    # Test 5: Test URL image
    print("5. Testing URL image...")
    test_url = "https://httpbin.org/image/png"
    test_url_image(test_url)
    print()

    print("=" * 50)
    print("📋 Summary:")
    print("- If good base64 works but others fail, the issue is truncated data")
    print("- If all base64 fails, there's a decoding problem")
    print("- If URL works but base64 doesn't, focus on base64 handling")
    print()
    print("💡 For ComfyUI integration:")
    print("- Make sure base64 data isn't truncated in transmission")
    print("- Add fallback images for corrupted data")
    print("- Consider using smaller test images")

if __name__ == "__main__":
    main()
