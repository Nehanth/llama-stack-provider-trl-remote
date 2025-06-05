#!/usr/bin/env python3
"""
Test script for DPO-trained model

This script loads the DPO-trained DistilGPT2 model from ./checkpoints/dpo_model/
and tests it with various prompts to see how the preference optimization affected
the model's responses.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_dpo_model():
    """Load the DPO-trained model and tokenizer."""
    print("🔄 Loading DPO-trained model...")
    
    try:
        model_path = "./checkpoints/dpo_model/"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded from {model_path}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"
        )
        print(f"✅ Model loaded from {model_path}")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate a response from the model for a given prompt."""
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    generated_text = response[len(prompt):].strip()
    
    return generated_text

def test_model():
    """Test the DPO-trained model with various prompts."""
    
    print("🧪 Testing DPO-Trained DistilGPT2 Model")
    print("=" * 50)
    
    # Load the model
    model, tokenizer = load_dpo_model()
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Test prompts (similar to training data)
    test_prompts = [
        "What is machine learning?",
        "Write a hello world program",
        "Explain the concept of fine-tuning",
        "How do you debug a program?",
        "What are the best practices for coding?"
    ]
    
    print("\n🎯 Testing Model Responses:")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔤 Test {i}: {prompt}")
        print("-" * 40)
        
        try:
            # Generate response
            response = generate_response(
                model, tokenizer, prompt, 
                max_length=150, 
                temperature=0.7
            )
            
            print(f"🤖 Model Response:")
            print(f"   {response}")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")
    print("\n💡 Note: Compare these responses to see if DPO training")
    print("   improved the quality compared to the base DistilGPT2 model.")

def compare_with_base_model():
    """Optional: Compare with original DistilGPT2 for reference."""
    
    print("\n🔄 Loading original DistilGPT2 for comparison...")
    
    try:
        base_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        prompt = "What is machine learning?"
        
        print(f"\n🔤 Comparison Prompt: {prompt}")
        print("-" * 40)
        
        # DPO-trained model response
        model, tokenizer = load_dpo_model()
        if model and tokenizer:
            dpo_response = generate_response(model, tokenizer, prompt, max_length=120)
            print(f"🎯 DPO-trained model: {dpo_response}")
        
        # Base model response
        base_response = generate_response(base_model, base_tokenizer, prompt, max_length=120)
        print(f"🏷️  Base DistilGPT2:   {base_response}")
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

if __name__ == "__main__":
    # Run the main test
    test_model()
    
    # Optional comparison (uncomment to enable)
    print("\n" + "=" * 70)
    compare_with_base_model() 