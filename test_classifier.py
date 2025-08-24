#!/usr/bin/env python3
"""
Simple test script for the Support Ticket Classifier
Run this to test the classifier without the Streamlit interface
"""

import os
from llm_classifier import SupportTicketClassifier
from data_generator import SupportTicketDataGenerator

def test_classifier():
    """Test the classifier with sample data."""
    
    print("🔍 Testing Support Ticket Classifier")
    print("=" * 50)
    
    try:
        # Initialize classifier
        classifier = SupportTicketClassifier()
        print("✅ Classifier initialized successfully!")
        
        # Get sample data
        data_gen = SupportTicketDataGenerator()
        sample_tickets = data_gen.generate_sample_data(3)
        
        print(f"\n📋 Testing with {len(sample_tickets)} sample tickets:")
        print("-" * 50)
        
        # Test zero-shot classification
        print("\n🔄 Testing Zero-shot Classification:")
        for i, ticket in enumerate(sample_tickets, 1):
            print(f"\nTicket {i}: {ticket[:60]}...")
            
            result = classifier.zero_shot_classify(ticket)
            if result:
                for j, pred in enumerate(result[:3]):
                    category = pred.get("category", "Unknown")
                    confidence = pred.get("confidence", 0.0)
                    print(f"  {j+1}. {category} (Confidence: {confidence:.2f})")
            else:
                print("  No classification result")
        
        # Test few-shot classification
        print("\n🎯 Testing Few-shot Classification:")
        for i, ticket in enumerate(sample_tickets, 1):
            print(f"\nTicket {i}: {ticket[:60]}...")
            
            result = classifier.few_shot_classify(ticket)
            if result:
                for j, pred in enumerate(result[:3]):
                    category = pred.get("category", "Unknown")
                    confidence = pred.get("confidence", 0.0)
                    print(f"  {j+1}. {category} (Confidence: {confidence:.2f})")
            else:
                print("  No classification result")
        
        print("\n✅ All tests completed successfully!")
        print("\n🚀 To run the full application:")
        print("streamlit run app.py")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {str(e)}")
        print("\n💡 Please ensure your OpenAI API key is set in one of the following ways:")
        print("1. Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        print("2. Set environment variable: OPENAI_API_KEY=your_api_key_here")
        print("3. Get your API key from: https://platform.openai.com/api-keys")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_classifier()
