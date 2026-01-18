#!/usr/bin/env python
"""
API Key Chat Test - Test your OpenAI API key by chatting interactively.
"""

from config import OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI

def chat_with_api():
    """Interactive chat session to test the OpenAI API key."""
    
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-admin"):
        print("⚠️  WARNING: Using a default/invalid API key!")
        print("Please set a valid OPENAI_API_KEY in your .env file.")
        return False
    
    try:
        print(f"✅ Initializing chat with model: {OPENAI_MODEL}")
        client = OpenAI(api_key=OPENAI_API_KEY)
        conversation = []
        
        print("\n" + "="*60)
        print("Chat Session Started - Type 'exit' to quit")
        print("="*60 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            conversation.append({"role": "user", "content": user_input})
            
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=conversation,
                    max_tokens=500,
                    temperature=0.7
                )
                
                assistant_message = response.choices[0].message.content
                conversation.append({"role": "assistant", "content": assistant_message})
                
                print(f"\nAssistant: {assistant_message}\n")
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
                conversation.pop()  # Remove failed user message
        
        return True
        
    except Exception as e:
        print(f"❌ Chat initialization FAILED!")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = chat_with_api()
    exit(0 if success else 1)
