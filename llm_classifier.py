import openai
import json
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupportTicketClassifier:
    def __init__(self, api_key: str = None):
        """Initialize the classifier with OpenAI API key."""
        # Try to get API key from parameter, then environment, then .env file
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            # Try to load from .env file directly
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as parameter.")
        
        # Initialize OpenAI client with the new API format
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Define support ticket categories
        self.categories = [
            "Technical Issue", "Billing", "Account Access", "Feature Request", 
            "Bug Report", "General Inquiry", "Security", "Performance", 
            "Integration", "Documentation", "Training", "Refund"
        ]
        
        # Few-shot examples for better classification
        self.few_shot_examples = [
            {
                "ticket": "I can't log into my account. It says invalid credentials.",
                "tags": ["Account Access", "Technical Issue", "Security"]
            },
            {
                "ticket": "My monthly subscription was charged twice this month.",
                "tags": ["Billing", "Refund", "Account Access"]
            },
            {
                "ticket": "The app crashes every time I try to upload a file.",
                "tags": ["Bug Report", "Technical Issue", "Performance"]
            },
            {
                "ticket": "Can you add a dark mode feature to the interface?",
                "tags": ["Feature Request", "General Inquiry", "Documentation"]
            },
            {
                "ticket": "How do I integrate your API with my existing system?",
                "tags": ["Integration", "Documentation", "Training"]
            }
        ]
    
    def zero_shot_classify(self, ticket_text: str) -> List[Dict[str, float]]:
        """
        Classify support ticket using zero-shot learning.
        Returns top 3 most probable tags with confidence scores.
        """
        prompt = f"""You are a support ticket classifier. Your task is to classify support tickets into the most appropriate categories.

Available categories: {', '.join(self.categories)}

Support Ticket: "{ticket_text}"

Instructions:
1. Analyze the ticket content carefully
2. Select the top 3 most relevant categories from the list above
3. Assign confidence scores between 0.0 and 1.0
4. Return ONLY a valid JSON array

Expected format:
[{{"category": "Category Name", "confidence": 0.95}}, {{"category": "Category Name", "confidence": 0.85}}, {{"category": "Category Name", "confidence": 0.75}}]

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies support tickets into categories. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                classifications = json.loads(result)
                # Validate and clean the results
                valid_classifications = []
                for item in classifications:
                    if isinstance(item, dict) and 'category' in item and 'confidence' in item:
                        category = item['category']
                        confidence = float(item['confidence'])
                        # Ensure category is in our list
                        if category in self.categories:
                            valid_classifications.append({
                                "category": category,
                                "confidence": min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
                            })
                
                # Sort by confidence and return top 3
                valid_classifications.sort(key=lambda x: x["confidence"], reverse=True)
                return valid_classifications[:3]
                
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return self._fallback_parse(result)
                
        except Exception as e:
            print(f"Error in zero-shot classification: {e}")
            return self._fallback_parse(ticket_text)
    
    def few_shot_classify(self, ticket_text: str) -> List[Dict[str, float]]:
        """
        Classify support ticket using few-shot learning with examples.
        Returns top 3 most probable tags with confidence scores.
        """
        # Build few-shot examples
        examples_text = ""
        for example in self.few_shot_examples:
            examples_text += f"""
Ticket: "{example['ticket']}"
Categories: {', '.join(example['tags'])}
"""
        
        prompt = f"""You are a support ticket classifier. Here are some examples of how to classify tickets:

{examples_text}

Now classify this new ticket into the most appropriate categories from: {', '.join(self.categories)}

New Ticket: "{ticket_text}"

Instructions:
1. Use the examples above as a guide
2. Select the top 3 most relevant categories
3. Assign confidence scores between 0.0 and 1.0
4. Return ONLY a valid JSON array

Expected format:
[{{"category": "Category Name", "confidence": 0.95}}, {{"category": "Category Name", "confidence": 0.85}}, {{"category": "Category Name", "confidence": 0.75}}]

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies support tickets into categories based on examples. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                classifications = json.loads(result)
                # Validate and clean the results
                valid_classifications = []
                for item in classifications:
                    if isinstance(item, dict) and 'category' in item and 'confidence' in item:
                        category = item['category']
                        confidence = float(item['confidence'])
                        # Ensure category is in our list
                        if category in self.categories:
                            valid_classifications.append({
                                "category": category,
                                "confidence": min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
                            })
                
                # Sort by confidence and return top 3
                valid_classifications.sort(key=lambda x: x["confidence"], reverse=True)
                return valid_classifications[:3]
                
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return self._fallback_parse(result)
                
        except Exception as e:
            print(f"Error in few-shot classification: {e}")
            return self._fallback_parse(ticket_text)
    
    def _fallback_parse(self, text: str) -> List[Dict[str, float]]:
        """Fallback parsing method if JSON parsing fails."""
        # Simple keyword-based classification
        text_lower = text.lower()
        classifications = []
        
        # Define keywords for each category
        category_keywords = {
            "Technical Issue": ["error", "problem", "issue", "not working", "broken", "failed", "crash"],
            "Billing": ["charge", "payment", "invoice", "bill", "subscription", "cost", "price", "money"],
            "Account Access": ["login", "password", "account", "access", "credentials", "sign in"],
            "Feature Request": ["add", "feature", "new", "request", "suggestion", "improvement"],
            "Bug Report": ["bug", "glitch", "defect", "malfunction", "wrong", "incorrect"],
            "General Inquiry": ["question", "how", "what", "when", "where", "why", "help"],
            "Security": ["security", "hack", "breach", "suspicious", "unauthorized", "privacy"],
            "Performance": ["slow", "performance", "speed", "timeout", "lag", "freeze"],
            "Integration": ["api", "integrate", "connect", "webhook", "third party"],
            "Documentation": ["documentation", "guide", "manual", "tutorial", "help"],
            "Training": ["training", "learn", "teach", "course", "education"],
            "Refund": ["refund", "money back", "cancel", "return", "credit"]
        }
        
        # Score each category based on keyword matches
        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                confidence = min(score / len(keywords) * 0.8 + 0.2, 0.95)  # Scale score to 0.2-0.95
                classifications.append({"category": category, "confidence": confidence})
        
        # Sort by confidence and return top 3
        classifications.sort(key=lambda x: x["confidence"], reverse=True)
        
        # If no matches found, return default categories
        if not classifications:
            return [
                {"category": "General Inquiry", "confidence": 0.5},
                {"category": "Technical Issue", "confidence": 0.3},
                {"category": "Documentation", "confidence": 0.2}
            ]
        
        return classifications[:3]
    
    def batch_classify(self, tickets: List[str], method: str = "zero_shot") -> List[List[Dict[str, float]]]:
        """
        Classify multiple tickets at once.
        method: "zero_shot" or "few_shot"
        """
        results = []
        for ticket in tickets:
            if method == "zero_shot":
                result = self.zero_shot_classify(ticket)
            else:
                result = self.few_shot_classify(ticket)
            results.append(result)
        return results
