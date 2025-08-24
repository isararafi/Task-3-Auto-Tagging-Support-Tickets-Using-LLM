import pandas as pd
import random
from typing import List, Dict

class SupportTicketDataGenerator:
    """Generate sample support ticket data for testing and demonstration."""
    
    def __init__(self):
        self.sample_tickets = [
            # Technical Issues
            "I'm getting a 404 error when trying to access the dashboard page.",
            "The application is running very slowly today, taking 30+ seconds to load.",
            "I can't upload files larger than 10MB, but the limit should be 100MB.",
            "The search function is not working properly - it returns no results.",
            "My browser keeps crashing when I try to export data to CSV.",
            
            # Billing Issues
            "I was charged twice for my subscription this month.",
            "My invoice shows incorrect pricing - I should have a discount.",
            "I want to cancel my subscription but can't find the option.",
            "The payment failed but my card was still charged.",
            "I need a refund for the unused portion of my annual plan.",
            
            # Account Access
            "I forgot my password and the reset link isn't working.",
            "My account was locked after too many failed login attempts.",
            "I can't access my account from a new device.",
            "My two-factor authentication is not working properly.",
            "I need to change my email address on my account.",
            
            # Feature Requests
            "Can you add a dark mode option to the interface?",
            "I'd like to see more export formats like PDF and Excel.",
            "Please add the ability to schedule reports.",
            "Can you implement a mobile app for iOS?",
            "I need bulk editing capabilities for multiple records.",
            
            # Bug Reports
            "The date picker shows incorrect dates in different timezones.",
            "When I delete a record, it doesn't actually get deleted.",
            "The notification system is sending duplicate emails.",
            "The API returns 500 errors randomly throughout the day.",
            "The dashboard charts are not displaying data correctly.",
            
            # General Inquiries
            "What are the system requirements for your software?",
            "How do I get started with the basic features?",
            "What's the difference between the free and premium plans?",
            "Do you offer training sessions for new users?",
            "What's your policy on data retention and privacy?",
            
            # Security Issues
            "I received a suspicious email claiming to be from your support team.",
            "I noticed unusual activity in my account logs.",
            "Can you help me set up additional security measures?",
            "I want to report a potential security vulnerability.",
            "My account was accessed from an unknown location.",
            
            # Performance Issues
            "The application is consuming too much memory on my computer.",
            "Database queries are taking longer than usual.",
            "The system is experiencing frequent timeouts.",
            "Backup operations are running very slowly.",
            "The application freezes when processing large datasets.",
            
            # Integration Issues
            "I'm having trouble connecting your API to my CRM system.",
            "The webhook integration is not receiving updates.",
            "How do I integrate with Salesforce?",
            "The SSO integration with Google is not working.",
            "I need help setting up the Zapier integration.",
            
            # Documentation
            "The API documentation is missing examples for Python.",
            "I can't find information about the new features.",
            "The user guide needs to be updated for the latest version.",
            "Where can I find troubleshooting guides?",
            "The documentation doesn't explain the error codes.",
            
            # Training
            "I need training for my team on the new features.",
            "Do you offer certification programs?",
            "Can you provide a demo for our company?",
            "I need help creating training materials for my users.",
            "Are there any video tutorials available?",
            
            # Refunds
            "I accidentally purchased the wrong plan, can I get a refund?",
            "The service doesn't meet my needs, I want a refund.",
            "I was charged for a feature that doesn't work as advertised.",
            "Can I get a refund for the unused portion of my subscription?",
            "I want to cancel and get a refund for the remaining time."
        ]
        
        # Ground truth labels for evaluation
        self.ground_truth = {
            "I'm getting a 404 error when trying to access the dashboard page.": ["Technical Issue", "Bug Report", "Performance"],
            "I was charged twice for my subscription this month.": ["Billing", "Refund", "Account Access"],
            "I forgot my password and the reset link isn't working.": ["Account Access", "Technical Issue", "Security"],
            "Can you add a dark mode option to the interface?": ["Feature Request", "General Inquiry", "Documentation"],
            "The date picker shows incorrect dates in different timezones.": ["Bug Report", "Technical Issue", "Performance"],
            "What are the system requirements for your software?": ["General Inquiry", "Documentation", "Training"],
            "I received a suspicious email claiming to be from your support team.": ["Security", "General Inquiry", "Account Access"],
            "The application is consuming too much memory on my computer.": ["Performance", "Technical Issue", "Bug Report"],
            "I'm having trouble connecting your API to my CRM system.": ["Integration", "Technical Issue", "Documentation"],
            "The API documentation is missing examples for Python.": ["Documentation", "General Inquiry", "Training"]
        }
    
    def generate_sample_data(self, num_tickets: int = 20) -> List[str]:
        """Generate a list of sample support tickets."""
        if num_tickets <= len(self.sample_tickets):
            return random.sample(self.sample_tickets, num_tickets)
        else:
            # If more tickets requested than available, repeat some
            return random.choices(self.sample_tickets, k=num_tickets)
    
    def get_evaluation_data(self) -> Dict[str, List[str]]:
        """Get a subset of data with ground truth labels for evaluation."""
        return {
            "tickets": list(self.ground_truth.keys()),
            "ground_truth": list(self.ground_truth.values())
        }
    
    def create_dataframe(self, tickets: List[str], classifications: List[List[Dict]] = None) -> pd.DataFrame:
        """Create a pandas DataFrame from tickets and their classifications."""
        data = []
        
        for i, ticket in enumerate(tickets):
            row = {"ticket_id": i + 1, "ticket_text": ticket}
            
            if classifications and i < len(classifications):
                for j, classification in enumerate(classifications[i]):
                    row[f"tag_{j+1}"] = classification.get("category", "")
                    row[f"confidence_{j+1}"] = classification.get("confidence", 0.0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_categories(self) -> List[str]:
        """Return the list of available categories."""
        return [
            "Technical Issue", "Billing", "Account Access", "Feature Request", 
            "Bug Report", "General Inquiry", "Security", "Performance", 
            "Integration", "Documentation", "Training", "Refund"
        ]
