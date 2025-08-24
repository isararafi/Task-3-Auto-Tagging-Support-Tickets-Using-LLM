import streamlit as st
import pandas as pd
import time
import os
from llm_classifier import SupportTicketClassifier
from data_generator import SupportTicketDataGenerator
from utils import (
    calculate_accuracy, calculate_precision_recall_f1, 
    plot_category_distribution, plot_confidence_distribution,
    plot_performance_comparison, format_classification_results,
    create_summary_statistics
)

# Page configuration
st.set_page_config(
    page_title="Support Ticket Auto-Tagging System",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
        color: #333333;
    }
    .metric-card strong {
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .classification-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        margin-bottom: 0.5rem;
        color: #212529;
    }
    .classification-result strong {
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .confidence-score {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

def get_classifier():
    """Get classifier instance with proper error handling."""
    try:
        return SupportTicketClassifier()
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.info("💡 Please ensure your OpenAI API key is set in the environment variables or .env file.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        return None

def get_sample_tickets():
    """Get consistent sample tickets for the dropdown."""
    data_gen = SupportTicketDataGenerator()
    # Use a fixed set of sample tickets instead of random generation
    sample_tickets = [
        "I can't log into my account. It says invalid credentials.",
        "My monthly subscription was charged twice this month.",
        "The app crashes every time I try to upload a file.",
        "Can you add a dark mode feature to the interface?",
        "How do I integrate your API with my existing system?",
        "I'm getting a 404 error when trying to access the dashboard page.",
        "I was charged twice for my subscription this month.",
        "I forgot my password and the reset link isn't working.",
        "The date picker shows incorrect dates in different timezones.",
        "What are the system requirements for your software?"
    ]
    return sample_tickets

def main():
    # Header
    st.markdown('<h1 class="main-header">🏷️ Support Ticket Auto-Tagging System</h1>', unsafe_allow_html=True)
    
    # Check API key availability
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.markdown("""
        <div class="warning-message">
            <strong>⚠️ OpenAI API Key Required</strong><br>
            Please set your OpenAI API key in one of the following ways:<br>
            1. Create a <code>.env</code> file with: <code>OPENAI_API_KEY=your_api_key_here</code><br>
            2. Set environment variable: <code>OPENAI_API_KEY=your_api_key_here</code><br>
            3. Get your API key from: <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI Platform</a>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Method selection
        method = st.selectbox(
            "Learning Method",
            ["Zero-shot", "Few-shot"],
            help="Choose between zero-shot (no examples) or few-shot (with examples) learning"
        )
        
        # Number of tickets
        num_tickets = st.slider(
            "Number of Sample Tickets",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of sample tickets to classify"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Available Categories")
        categories = [
            "Technical Issue", "Billing", "Account Access", "Feature Request", 
            "Bug Report", "General Inquiry", "Security", "Performance", 
            "Integration", "Documentation", "Training", "Refund"
        ]
        for i, category in enumerate(categories):
            st.write(f"{i+1}. {category}")
        
        st.markdown("---")
        st.markdown("### 🔧 API Status")
        if api_key:
            st.success("✅ API Key Configured")
        else:
            st.error("❌ API Key Missing")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Single Ticket", "📋 Batch Processing", "📈 Performance Analysis"])
    
    with tab1:
        st.header("Single Ticket Classification")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Enter ticket manually", "Use sample ticket"],
            horizontal=True
        )
        
        if input_method == "Enter ticket manually":
            ticket_text = st.text_area(
                "Enter support ticket text:",
                height=150,
                placeholder="Describe your issue here..."
            )
        else:
            # Sample ticket selector with consistent options
            sample_tickets = get_sample_tickets()
            selected_ticket = st.selectbox(
                "Select a sample ticket:",
                sample_tickets,
                index=0
            )
            ticket_text = selected_ticket
        
        if st.button("🚀 Classify Ticket", type="primary"):
            if not ticket_text.strip():
                st.error("⚠️ Please enter a ticket text to classify.")
            else:
                with st.spinner("🤖 Classifying ticket..."):
                    classifier = get_classifier()
                    if classifier is None:
                        return
                    
                    try:
                        if method == "Zero-shot":
                            result = classifier.zero_shot_classify(ticket_text)
                        else:
                            result = classifier.few_shot_classify(ticket_text)
                        
                        # Display results
                        st.success("✅ Classification completed!")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📋 Classification Results")
                            if result:
                                for i, pred in enumerate(result[:3]):
                                    confidence = pred.get("confidence", 0.0)
                                    category = pred.get("category", "Unknown")
                                    
                                    # Color code based on confidence
                                    if confidence >= 0.8:
                                        color = "🟢"
                                    elif confidence >= 0.6:
                                        color = "🟡"
                                    else:
                                        color = "🔴"
                                    
                                    st.markdown(f"""
                                    <div class="classification-result">
                                        <strong>{color} {i+1}. {category}</strong><br>
                                        <span class="confidence-score">Confidence: {confidence:.2f}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No classification results available.")
                        
                        with col2:
                            st.subheader("📊 Summary")
                            if result:
                                avg_confidence = sum(p.get("confidence", 0) for p in result) / len(result)
                                st.metric("Average Confidence", f"{avg_confidence:.2f}")
                                st.metric("Top Category", result[0].get("category", "Unknown"))
                        
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
    
    with tab2:
        st.header("Batch Ticket Processing")
        
        # Batch processing options
        batch_option = st.radio(
            "Choose batch processing option:",
            ["Use sample data", "Upload CSV file", "Enter multiple tickets"],
            horizontal=True
        )
        
        tickets = []
        
        if batch_option == "Use sample data":
            data_gen = SupportTicketDataGenerator()
            tickets = data_gen.generate_sample_data(num_tickets)
            st.info(f"📊 Using {len(tickets)} sample tickets for classification.")
            
        elif batch_option == "Upload CSV file":
            uploaded_file = st.file_uploader(
                "Upload CSV file with tickets",
                type=['csv'],
                help="CSV should have a column named 'ticket_text' or 'ticket'"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'ticket_text' in df.columns:
                        tickets = df['ticket_text'].tolist()
                    elif 'ticket' in df.columns:
                        tickets = df['ticket'].tolist()
                    else:
                        st.error("CSV must contain a column named 'ticket_text' or 'ticket'")
                        tickets = []
                    
                    st.success(f"📁 Successfully loaded {len(tickets)} tickets from CSV.")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
                    tickets = []
        
        elif batch_option == "Enter multiple tickets":
            ticket_input = st.text_area(
                "Enter multiple tickets (one per line):",
                height=200,
                placeholder="Ticket 1\nTicket 2\nTicket 3..."
            )
            if ticket_input:
                tickets = [t.strip() for t in ticket_input.split('\n') if t.strip()]
                st.info(f"📝 Found {len(tickets)} tickets to classify.")
        
        if tickets and st.button("🚀 Process Batch", type="primary"):
            with st.spinner(f"🤖 Processing {len(tickets)} tickets..."):
                classifier = get_classifier()
                if classifier is None:
                    return
                
                try:
                    # Process tickets
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, ticket in enumerate(tickets):
                        if method == "Zero-shot":
                            result = classifier.zero_shot_classify(ticket)
                        else:
                            result = classifier.few_shot_classify(ticket)
                        results.append(result)
                        
                        # Update progress
                        progress = (i + 1) / len(tickets)
                        progress_bar.progress(progress)
                        time.sleep(0.1)  # Small delay for visual effect
                    
                    # Create results DataFrame
                    data_gen = SupportTicketDataGenerator()
                    df_results = data_gen.create_dataframe(tickets, results)
                    
                    st.success(f"✅ Successfully classified {len(tickets)} tickets!")
                    
                    # Display results
                    st.subheader("📊 Classification Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"ticket_classifications_{method.lower()}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    stats = create_summary_statistics(df_results)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Tickets", stats["total_tickets"])
                    with col2:
                        st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
                    with col3:
                        st.metric("Categories Used", stats["categories_used"])
                    with col4:
                        st.metric("Most Common", stats["most_common_category"])
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_dist = plot_category_distribution(df_results)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        fig_conf = plot_confidence_distribution(df_results)
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during batch processing: {str(e)}")
    
    with tab3:
        st.header("Performance Analysis")
        
        st.info("📈 This section compares zero-shot vs few-shot learning performance using evaluation data.")
        
        if st.button("🔍 Run Performance Analysis", type="primary"):
            with st.spinner("🤖 Running performance analysis..."):
                classifier = get_classifier()
                if classifier is None:
                    return
                
                try:
                    # Get evaluation data
                    data_gen = SupportTicketDataGenerator()
                    eval_data = data_gen.get_evaluation_data()
                    tickets = eval_data["tickets"]
                    ground_truth = eval_data["ground_truth"]
                    
                    # Zero-shot classification
                    st.subheader("🔄 Zero-shot Classification")
                    zero_shot_results = []
                    for ticket in tickets:
                        result = classifier.zero_shot_classify(ticket)
                        zero_shot_results.append(result)
                    
                    # Few-shot classification
                    st.subheader("🎯 Few-shot Classification")
                    few_shot_results = []
                    for ticket in tickets:
                        result = classifier.few_shot_classify(ticket)
                        few_shot_results.append(result)
                    
                    # Calculate metrics
                    zero_shot_accuracy = calculate_accuracy(zero_shot_results, ground_truth)
                    few_shot_accuracy = calculate_accuracy(few_shot_results, ground_truth)
                    
                    zero_shot_metrics = calculate_precision_recall_f1(zero_shot_results, ground_truth)
                    few_shot_metrics = calculate_precision_recall_f1(few_shot_results, ground_truth)
                    
                    zero_shot_metrics["accuracy"] = zero_shot_accuracy
                    few_shot_metrics["accuracy"] = few_shot_accuracy
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Zero-shot Metrics")
                        st.metric("Accuracy", f"{zero_shot_accuracy:.3f}")
                        st.metric("Precision", f"{zero_shot_metrics['precision']:.3f}")
                        st.metric("Recall", f"{zero_shot_metrics['recall']:.3f}")
                        st.metric("F1 Score", f"{zero_shot_metrics['f1_score']:.3f}")
                    
                    with col2:
                        st.subheader("📊 Few-shot Metrics")
                        st.metric("Accuracy", f"{few_shot_accuracy:.3f}")
                        st.metric("Precision", f"{few_shot_metrics['precision']:.3f}")
                        st.metric("Recall", f"{few_shot_metrics['recall']:.3f}")
                        st.metric("F1 Score", f"{few_shot_metrics['f1_score']:.3f}")
                    
                    # Performance comparison chart
                    st.subheader("📈 Performance Comparison")
                    fig_comp = plot_performance_comparison(zero_shot_metrics, few_shot_metrics)
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Detailed results
                    st.subheader("📋 Detailed Results")
                    
                    # Create comparison DataFrame
                    comparison_data = []
                    for i, (ticket, zero_result, few_result, truth) in enumerate(
                        zip(tickets, zero_shot_results, few_shot_results, ground_truth)
                    ):
                        comparison_data.append({
                            "Ticket": ticket[:50] + "..." if len(ticket) > 50 else ticket,
                            "Ground Truth": ", ".join(truth),
                            "Zero-shot Top": zero_result[0]["category"] if zero_result else "None",
                            "Zero-shot Conf": f"{zero_result[0]['confidence']:.2f}" if zero_result else "0.00",
                            "Few-shot Top": few_result[0]["category"] if few_result else "None",
                            "Few-shot Conf": f"{few_result[0]['confidence']:.2f}" if few_result else "0.00"
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    st.success("✅ Performance analysis completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error during performance analysis: {str(e)}")

if __name__ == "__main__":
    main()
