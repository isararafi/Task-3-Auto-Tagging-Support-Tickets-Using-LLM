# Auto Tagging Support Tickets Using LLM

This project implements an automated support ticket tagging system using Large Language Models (LLM) with prompt engineering techniques.

## Features

- **Zero-shot Learning**: Direct classification without training examples
- **Few-shot Learning**: Classification with example demonstrations
- **Multi-class Prediction**: Returns top 3 most probable tags per ticket
- **Performance Comparison**: Compare zero-shot vs few-shot accuracy
- **Interactive UI**: Streamlit-based web interface

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API Key**:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `llm_classifier.py`: LLM-based classification logic
- `data_generator.py`: Sample data generation
- `utils.py`: Utility functions
- `requirements.txt`: Python dependencies

## Usage

1. Start the application using `streamlit run app.py`
2. Choose between Zero-shot or Few-shot learning
3. Input support tickets or use sample data
4. View classification results and performance metrics

## Skills Demonstrated

- Prompt engineering
- LLM-based text classification
- Zero-shot and few-shot learning
- Multi-class prediction and ranking
