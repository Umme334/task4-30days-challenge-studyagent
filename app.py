import streamlit as st
from pypdf import PdfReader
from agent_logic import get_study_agent, run_agent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Study Notes Agent", page_icon="📚", layout="wide")

st.title("📚 Study Notes Agent")
st.markdown("Upload your study material (PDF) and let the AI summarize it or create a quiz for you.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Provider Selection
    provider = st.selectbox("LLM Provider", ["OpenAI", "Custom / OpenRouter", "Ollama (Local)"])
    
    api_key = None
    base_url = None
    
    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            # Clear base URL if switching back to standard OpenAI
            if "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]

    elif provider == "Custom / OpenRouter":
        base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
        api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    elif provider == "Ollama (Local)":
        base_url = st.text_input("Base URL", value="http://localhost:11434/v1")
        st.info("Make sure Ollama is running locally (`ollama serve`).")
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = "ollama" # Placeholder key required by some clients

    # Model Configuration
    st.subheader("Model Settings")
    default_model = "gpt-4o-mini"
    if provider == "Ollama (Local)":
        default_model = "llama3"
    elif provider == "Custom / OpenRouter":
        default_model = "google/gemini-2.0-flash-001"
        
    model_name = st.text_input("Model Name", value=default_model, help="e.g., gpt-4o-mini, llama3, google/gemini-2.0-flash-001")

    # Context7 Integration (Placeholder)
    st.subheader("Tools & Context")
    c7_key = st.text_input("Context7 API Key (Optional)", type="password", help="Key for Context7 MCP Tools.")
    if c7_key:
        os.environ["CONTEXT7_API_KEY"] = c7_key

    st.markdown("---")
    if st.button("Reset App", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    st.markdown("---")
    st.markdown("Built with **OpenAgents SDK** and **Streamlit**.")

# Validation
if not os.environ.get("OPENAI_API_KEY") and provider != "Ollama (Local)":
    st.warning(f"Please provide an API Key for {provider} in the sidebar to proceed.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    try:
        # Extract text from PDF
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        word_count = len(text.split())
        st.info(f"PDF loaded successfully! ({len(reader.pages)} pages, approx. {word_count} words)")
        
        # Main Interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summarizer")
            if st.button("📝 Generate Summary", use_container_width=True):
                with st.spinner("Analyzing text and generating summary..."):
                    agent = get_study_agent(model_name=model_name)
                    prompt = f"Please generate a clean, meaningful summary of the following text:\n\n{text}"
                    try:
                        summary = run_agent(agent, prompt)
                        st.session_state['summary'] = summary
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        with col2:
            st.subheader("Quiz Generator")
            quiz_type = st.selectbox("Quiz Style", ["Multiple Choice", "Mixed (MCQ + Short Answer)"])
            num_questions = st.slider("Number of Questions", 3, 10, 5)
            
            if st.button("❓ Generate Quiz", use_container_width=True):
                with st.spinner("Reading text and creating quiz..."):
                    agent = get_study_agent(model_name=model_name)
                    prompt = (
                        f"Generate a {quiz_type} quiz with {num_questions} questions based on the following text. "
                        f"Include the answer key at the end.\n\n{text}"
                    )
                    try:
                        quiz = run_agent(agent, prompt)
                        st.session_state['quiz'] = quiz
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        # Display Results
        st.divider()
        
        if 'summary' in st.session_state:
            st.markdown("### 📝 Summary")
            st.markdown(st.session_state['summary'])
            st.divider()
            
        if 'quiz' in st.session_state:
            st.markdown("### ❓ Quiz")
            st.markdown(st.session_state['quiz'])

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

else:
    st.info("Please upload a PDF to get started.")

