import streamlit as st
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import time
import os

# Page config
st.set_page_config(
    page_title="Literature Review Generator",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Literature Review Generator")
st.markdown("Generate systematic literature reviews using AI")

# Sidebar for info
with st.sidebar:
    st.markdown("### About")
    st.markdown("This app uses a fine-tuned PEGASUS model to generate literature reviews.")
    st.markdown("---")
    
    # Model info
    st.markdown("### Model Info")
    st.code("Afrii/literature-review-pegasus", language="text")
    
    # Device info
    if torch.cuda.is_available():
        st.success("✅ GPU Available")
    else:
        st.info("🖥️ Running on CPU")

# Load model from Hugging Face
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model from Hugging Face Hub"""
    try:
        # Your model ID
        model_id = "Afrii/literature-review-pegasus"
        
        # Show loading message
        status = st.empty()
        status.info("📥 Downloading model from Hugging Face... (First time only)")
        
        # Load tokenizer and model
        tokenizer = PegasusTokenizer.from_pretrained(model_id)
        model = PegasusForConditionalGeneration.from_pretrained(model_id)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        status.success("✅ Model loaded successfully!")
        time.sleep(1)
        status.empty()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load the model
tokenizer, model, device = load_model()

# System prompt
SYSTEM_PROMPT = """Write a systematic literature review from the above literatures:
1. Do not give any heading and subheading
2. Start from older years to recent years
3. Address each article's methodology, findings, and limitations
4. End with a research gap
5. Use at least 30 references
6. Give references at the end

Literature Review:"""

# Main app content
if model and tokenizer:
    # Input section
    st.subheader("📝 Input Literature Text")
    
    input_text = st.text_area(
        "Paste your literature text below (each article should include year, methodology, findings, limitations):",
        height=250,
        placeholder="""Example format:
Smith et al. (2015) studied machine learning in healthcare using neural networks. Achieved 85% accuracy but had limited interpretability.

Jones et al. (2016) addressed interpretability using decision trees. Achieved 80% accuracy with better transparency.

Brown et al. (2017) combined neural networks with explainable AI. Improved accuracy to 88% while maintaining interpretability.

Include at least 5-10 articles for best results.""",
        help="Each article should include: Author(s), Year, Methodology, Findings, Limitations"
    )
    
    # Word count
    if input_text:
        words = len(input_text.split())
        st.caption(f"📊 Word count: {words} words")
        
        if words < 100:
            st.warning("⚠️ Very short input. Add more content for better results.")
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button(
            "🚀 Generate Literature Review",
            type="primary",
            use_container_width=True,
            disabled=not input_text.strip()
        )
    
    # Generation function
    def generate_review(text):
        if not text.strip():
            return None
        
        # Combine with system prompt
        full_input = SYSTEM_PROMPT + "\n\n" + text
        
        try:
            # Tokenize
            inputs = tokenizer(
                full_input,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=600,
                    min_length=200,
                    length_penalty=2.0,
                    num_beams=4,
                    temperature=0.8,
                    early_stopping=True
                )
            
            # Decode
            review = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean prompt from output
            if SYSTEM_PROMPT in review:
                review = review.replace(SYSTEM_PROMPT, "").strip()
            
            return review
            
        except Exception as e:
            st.error(f"❌ Generation error: {str(e)}")
            return None
    
    # Generate when button clicked
    if generate_btn and input_text.strip():
        with st.spinner("🔄 Generating literature review... (This may take 30-60 seconds)"):
            start_time = time.time()
            review = generate_review(input_text)
            generation_time = time.time() - start_time
        
        if review:
            # Display results
            st.subheader("📄 Generated Literature Review")
            
            # Review display
            st.text_area("Review", review, height=400)
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Generation Time", f"{generation_time:.1f}s")
            with col2:
                st.metric("📝 Words", len(review.split()))
            with col3:
                st.metric("🔤 Characters", len(review))
            with col4:
                st.metric("📑 Paragraphs", review.count('\n\n') + 1)
            
            # Download section
            st.markdown("---")
            st.subheader("💾 Download")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download as Text",
                    data=review,
                    file_name="literature_review.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Generate Another", use_container_width=True):
                    st.rerun()
    
    elif generate_btn and not input_text.strip():
        st.warning("⚠️ Please enter some text first")

else:
    st.error("❌ Model failed to load. Please check your internet connection and try again.")
    st.info("""
    **Troubleshooting:**
    1. Check if the model ID is correct: `Afrii/literature-review-pegasus`
    2. Make sure you're connected to the internet
    3. Try refreshing the page
    """)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, PyTorch, and Hugging Face Transformers")
