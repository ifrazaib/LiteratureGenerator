import streamlit as st
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import time
import sys

# Check for sentencepiece
try:
    import sentencepiece
    st.sidebar.success("✅ sentencepiece installed")
except ImportError:
    st.error("❌ sentencepiece not installed! Run: `pip install sentencepiece`")
    st.stop()

# Page config
st.set_page_config(
    page_title="Literature Review Generator",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Literature Review Generator")
st.markdown("Generate systematic literature reviews using AI")

# Load model from Hugging Face
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model from Hugging Face Hub"""
    try:
        # Your model ID
        model_id = "Afrii/literature-review-pegasus"
        
        # Show loading message
        status = st.empty()
        status.info("📥 Downloading model from Hugging Face...")
        
        # Load tokenizer and model
        tokenizer = PegasusTokenizer.from_pretrained(model_id)
        model = PegasusForConditionalGeneration.from_pretrained(model_id)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        status.success("✅ Model loaded successfully!")
        time.sleep(1)
        status.empty()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        # Specific help for common errors
        if "SentencePiece" in str(e):
            st.error("""
            **Fix this error:**
            Run in terminal: `pip install sentencepiece`
            Then restart the app.
            """)
        elif "401" in str(e) or "403" in str(e):
            st.error("""
            **Authentication Error:**
            Your model might be private. Try:
            1. Make model public on Hugging Face
            2. Or add access token in code
            """)
        elif "404" in str(e):
            st.error("""
            **Model not found:**
            Check if the model exists: https://huggingface.co/Afrii/literature-review-pegasus
            """)
        
        return None, None, None

# Load the model
tokenizer, model, device = load_model()

# Main app
if model and tokenizer:
    st.success("✅ Model ready!")
    
    # Input
    st.subheader("📝 Input Text")
    text = st.text_area("Paste literature:", height=200)
    
    if st.button("Generate"):
        if text:
            with st.spinner("Generating..."):
                # Prepare prompt
                prompt = f"""Write a systematic literature review from the above literatures:
                1. No headings
                2. Chronological order
                3. Address methodology, findings, limitations
                4. End with research gap
                5. Include references
                
                {text}
                """
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_length=600,
                    min_length=200,
                    num_beams=4,
                    temperature=0.8,
                    early_stopping=True
                )
                
                # Decode
                review = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Show result
            st.subheader("📄 Generated Review")
            st.write(review)
            
            # Download
            st.download_button("Download", review, "review.txt")
        else:
            st.warning("Enter text first")
else:
    st.error("Model not loaded. Check errors above.")
