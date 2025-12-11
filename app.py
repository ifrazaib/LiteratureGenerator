import streamlit as st
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import os
import sys
import time

# ========== SETUP ==========
# Reduce memory usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Literature Review Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .stTextArea textarea {
        font-size: 14px;
        line-height: 1.5;
    }
    .generated-text {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        margin: 10px 0;
        white-space: pre-wrap;
        font-family: 'Georgia', serif;
        line-height: 1.6;
    }
    .stat-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ========== SYSTEM PROMPT ==========
SYSTEM_PROMPT = """Write a systematic literature review from the above literatures while obeying the following rules:
1. Do not give any heading and subheading
2. Start literature from back years to the recent years, like from 2015, then 16,......, 2025, in a systematic way
3. The review must be in a way, to address an article, its methodology in a sentence, its drawbacks, and then the next article which address that limitation or drawback.
4. Ending it in a refined research gap
5. Use at least 30 references, meaning thirty articles.
6. Use the references from the given references and not others.
7. Give all the references (Bibliography) at the end

Literature Review:"""

# ========== MODEL LOADING ==========
@st.cache_resource
def load_model():
    """Load YOUR fine-tuned model from local files"""
    try:
        model_path = "./models/pegasus-literature-review-final"
        
        # Check if model files exist
        if not os.path.exists(model_path):
            st.error("❌ Model directory not found!")
            return None, None, None
        
        # Check for essential files
        essential_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
        for file in essential_files:
            if not os.path.exists(os.path.join(model_path, file)):
                st.error(f"❌ Missing essential file: {file}")
                return None, None, None
        
        st.info("🔄 Loading your fine-tuned model...")
        
        # Load tokenizer and model from local files
        tokenizer = PegasusTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        model = PegasusForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Use CPU (Streamlit Cloud has no GPU)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        st.success("✅ Your fine-tuned model loaded successfully!")
        return tokenizer, model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def clean_output(text):
    """Clean generated output"""
    if not text:
        return ""
    
    # Remove system prompt if present
    if SYSTEM_PROMPT in text:
        text = text.replace(SYSTEM_PROMPT, "").strip()
    
    # Remove prompt fragments
    fragments = [
        "Write a systematic literature review",
        "Do not give any heading",
        "Start literature from back years",
        "Literature Review:",
        "Rules:",
        "1.", "2.", "3.", "4.", "5.", "6.", "7."
    ]
    
    for fragment in fragments:
        if fragment in text:
            text = text.replace(fragment, "").strip()
    
    return text.strip()

# ========== MAIN APP ==========
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Academic Literature Review Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#666;">Using your fine-tuned PEGASUS model</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Max Length", 150, 500, 300)
            min_length = st.slider("Min Length", 50, 200, 100)
        
        with col2:
            num_beams = st.slider("Beams", 1, 4, 2)
            temperature = st.slider("Temp", 0.1, 1.0, 0.8, 0.1)
        
        length_penalty = st.selectbox("Length Penalty", [1.0, 1.5, 2.0, 2.5], index=2)
        no_repeat_ngram = st.selectbox("No Repeat", [2, 3], index=1)
        
        st.markdown("---")
        st.markdown("### ℹ️ Model Info")
        st.markdown("""
        - **Model**: Your fine-tuned PEGASUS
        - **Training**: Custom literature reviews
        - **Status**: Loaded from local files
        """)
        
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared!")
            time.sleep(1)
            st.rerun()
    
    # Load model
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None:
        st.error("""
        ⚠️ **Model not loaded!**
        
        Please ensure:
        1. Model files are in `models/pegasus-literature-review-final/`
        2. All files are uploaded to GitHub
        3. Required files: `config.json`, `pytorch_model.bin`, `tokenizer_config.json`
        """)
        
        # Show what files we have
        if os.path.exists("./models/pegasus-literature-review-final"):
            st.info("**Files found in model directory:**")
            import glob
            files = glob.glob("./models/pegasus-literature-review-final/*")
            for file in files:
                st.write(f"- {os.path.basename(file)}")
        
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Input Literature Text")
        
        input_text = st.text_area(
            "Paste your literature text:",
            height=300,
            placeholder="""Example format:
Smith et al. (2015) studied topic X using method Y. Found results but noted limitations A.

Jones et al. (2016) addressed limitation A using method Z. Achieved better results but had limitation B.

Brown et al. (2017) solved limitation B with new approach. Improved accuracy by 15%.

Include multiple articles with years, methods, findings, and limitations for best results."""
        )
        
        if input_text:
            words = len(input_text.split())
            st.info(f"📊 Input: {words:,} words, ~{words//0.75:,.0f} tokens")
    
    with col2:
        st.markdown("### 📋 System Prompt")
        with st.expander("View prompt rules"):
            st.code(SYSTEM_PROMPT)
        
        st.markdown("### ⚡ Quick Tips")
        st.markdown("""
        - Include 5+ articles
        - Mention years (2015, 2016, etc.)
        - Describe methodologies
        - Note limitations
        - 300-800 words ideal
        """)
    
    st.markdown("---")
    
    # Generate button
    if st.button("🚀 Generate Literature Review", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating systematic review... This may take 30-60 seconds."):
                try:
                    # Prepare input
                    full_input = SYSTEM_PROMPT + "\n\n" + input_text
                    
                    # Tokenize
                    inputs = tokenizer(
                        full_input,
                        max_length=512,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            inputs.input_ids,
                            max_length=max_length,
                            min_length=min_length,
                            num_beams=num_beams,
                            temperature=temperature,
                            length_penalty=length_penalty,
                            no_repeat_ngram_size=no_repeat_ngram,
                            early_stopping=True
                        )
                    
                    generation_time = time.time() - start_time
                    
                    # Decode
                    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Clean output
                    cleaned_output = clean_output(raw_output)
                    
                    # Display results
                    st.markdown("### 📄 Generated Literature Review")
                    
                    st.markdown('<div class="generated-text">', unsafe_allow_html=True)
                    st.write(cleaned_output)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Statistics
                    output_words = len(cleaned_output.split())
                    output_chars = len(cleaned_output)
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("⏱️ Time", f"{generation_time:.1f}s")
                    with col_stat2:
                        st.metric("📝 Words", output_words)
                    with col_stat3:
                        st.metric("🔤 Characters", output_chars)
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Review",
                        data=cleaned_output,
                        file_name=f"literature_review_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Generation error: {str(e)}")
                    st.info("""
                    **Possible fixes:**
                    1. Reduce input length
                    2. Lower max_length parameter
                    3. Try with simpler text
                    """)
    
    # Footer
    st.markdown("---")
    st.caption("Powered by your fine-tuned PEGASUS model • Built with Streamlit")

# Run the app
if __name__ == "__main__":
    main()
