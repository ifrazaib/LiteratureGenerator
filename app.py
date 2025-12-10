import streamlit as st
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import textwrap
import time
from datetime import datetime
import os
import json
from pathlib import Path

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Academic Literature Review Generator",
    page_icon="",
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
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .stTextArea textarea {
        font-family: 'Monaco', 'Courier New', monospace;
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
        line-height: 1.8;
        font-size: 16px;
    }
    .stat-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #3B82F6;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #cff4fc;
        border-left: 5px solid #0dcaf0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
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

# ========== HELPER FUNCTIONS ==========
@st.cache_resource
def load_model():
    """Load the fine-tuned PEGASUS model with caching"""
    try:
        model_path = "./models/pegasus-literature-review-final"
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            st.error(" Model directory not found! Please ensure the model is in the 'models' folder.")
            return None, None, None
        
        # Check for essential files
        required_files = ['config.json', 'pytorch_model.bin']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            st.error(f" Missing model files: {', '.join(missing_files)}")
            return None, None, None
        
        with st.spinner(" Loading model... This might take a minute on first run."):
            tokenizer = PegasusTokenizer.from_pretrained(model_path)
            model = PegasusForConditionalGeneration.from_pretrained(model_path)
            
            # Set device (use GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"📱 Using device: {device}")
            model.to(device)
            model.eval()
        
        st.success(" Model loaded successfully!")
        return tokenizer, model, device
    
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
        st.info("Please ensure you have downloaded all model files to the 'models' folder.")
        return None, None, None

def clean_generated_output(generated_text):
    """
    Clean the generated output to remove any prompt fragments
    """
    if not generated_text:
        return ""
    
    # Remove the system prompt if it appears in output
    if SYSTEM_PROMPT in generated_text:
        generated_text = generated_text.replace(SYSTEM_PROMPT, "").strip()
    
    # Remove common prompt fragments
    prompt_fragments = [
        "Write a systematic literature review from the above literatures",
        "Do not give any heading and subheading",
        "Start literature from back years to the recent years",
        "The review must be in a way, to address an article",
        "Ending it in a refined research gap",
        "Use at least 30 references",
        "Use the references from the given references",
        "Give all the references (Bibliography) at the end",
        "Literature Review:",
        "Rules:",
        "1.",
        "2.",
        "3.",
        "4.",
        "5.",
        "6.",
        "7."
    ]
    
    for fragment in prompt_fragments:
        if fragment in generated_text:
            generated_text = generated_text.replace(fragment, "").strip()
    
    # Clean up any duplicate newlines or spaces
    paragraphs = [para.strip() for para in generated_text.split('\n\n') if para.strip()]
    generated_text = '\n\n'.join(paragraphs)
    
    return generated_text

def format_paragraph(text, width=80):
    """Format text into nicely wrapped paragraphs"""
    if not text.strip():
        return ""
    
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph.strip(), width=width)
            formatted_paragraphs.append(wrapped)
    
    return '\n\n'.join(formatted_paragraphs)

def generate_review(literature_text, tokenizer, model, device, params):
    """Generate literature review with given parameters"""
    if not literature_text.strip():
        return None
    
    # Combine system prompt with input
    full_input = SYSTEM_PROMPT + "\n\n" + literature_text
    
    try:
        # Tokenize
        inputs = tokenizer(
            full_input,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with progress bar
        progress_text = "Generating literature review..."
        progress_bar = st.progress(0)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=params['max_length'],
                min_length=params['min_length'],
                length_penalty=params['length_penalty'],
                num_beams=params['num_beams'],
                early_stopping=True,
                no_repeat_ngram_size=params['no_repeat_ngram'],
                temperature=params['temperature'],
                top_p=params['top_p'],
            )
        
        progress_bar.progress(100)
        
        # Decode and clean
        raw_review = tokenizer.decode(summary_ids[0].cpu(), skip_special_tokens=True)
        cleaned_review = clean_generated_output(raw_review)
        
        return cleaned_review
    
    except torch.cuda.OutOfMemoryError:
        st.error("⚠️ GPU out of memory! Try with shorter text or reduce max_length.")
        return None
    except Exception as e:
        st.error(f" Generation error: {str(e)}")
        return None

# ========== MAIN APP ==========
def main():
    # Header
    st.markdown('<h1 class="main-header"> Academic Literature Review Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
        st.markdown("###  About")
        st.markdown("""
        Generate systematic literature reviews using AI.
        Trained on academic papers using fine-tuned PEGASUS model.
        """)
        
        st.markdown("---")
        st.markdown("###  Generation Parameters")
        
        # Generation parameters in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.slider("Max Length", 200, 1024, 512, 50,
                                 help="Maximum length of generated review")
            min_length = st.slider("Min Length", 50, 500, 150, 10,
                                 help="Minimum length of generated review")
            num_beams = st.slider("Beams", 1, 8, 4,
                                help="Number of beams for beam search (higher = better quality but slower)")
        
        with col2:
            temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1,
                                  help="Creativity of generation (lower = more focused)")
            length_penalty = st.selectbox("Length Penalty", [0.5, 1.0, 1.5, 2.0, 2.5], index=3,
                                        help="Encourage longer or shorter outputs")
            no_repeat_ngram = st.selectbox("No Repeat N-gram", [2, 3, 4, 5], index=1,
                                         help="Prevent repetition of n-grams")
        
        top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.05,
                         help="Nucleus sampling parameter")
        
        st.markdown("---")
        st.markdown("###  Tips")
        st.markdown("""
        - Input should contain multiple research articles
        - Include years, methodologies, findings
        - Mention limitations of each study
        - Aim for 500-1000 words input
        - Review generation takes 30-60 seconds
        """)
        
        # Clear cache button
        if st.button(" Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading...")
            time.sleep(1)
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header"> Input Literature Text</h3>', unsafe_allow_html=True)
        
        # Text input area
        input_text = st.text_area(
            "Paste your literature text here:",
            height=350,
            placeholder="""Example format:
Smith et al. (2015) studied machine learning in healthcare using neural networks. Achieved 85% accuracy but limited interpretability.

Jones et al. (2016) addressed interpretability using decision trees. Achieved 80% accuracy with better transparency.

Brown et al. (2017) combined neural networks with explainable AI. Improved accuracy to 88% while maintaining interpretability.

Williams et al. (2018) introduced attention mechanisms. Reached 90% accuracy with better understanding.

Davis et al. (2019) applied transformers to medical data. Achieved 92% accuracy but required extensive computation.

Include at least 5-10 articles with years, methods, findings, and limitations for best results.""",
            help="Each article should include: Author(s), Year, Methodology, Findings, Limitations"
        )
        
        # Example button
        if st.button(" Load Example", use_container_width=True):
            example_text = """Smith et al. (2015) studied machine learning applications in healthcare using convolutional neural networks. They achieved 85% accuracy in disease diagnosis but noted limited model interpretability as a major drawback.

Jones et al. (2016) addressed the interpretability issue by employing decision tree algorithms. They achieved 80% accuracy with significantly better transparency, though at the cost of some predictive performance.

Brown et al. (2017) combined neural networks with explainable AI techniques like LIME and SHAP. This hybrid approach improved accuracy to 88% while maintaining reasonable interpretability, though computational requirements increased.

Williams et al. (2018) introduced attention mechanisms to neural networks for medical image analysis. This innovation reached 90% accuracy with enhanced model understanding, but required extensive labeled data.

Davis et al. (2019) applied transformer models to electronic health records, achieving 92% accuracy in predictive tasks. However, their approach demanded substantial computational resources and specialized hardware.

Miller et al. (2020) optimized transformer architectures through knowledge distillation, reducing computation time by 40% while maintaining 91% accuracy. Their work made transformer models more accessible for clinical settings.

Taylor et al. (2021) implemented federated learning for privacy-preserving healthcare AI, achieving 89% accuracy while protecting patient data. The approach showed promise but suffered from communication overhead.

Anderson et al. (2022) developed multi-modal fusion models combining medical images with clinical notes. Their ensemble approach reached 94% accuracy, though integration complexity remained a challenge.

Wilson et al. (2023) introduced quantum-inspired algorithms for medical data analysis, showing 87% accuracy with exponential speedup potential. However, practical implementation required specialized quantum hardware.

Martinez et al. (2024) created self-supervised learning frameworks requiring minimal labeled data, achieving 90% accuracy with only 10% labeled samples. The approach reduced annotation costs significantly."""
            st.session_state.example_loaded = example_text
            st.rerun()
        
        # Load example if button was clicked
        if 'example_loaded' in st.session_state:
            input_text = st.session_state.example_loaded
        
        # Input statistics
        if input_text:
            word_count = len(input_text.split())
            char_count = len(input_text)
            estimated_tokens = word_count // 0.75
            
            st.markdown(f"""
            <div class="stat-box">
                <strong> Input Statistics</strong><br>
                • Words: {word_count:,}<br>
                • Characters: {char_count:,}<br>
                • Estimated tokens: {estimated_tokens:,.0f}
            </div>
            """, unsafe_allow_html=True)
            
            if word_count < 100:
                st.warning(" Very short input. Add more content for better results.")
            elif estimated_tokens > 500:
                st.warning(" Input may be too long. Consider splitting into sections.")
    
    with col2:
        st.markdown('<h3 class="sub-header"> Configuration</h3>', unsafe_allow_html=True)
        
        # Show system prompt
        with st.expander(" View System Prompt", expanded=False):
            st.code(SYSTEM_PROMPT, language="text")
        
        # Device information
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        st.markdown(f"""
        <div class="info-box">
            <strong> System Information</strong><br>
            • Device: {device_info}<br>
            • Max input tokens: 512<br>
            • Model: PEGASUS-Large<br>
            • Fine-tuned: Yes
        </div>
        """, unsafe_allow_html=True)
        
        # Quick tips
        st.markdown("""
        <div class="info-box">
            <strong> Quick Tips</strong><br>
            1. Use consistent article format<br>
            2. Include years for chronological order<br>
            3. Mention methodologies clearly<br>
            4. Specify limitations of each study<br>
            5. Aim for 10+ articles for comprehensive review
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        generate_button = st.button(
            " Generate Literature Review",
            type="primary",
            use_container_width=True,
            disabled=not input_text.strip()
        )
    
    # Load model
    tokenizer, model, device = load_model()
    
    # Generation logic
    if generate_button and input_text.strip() and tokenizer and model:
        # Prepare parameters
        params = {
            'max_length': max_length,
            'min_length': min_length,
            'num_beams': num_beams,
            'temperature': temperature,
            'length_penalty': length_penalty,
            'no_repeat_ngram': no_repeat_ngram,
            'top_p': top_p,
        }
        
        with st.spinner("Generating literature review... This may take 30-60 seconds."):
            start_time = time.time()
            review = generate_review(input_text, tokenizer, model, device, params)
            generation_time = time.time() - start_time
        
        if review:
            # Display results
            st.markdown('<h3 class="sub-header"> Generated Literature Review</h3>', unsafe_allow_html=True)
            
            # Formatted output
            formatted_review = format_paragraph(review, width=80)
            
            st.markdown('<div class="generated-text">', unsafe_allow_html=True)
            st.text(formatted_review)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistics
            review_words = len(review.split())
            review_chars = len(review)
            review_paragraphs = review.count('\n\n') + 1
            review_sentences = review.count('.') + review.count('!') + review.count('?')
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric(" Time", f"{generation_time:.1f}s")
            with col_stat2:
                st.metric(" Words", f"{review_words:,}")
            with col_stat3:
                st.metric(" Characters", f"{review_chars:,}")
            with col_stat4:
                st.metric(" Paragraphs", review_paragraphs)
            
            # Check for references
            has_references = any(keyword in review.lower() 
                               for keyword in ['reference', 'bibliography', 'cited', 'et al.', '[', ']'])
            
            st.markdown(f"""
            <div class="stat-box">
                <strong> Review Analysis</strong><br>
                • Sentences: {review_sentences}<br>
                • Avg. sentence length: {review_words/review_sentences:.1f} words<br>
                • References included: {' Yes' if has_references else ' Limited'}<br>
                • Chronological order: {' Likely' if '2015' in review and '202' in review else ' Check'}
            </div>
            """, unsafe_allow_html=True)
            
            # Download section
            st.markdown("---")
            st.markdown('<h3 class="sub-header"> Download Results</h3>', unsafe_allow_html=True)
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            # Prepare download content
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Text file content
            text_content = f"""LITERATURE REVIEW GENERATION REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Generation time: {generation_time:.1f} seconds
Model: PEGASUS Literature Review Generator (Fine-tuned)

INPUT TEXT (First 1000 characters):
{"="*60}
{input_text[:1000]}{"..." if len(input_text) > 1000 else ""}

INPUT STATISTICS:
- Words: {len(input_text.split()):,}
- Characters: {len(input_text):,}
- Articles estimated: {input_text.count('et al.') or input_text.count('(')}

GENERATED LITERATURE REVIEW:
{"="*60}
{review}

REVIEW STATISTICS:
- Words: {review_words:,}
- Characters: {review_chars:,}
- Paragraphs: {review_paragraphs}
- Sentences: {review_sentences}
- Generation parameters: {json.dumps(params, indent=2)}
"""
            
            with col_dl1:
                st.download_button(
                    label=" Download as Text",
                    data=text_content,
                    file_name=f"literature_review_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Formatted version
            with col_dl2:
                st.download_button(
                    label=" Download Formatted",
                    data=formatted_review,
                    file_name=f"literature_review_formatted_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # JSON download
            json_data = {
                "timestamp": datetime.now().isoformat(),
                "generation_time_seconds": generation_time,
                "input": {
                    "text_preview": input_text[:500],
                    "statistics": {
                        "words": len(input_text.split()),
                        "characters": len(input_text)
                    }
                },
                "output": {
                    "review": review,
                    "statistics": {
                        "words": review_words,
                        "characters": review_chars,
                        "paragraphs": review_paragraphs,
                        "sentences": review_sentences
                    }
                },
                "parameters": params,
                "model_info": {
                    "name": "PEGASUS-Large",
                    "fine_tuned": True,
                    "max_input_tokens": 512
                }
            }
            
            with col_dl3:
                st.download_button(
                    label=" Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"literature_review_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # New generation button
            st.markdown("---")
            if st.button(" Generate Another Review", use_container_width=True):
                st.rerun()
    
    elif generate_button and not input_text.strip():
        st.warning(" Please enter some text to generate a review.")
    
    elif generate_button and (tokenizer is None or model is None):
        st.error(" Model not loaded. Please check if model files are in the 'models' folder.")
        
        # Show model folder structure
        st.info("**Expected folder structure:**")
        st.code("""
literature-review-app/
├── app.py
├── models/
│   └── pegasus-literature-review-final/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── vocab.txt
│       ├── special_tokens_map.json
│       └── generation_config.json
└── requirements.txt
        """)

# ========== RUN APP ==========
if __name__ == "__main__":
    main()