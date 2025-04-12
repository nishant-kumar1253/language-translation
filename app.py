from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st
import re

# Language mapping for better user experience
LANGUAGE_MAPPING = {
    "English (en_XX)": "en_XX",
    "Hindi (hi_IN)": "hi_IN",
    "German (de_DE)": "de_DE",
    "Spanish (es_XX)": "es_XX",
    "Korean (ko_KR)": "ko_KR",
    "Chinese (zh_CN)": "zh_CN",
    "Bengali (bn_IN)": "bn_IN",
    "French (fr_XX)": "fr_XX",
    "Japanese (ja_XX)": "ja_XX",
    "Russian (ru_RU)": "ru_RU",
    "Tamil (ta_IN)": "ta_IN"
}

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if torch.cuda.is_available():
            model = model.half()
            
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        try:   #i'm using second try block if first try block fails like model fails then second will be executed 
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt",
                device_map="auto",
                torch_dtype="auto"
            )
            return tokenizer, model, torch.device("cpu")
        except Exception as fallback_e:
            st.error(f"Fallback loading also failed: {str(fallback_e)}")
            return None, None, None

tokenizer, model, device = load_model()

def translate_text(text, source_lang, target_lang, preserve_names=True):
    try:
        # First identify potential names in the text
        potential_names = []
        if preserve_names:
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            potential_names = [word for word in words if len(word) > 2]
        
        tokenizer.src_lang = source_lang
        encoded_text = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad(): # grad is used to disable gradient
            generate_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        translated_text = tokenizer.decode(generate_tokens[0], skip_special_tokens=True)
        
        # Post-processing to preserve names if enabled
        if preserve_names and potential_names:
            for name in potential_names:
                translated_text = re.sub(
                    r'\b' + re.escape(name.lower()) + r'\b',
                    name,
                    translated_text,
                    flags=re.IGNORECASE
                )
        
        return translated_text
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

def translate_mixed_text(text, source_lang, target_lang):
    # Split text into English and non-English (Hindi) segments
    segments = []
    current_segment = ""
    is_english = True  # Start assuming first segment is English
    
    for char in text:
        # Check if character is Devanagari (Hindi) or Latin (English)
        if is_english and ('\u0900' <= char <= '\u097F'):
            if current_segment:
                segments.append((current_segment, True))
                current_segment = ""
            is_english = False
        elif not is_english and char.isalpha() and not ('\u0900' <= char <= '\u097F'):
            if current_segment:
                segments.append((current_segment, False))
                current_segment = ""
            is_english = True
        
        current_segment += char
    
    if current_segment:
        segments.append((current_segment, is_english))
    
    # Process each segment
    results = []
    for segment, is_eng in segments:
        if is_eng and segment.strip():
            try:
                translated = translate_text(segment, source_lang, target_lang)
                results.append(translated)
            except:
                results.append(segment)  # fallback to original
        else:
            results.append(segment)
    
    return "".join(results)

# Streamlit UI
st.set_page_config(page_title="Multilingual Translator", page_icon="🌐")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fcf9ca;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌐 Multilingual Translator")
st.markdown("Translate text between multiple languages using Facebook's mBART-50 model with improved mixed-language handling")

# Initialize session state
if 'last_translation' not in st.session_state:
    st.session_state['last_translation'] = None
if 'last_source_text' not in st.session_state:
    st.session_state['last_source_text'] = ""

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Enhanced Features:**
    - Better handling of mixed English-Hindi text
    - Improved name preservation
    - More accurate paragraph translation
    
    **Supported Languages:**
    - English, Hindi, German, Spanish
    - Korean, Chinese, Bengali
    - French, Japanese, Russian, Tamil
    
    [Model Details](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
    """)
    
    if tokenizer and model:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model failed to load")

# Main content
col1, col2 = st.columns(2)

with col1:
    source_language = st.selectbox(
        "Source Language",
        options=list(LANGUAGE_MAPPING.keys()),
        index=0,
        key='source_lang'
    )

with col2:
    default_target_idx = 1 if "English" in source_language else 0
    target_language = st.selectbox(
        "Target Language",
        options=list(LANGUAGE_MAPPING.keys()),
        index=default_target_idx,
        key='target_lang'
    )

# Use a form to prevent constant reruns
with st.form("translation_form"):
    text = st.text_area(
        "Text to Translate",
        placeholder="Enter text (can mix English and Hindi)...",
        height=200,
        key='input_text'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        preserve_names = st.checkbox(
            "Preserve names",
            value=True,
            help="Attempt to keep proper nouns in their original form"
        )
    with col2:
        handle_mixed = st.checkbox(
            "Handle mixed language",
            value=True,
            help="Better handling of text containing both English and Hindi"
        )
    
    submitted = st.form_submit_button("Translate", type="primary")

# Handle translation when form is submitted
if submitted:
    if not text.strip():
        st.warning("Please enter some text to translate!")
    else:
        if tokenizer is None or model is None:
            st.error("Model failed to load. Please try refreshing the page.")
            st.stop()
            
        with st.spinner("Translating..."):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                source_lang_code = LANGUAGE_MAPPING[source_language]
                target_lang_code = LANGUAGE_MAPPING[target_language]
                
                if handle_mixed and (source_lang_code == "en_XX" and target_lang_code == "hi_IN"):
                    translation = translate_mixed_text(
                        text, 
                        source_lang_code, 
                        target_lang_code
                    )
                else:
                    translation = translate_text(
                        text, 
                        source_lang_code, 
                        target_lang_code,
                        preserve_names=preserve_names
                    )
                
                st.session_state['last_translation'] = translation
                st.session_state['last_source_text'] = text
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    st.error("The translation failed due to memory limitations. Try with shorter text.")
                else:
                    st.error(f"An error occurred: {str(e)}")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Display results if available
if st.session_state['last_translation'] is not None:
    st.subheader("Translation Result")
    st.success(st.session_state['last_translation'])
    
    # Show original text for comparison
    with st.expander("Original Text"):
        st.text(st.session_state['last_source_text'])

# Add examples
with st.expander("💡 Example Translations"):
    st.markdown("""
    **Mixed English-Hindi Input:**  
    "You can insert commas to group numbers, yeh hai mera paragraph"  
    **Output:**  
    "आप संख्याओं को समूहित करने के लिए अल्पविराम लगा सकते हैं, yeh hai mera paragraph"
    
    **English to Hindi:**  
    "The weather is nice today"  
    **Output:**  
    "आज मौसम अच्छा है"
    
    **Hindi to English:**  
    "मेरा नाम जॉन है"  
    **Output:**  
    "My name is John"
    """)