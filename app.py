import streamlit as st
import os
import sys

# Dynamically add the root directory to the system path.
# This ensures that custom modules (like 'config' and 'src') can be imported 
# flawlessly regardless of where the Streamlit command is executed from.

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import PathConfig, ModelConfig
from src.model.ner_model import BroadcastNERModel
from src.inference.predictor import BroadcastNERPredictor

# Set up the foundational configuration for the Streamlit page
st.set_page_config(
    page_title="Broadcast NER Dashboard",
    layout="wide"
)

@st.cache_resource(show_spinner="Initializing Model & Tokenizer...")
def load_predictor() -> BroadcastNERPredictor:
    """
    Loads the trained NER model and its associated tokenizer.
    
    We use Streamlit's @st.cache_resource decorator here because loading a PyTorch
    model and BERT tokenizer into memory is computationally expensive. Caching ensures 
    this only happens once during the app's lifecycle, providing instantaneous predictions 
    for the user thereafter.
    
    Returns:
        BroadcastNERPredictor: The instantiated predictor engine.
    """
    path_cfg = PathConfig()
    model_cfg = ModelConfig()
    
    # Graceful error handling: ensure the user has trained the model before launching UI
    if not os.path.exists(path_cfg.best_model_path):
        st.error(f"Checkpoint not found at {path_cfg.best_model_path}. Please run train.py first.")
        st.stop()
        
    # Load the model directly onto the CPU for inference, as GPU isn't strictly necessary 
    # for processing single sentences in real-time.
    model = BroadcastNERModel.load_model(path_cfg.best_model_path, device="cpu")
    
    # Initialize the predictor using the exact tokenizer model name defined in the config
    # to avoid any tokenizer mismatch issues. We set max_length=512 to allow the model 
    # to process long paragraphs without truncating the end of the text.
    predictor = BroadcastNERPredictor(
        model=model, 
        model_name=model_cfg.model_name, 
        device="cpu",
        max_length=512
    )
    return predictor

# --- Main Application UI ---
st.title("Broadcast NER")
st.markdown("Named Entity Recognition system fine-tuned on the CoNLL-2003 dataset using BERT.")

# Load the model into memory
predictor = load_predictor()

# Input area for the user to type or paste text
text = st.text_area(
    "Enter Text to Analyze", 
    placeholder="Type a broadcast transcript here... (e.g., 'U.N. Secretary-General Kofi Annan visited Baghdad today.')", 
    height=150
)

# Trigger inference when the user clicks the analyze button
if st.button("Analyze Entities", type="primary"):
    if text.strip():
        # Provide visual feedback while the model processes the text
        with st.spinner("Running inference..."):
            prediction = predictor.predict(text.strip())
        
        st.subheader("Extraction Results")
        
        # Handle cases where the model finds zero entities gracefully
        if not prediction.entities:
            st.info("No entities found in the provided text.")
        else:
            # Map standard CoNLL-2003 tags to human-readable names
            label_map = {
                'PER': 'Person',
                'LOC': 'Location',
                'ORG': 'Organization',
                'MISC': 'Miscellaneous'
            }
            
            # Define a visually pleasing color palette for entity highlighting
            color_map = {
                'PER': '#3b82f6',   # Blue
                'LOC': '#10b981',   # Green
                'ORG': '#8b5cf6',   # Purple
                'MISC': '#f59e0b'   # Orange
            }
            
            original_text = text.strip()
            highlighted_html = ""
            current_idx = 0
            
            # List to store structured data for the metrics table
            entity_data = []
            
            # Iterate through the predictions and construct an HTML string with inline styling
            for ent in prediction.entities:
                start_char = original_text.find(ent.text, current_idx)
                if start_char != -1:
                    end_char = start_char + len(ent.text)
                    
                    # Append the normal text that precedes the entity
                    highlighted_html += original_text[current_idx:start_char].replace('\n', '<br>')
                    
                    display_label = label_map.get(ent.label, ent.label)
                    color = color_map.get(ent.label, '#6b7280') # Default to grey if tag is unknown
                    
                    # Construct the highlighted entity badge
                    highlighted_html += (
                        f'<span style="background-color: {color}33; border: 1px solid {color}; '
                        f'border-radius: 4px; padding: 2px 4px; margin: 0 2px;">'
                        f'{ent.text} <span style="font-size: 0.8em; font-weight: bold; color: {color};">{display_label}</span></span>'
                    )
                    
                    # Move the pointer to the end of the current entity
                    current_idx = end_char
                
                # Collect metric data for the summary dataframe
                entity_data.append({
                    "Entity": ent.text,
                    "Classification": display_label,
                    "Confidence Score": f"{ent.confidence:.1%}"
                })
                    
            # Append any remaining unhighlighted text to the HTML string
            highlighted_html += original_text[current_idx:].replace('\n', '<br>')
            
            # Render the constructed HTML securely in Streamlit
            st.markdown(
                f'<div style="line-height: 2; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: #f9fafb; color: #1f2937;">{highlighted_html}</div>', 
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            
            # Present statistical summary alongside a detailed table of entities
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Total Entities Detected", len(prediction.entities))
                
                # Calculate average model confidence across all extracted entities
                avg_conf = sum(ent.confidence for ent in prediction.entities) / len(prediction.entities)
                st.metric("Average Confidence", f"{avg_conf:.1%}")
                
            with col2:
                # Display the extracted entities in a clean dataframe
                st.dataframe(entity_data, use_container_width=True)
                
    else:
        st.warning("Please enter some text into the box to begin analysis.")
