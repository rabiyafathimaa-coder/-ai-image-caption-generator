import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from io import BytesIO

# ✅ Page configuration
st.set_page_config(page_title="AI Image Caption Generator", layout="wide", page_icon="📘")

# ✅ Load model (cached)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ✅ Title and subtitle
st.markdown("<h1 style='text-align: center;'>🖼️ AI Image Caption Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Turn any image into a one-sentence story using AI.</p>", unsafe_allow_html=True)

# ✅ Sidebar: How to use
with st.sidebar:
    st.markdown("""
    <div style="background-color: #e6f2ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4B8BBE; font-size: 16px;">
        <h4 style="color: #4B8BBE;">🧭 How to Use This App:</h4>
        <ol style="padding-left: 20px; color: #333;">
            <li><strong>Upload</strong> any image (JPEG, PNG, or JPG).</li>
            <li>Wait a few seconds for AI to process the image.</li>
            <li>Your custom <strong>one-line story</strong> will appear.</li>
            <li>You can <strong>edit</strong> the story and <strong>download</strong> it.</li>
        </ol>
        <p style="color: #555; font-style: italic;">Tip: Use expressive, clear images for best results!</p>
    </div>
    """, unsafe_allow_html=True)

# ✅ File upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    # Display image and process in two columns
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### ✨ Generating Caption...")
        progress = st.progress(0)

        inputs = processor(images=image, return_tensors="pt")
        progress.progress(30)

        out = model.generate(**inputs)
        progress.progress(70)

        final_story = processor.decode(out[0], skip_special_tokens=True)
        progress.progress(100)

        st.markdown("### 📘 Generated Caption")
        story_text = st.text_area("You can edit the caption below:", value=final_story, height=150)
        word_count = len(story_text.split())
        st.caption(f"📝 Word count: {word_count}")

        # Download button
        st.download_button(
            label="📥 Download Caption",
            data=story_text,
            file_name="story.txt",
            mime="text/plain"
        )

        # ✅ Voice narration using gTTS
        if st.button("🔊 Play Caption as Audio"):
            tts = gTTS(text=story_text, lang='en')
            audio_bytes_io = BytesIO()
            tts.write_to_fp(audio_bytes_io)
            audio_bytes_io.seek(0)

            st.audio(audio_bytes_io, format="audio/mp3")
