# app_enhanced_with_docs.py
import streamlit as st
import torch
import time
import json
from generator import load_pipeline, generate_images

st.set_page_config(page_title="AI Image Generator", layout="wide")
st.title("🎨 AI-Powered Image Generator")
st.write("Feel free to use Simple or Avanced Modes. The generated images will be store on the folder `outputs/`.")

# -----------------------------
# Sidebar (info + policy)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.markdown(f"**Device:** {device}")
    st.markdown("---")

    st.markdown("### Hardware notes")
    st.markdown("- GPU (CUDA) = very fast\n- CPU = slow, use fewer steps")
    st.markdown("---")

    # Operational & Safety toggles
    st.markdown("### Operational & Safety Controls")
    enforce_wm = st.checkbox("Enforce watermark (prevents disabling)", value=True)
    strict_filtering = st.checkbox("Strict content filtering", value=True)
    st.markdown("---")
    st.markdown("Need help? See the docs & research sections in the main UI.")

# --- model info dictionary (can also be loaded from models.json) ---
# --- MODEL INFO dict (use your existing MODEL_INFO) ---
MODEL_INFO = {
    "runwayml/stable-diffusion-v1-5": {
        "name": "Stable Diffusion v1.5",
        "strengths": [
            "Fast and lightweight",
            "Produces stable and predictable results",
            "Great for general-purpose image generation"
        ],
        "recommended_for": "Illustrations, concept art, everyday prompts"
    },
    "dreamlike-art/dreamlike-photoreal-2.0": {
        "name": "Dreamlike Photoreal 2.0",
        "strengths": [
            "Very strong photorealism",
            "Excellent lighting and texture quality",
            "Produces lifelike subjects"
        ],
        "recommended_for": "Photorealistic portraits, realistic scenes, product renders"
    },
    "prompthero/openjourney": {
        "name": "OpenJourney",
        "strengths": [
            "Midjourney-inspired artistic style",
            "Stylized, dramatic compositions",
            "Strong fantasy and cinematic look"
        ],
        "recommended_for": "Fantasy art, stylized illustrations, anime-like or MJ-style outputs"
    },
    "hakurei/waifu-diffusion": {
        "name": "Waifu Diffusion",
        "strengths": [
            "Excellent anime-style rendering",
            "Clean line-art and vibrant colors",
            "Great for characters & stylized scenes"
        ],
        "recommended_for": "Anime artwork, manga-style portraits, character design"
    }
}

# -----------------------------
# MODEL SELECTION (compact info for selected model)
# -----------------------------
st.header("⚙️ Model Selection")

model_list = list(MODEL_INFO.keys())

colA, colB, colC = st.columns([6, 1.6, 1.6])

with colA:
    # show friendly name in the dropdown using format_func
    def fmt(mid):
        info = MODEL_INFO.get(mid, {})
        return info.get("name", mid)

    # use a unique key for the selectbox
    model_id = st.selectbox("Choose AI Model", model_list, index=0, format_func=fmt, key="model_select_dropdown")

    # compact expander: only for the selected model
    selected_info = MODEL_INFO.get(model_id, {})
    with st.expander(f"About — {selected_info.get('name', model_id)}", expanded=False):
        st.write(f"**Model ID:** `{model_id}`")
        st.write("**Recommended for:**", selected_info.get("recommended_for", "-"))
        st.write("**Strengths:**")
        for s in selected_info.get("strengths", []):
            st.write(f"- {s}")

with colB:
    st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
    load_btn = st.button("Load Model", use_container_width=True, key="load_model_btn")

with colC:
    st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
    reload_btn = st.button("Reload Model", use_container_width=True, key="reload_model_btn")

# LOAD / RELOAD actions (same behavior as before)
if load_btn:
    with st.spinner("Loading model..."):
        try:
            pipe = load_pipeline(model_id, device=device)
            st.session_state["pipe"] = pipe
            st.session_state["model_id"] = model_id
            st.success(f"Model loaded: {model_id}")
        except Exception as e:
            st.error(f"Load failed: {e}")

if reload_btn:
    st.session_state.pop("pipe", None)
    with st.spinner("Reloading..."):
        try:
            pipe = load_pipeline(model_id, device=device)
            st.session_state["pipe"] = pipe
            st.session_state["model_id"] = model_id
            st.success("Reloaded model")
        except Exception as e:
            st.error(f"Reload failed: {e}")

# --------------------------------------------------------
# MAIN AREA — MODEL SELECTOR (Dipindah ke tampilan utama)
# --------------------------------------------------------

# -------------------------
# Prompt Controls
# -------------------------
BANNED_TERMS = ["porn", "nude", "nsfw", "child", "rape", "kill", "terror", "bomb", "suicide"]

def is_prompt_allowed(prompt: str, strict: bool = True):
    p = prompt.lower()
    for w in BANNED_TERMS:
        if w in p:
            return False, w
    # basic heuristic examples
    if strict:
        # block any mentions of minors or sexual/violent content beyond BANNED_TERMS
        taboo = ["minor", "underage", "childporn", "sexual"]
        for t in taboo:
            if t in p:
                return False, t
    return True, None


mode = st.radio("Mode:", ["Simple", "Advanced"], index=0, horizontal=True)

with st.expander("🧾 Example prompts"):
    st.markdown("""
- `a futuristic city at sunset, ultra-detailed, photorealistic`
- `robot portrait in Van Gogh style`
""")

left, right = st.columns([1, 1])

with left:
    st.subheader("Prompt")
    prompt = st.text_area("Prompt", value="a futuristic city at sunset, ultra-detailed")

    st.subheader("Output options")
    out_format = st.selectbox("Format", ["PNG", "JPEG"])
    watermark = st.checkbox("Add Watermark", value=True)
    if enforce_wm:
        # force watermark on
        watermark = True
        st.caption("Watermark enforced by policy")

    watermark_text = st.text_input("Watermark text", value="AI-Generated")

    if mode == "Simple":
        n_images = st.slider("Number of images", 1, 4, 1)
        steps = 30
        guidance = 7.5
        negative_prompt = "lowres, blurry, bad anatomy"
        seed = 0
        style = ""
    else:
        style = st.selectbox("Style", ["", "photorealistic", "artistic", "cartoon", "Van Gogh"])
        negative_prompt = st.text_input("Negative prompt", value="lowres, blurry, watermark")
        n_images = st.slider("Number of images", 1, 6, 1)
        guidance = st.slider("CFG Scale", 1.0, 15.0, 7.5)
        steps = st.slider("Steps", 10, 80, 30)
        seed = st.number_input("Seed (0 = random)", min_value=0, value=0)

with right:
    st.subheader("Output Panel")
    result_container = st.container()
    progress_text = st.empty()
    progress_bar = st.progress(0)

# ---------------------------------
# Generate button
# ---------------------------------
generate_btn = st.button("🎨 Generate")

if generate_btn:
    if "pipe" not in st.session_state:
        # Simple warning
        st.warning("Please select a model and click 'Load Model' before generating images.")
        # Optionally focus user to model selector with an info line:
        st.info("Tip: Choose a model from the dropdown (Model Selection) and click 'Load Model' — then try Generate again.")
    else:
        # lanjutkan proses generate seperti biasa...
        pipe = st.session_state["pipe"]
        # ... existing generate code ...

# ---------------------------------
# Docs & Research panels (ESSENCE)
# ---------------------------------
st.markdown("---")
cols = st.columns([2, 1])

with cols[0]:
    st.header("📚 Quick Docs & How to Run")

    with st.expander("Running on GPU (recommended)"):
        st.markdown("""
        - Requires NVIDIA GPU + CUDA
        - Install PyTorch with CUDA support (example for CUDA 12.1):
          ```
          pip install torch --index-url https://download.pytorch.org/whl/cu121
          pip install -r requirements.txt
          ```
        - Much faster inference and recommended for production use.
        """)

    with st.expander("Running on CPU (fallback)"):
        st.markdown("""
        - Works on any machine but significantly slower.
        - Use fewer steps and fewer parallel images to reduce latency.
        - Example CPU install:
          ```
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          ```
        """)

    with st.expander("Deployment notes & environment variables"):
        st.markdown("""
        - Use `TORCH_DEVICE=cpu` or `cuda` to force device selection.
        - Ensure sufficient VRAM for chosen model (SDXL and larger models require more).
        - For servers: run with `--server.address 0.0.0.0` and secure your endpoint.
        - Consider `xformers`, `accelerate`, or `torch.compile` for perf improvements.
        """)

    st.header("⚖️ Ethical AI Use")

    with st.expander("Responsible use guidelines"):
        st.markdown("""
        - Do not generate content that is illegal, sexually explicit, violent, hateful, or involves minors.
        - Avoid content that promotes real-world harm or criminal activity.
        - Avoid creating deepfakes or impersonation of real people without consent.
        - Provide clear user guidance about model limitations and potential biases.
        """)

    with st.expander("Content filtering & enforcement"):
        st.markdown("""
        - The app enforces a banned-terms filter (configurable) to block obvious unsafe prompts.
        - Toggle *Strict content filtering* in the sidebar to increase blocking sensitivity.
        - Watermarking should be enforced for public outputs to indicate AI origin.
        - Keep logs and metadata (prompts, timestamps, model id) for auditability.
        """)

    with st.expander("Watermarking & Transparency"):
        st.markdown("""
        - Watermark generated images to signal AI origin (text or subtle visual mark).
        - Optionally include metadata JSON next to images describing prompt, model, and seed.
        - If sharing images externally, disclose they are AI-generated and include usage terms.
        """)

    with st.expander("Why this matters"):
        st.markdown("""
        - Protects user safety and reduces misuse risk.
        - Helps comply with legal & platform policies.
        - Improves trust and transparency for downstream consumers.
        """)

    st.markdown("---")

with cols[1]:
    # You can keep other content here: quick links, recent runs, or small UI widgets.
    st.header("🔬 Research notes")
    with  st.expander("GANs (Generative Adversarial Networks"):
        st.markdown("""good for fast generation and some artistic styles. They use a generator vs. discriminator training setup (e.g., StyleGAN). """)
    with  st.expander("Diffusion Models"):
        st.markdown("""state-of-the-art for high-fidelity conditional image generation (Stable Diffusion, DALL·E 2, Imagen). They iteratively denoise from noise to image and support classifier-free guidance.""")   
    with  st.expander("Prompt engineering best practices"):
        st.markdown("""
            - Be specific about subject, style, camera/lens, lighting, and desired detail level.
            - Use negative prompts to suppress artifacts (e.g., "lowres, bad anatomy, watermark").
            - Iterate — small prompt changes often produce large differences. """)

# ---------------------------------
# Generate action
# ---------------------------------
if generate_btn:
    allowed, term = is_prompt_allowed(prompt, strict=strict_filtering)
    if not allowed:
        st.error(f"Blocked word detected: **{term}**")
    elif "pipe" not in st.session_state:
        st.error("Model belum di-load!")
    else:
        pipe = st.session_state["pipe"]
        seed_val = None if seed == 0 else seed

        start = time.perf_counter()

        def progress_cb(frac, msg):
            try:
                progress_bar.progress(max(0.0, min(frac, 1.0)))
                elapsed = int(time.perf_counter() - start)
                progress_text.text(f"{msg} – {elapsed}s")
            except Exception:
                pass

        with st.spinner("Generating..."):
            try:
                results = generate_images(
                    pipe,
                    prompt,
                    style=style,
                    negative_prompt=negative_prompt,
                    n_images=n_images,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    seed=seed_val,
                    add_wm=watermark,
                    watermark_text=watermark_text,
                    out_format=out_format,
                    progress_callback=progress_cb
                )
            except Exception as e:
                st.error(f"Failed: {e}")
                results = []

        progress_bar.progress(1.0)
        progress_text.text("Done!")

        if results:
            cols = result_container.columns(len(results))
            for col, res in zip(cols, results):
                col.image(res["path"])
                with open(res["path"], "rb") as f:
                    col.download_button("Download", f, file_name=res["filename"]) 
        else:
            st.warning("No image generated.")

# End of file
