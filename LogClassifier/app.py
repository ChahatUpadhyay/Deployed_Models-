# app.py
import os
import pandas as pd
import gradio as gr
from gradio.themes import Soft
from classify import classify  # Your existing classify() logic

# Ensure output directory exists
dest_dir = "resources"
os.makedirs(dest_dir, exist_ok=True)

# --- Business logic: annotate uploaded CSV and return path to new CSV ---
def annotate(csv_file):
    try:
        df = pd.read_csv(csv_file.name)
        if not {"source", "log_message"}.issubset(df.columns):
            raise gr.Error("CSV must contain 'source' and 'log_message' columns.")
        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
        out_path = os.path.join(dest_dir, "annotated_output.csv")
        df.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        raise gr.Error(f"Failed to annotate CSV: {str(e)}")

# --- Gradio UI Definition with Bright Theme ---
with gr.Blocks(
    theme=Soft(
        primary_hue="green",
        secondary_hue="yellow"
    ),
    css="""
        body { background-color: #f0fff4; }
        #header_html { border-radius: 8px; padding: 16px; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        #header_html h1 { color: #2b8a3e; margin: 0; }
        #header_html p { color: #4a4a4a; margin: 8px 0 0; }
        #file-input .file-component { border: 2px dashed #a3e635 !important; border-radius: 4px; }
        #classify-btn button { background-color: #a3e635 !important; color: white !important; }
        #file-output .file-component { border-radius: 4px; }
    """
) as demo:
    # Header HTML
    gr.HTML(
        """
        <div id='header_html' style='text-align:center;'>
            <h1>📝 Log Error Classification</h1>
            <p>Upload a CSV and download it with each log labeled.</p>
        </div>
        """
    )

    # Main Interface
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            file_input = gr.File(
                label="📁 Upload CSV",
                file_types=[".csv"],
                elem_id="file-input"
            )
            classify_btn = gr.Button(
                "🚀 Classify Logs",
                variant="primary",
                elem_id="classify-btn"
            )
        with gr.Column(scale=1, min_width=300):
            file_output = gr.File(
                label="⬇️ Download Annotated CSV",
                interactive=False,
                elem_id="file-output"
            )

    # Footer Markdown
    gr.Markdown(
        """
        **Supported columns:** `source`, `log_message`  
        **Under the hood:** regex · BERT · LLM
        """
    )

    # Wire up button action
    classify_btn.click(
        fn=annotate,
        inputs=file_input,
        outputs=file_output,
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
