import gradio as gr
from inference import predict
import os

def vqa_interface(image, question):
    if image is None or question.strip() == "":
        return "Please upload an image and enter a question."

    return predict(image, question)


iface = gr.Interface(
    fn=vqa_interface,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs="text",
    title="🧠 Smart Visual Question Answering System"
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print(f"Starting server on port {port}...")

    iface.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )