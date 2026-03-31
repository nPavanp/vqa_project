import gradio as gr
from inference import predict
import os

port = int(os.environ.get("PORT", 10000))


def vqa_interface(image, question):
    try:
        if image is None or question.strip() == "":
            return "Please upload an image and enter a question."

        return predict(image, question)

    except Exception as e:
        print("ERROR:", str(e))
        return f"Error: {str(e)}"


iface = gr.Interface(
    fn=vqa_interface,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🧠 Smart Visual Question Answering System",
    description="Upload any image and ask anything",
)


if __name__ == "__main__":
    print("Starting Gradio server...")
    iface.launch(
        server_name="0.0.0.0",
        server_port=port
    )