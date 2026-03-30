import gradio as gr
from inference import predict

def vqa_interface(image, question):
    try:
        if image is None or question.strip() == "":
            return "Please upload an image and enter a question."

        answer = predict(image, question)
        return answer

    except Exception as e:
        print("ERROR:", str(e))
        return f"Error: {str(e)}"


iface = gr.Interface(
    fn=vqa_interface,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(
            label="Ask a Question",
            placeholder="e.g. What is in the image?"
        )
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🧠 Smart Visual Question Answering System",
    description="Upload any image and ask anything (works for medical + general images)",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)