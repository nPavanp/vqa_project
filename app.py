print("STEP 1: Starting app.py")

import gradio as gr
print("STEP 2: Imported gradio")

from inference import predict
print("STEP 3: Imported inference")

import os
port = int(os.environ.get("PORT", 10000))

print("STEP 4: Creating interface")

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
        gr.Image(type="filepath"),
        gr.Textbox()
    ],
    outputs="text"
)

print("STEP 5: Before launch")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=port)