import gradio as gr
import os

port = int(os.environ.get("PORT", 10000))

def test(x):
    return "Working!"

iface = gr.Interface(fn=test, inputs="text", outputs="text")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=port)