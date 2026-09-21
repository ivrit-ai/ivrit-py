```python
import os
import gradio as gr
from runpod import RunPod
from runpod import rp
from runpod import runpod

os.environ["GRadio"] = "True"

class RunPodIntegration:
    def __init__(self):
        self.session = None

    def create_chat_completion(self, messages, **kwargs):
        if self.session is None:
            self.session = rp.RunPodSession()
        return self.session.create_chat_completion(messages=messages, **kwargs)

with gr.Blocks() as demo:
    chatbot = gr.ChatBot()
    with gr.Row():
        inp = gr.Textbox(
            label="Type your message and press enter to get a response.",
            placeholder="Type your message here...",
        ).style(autosize=True)
    inp.submit(
        fn=chatbot.postprocess,
        inputs=inp,
        outputs=inp,
        queue=False,
    )
    chatbot.messages(
        chatbot,
        inp,
        label="Chat with RunPod",
    ).style(autosize=True)

runpod = RunPodIntegration()
```