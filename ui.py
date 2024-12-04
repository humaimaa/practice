import gradio as gr
from example1 import chain



with gr.Blocks(title="Chatbot") as demo:
   gr.Markdown("#chat with GPT-4 DEMO")
   with gr.row():
     gr.Markdown("")
   with gr.Column(scale=6):
      chatbox = gr.Chatbot(type="messages")

      with gr.Row():
       textbox = gr.Textbox(scale=7,container=False,placeholder="ask me anything")
       submit_button = gr.Button(value="Submit",scale=3,variant="primary")
gr.Markdown("")

def chat(question,history):
    history.append({"role":"user","content":question})
    if question == "":
     response "please ask a question"
     history.append({"role":"assistant","content":response})
    else:
      history.append({"role":"assistant","content":""})
      response chain.stream(question)

  for i in response:
   history[-1]['content']+= i
   yield " ",i

# gr.ChatInterface(
#     fn=chat,
#     type="messages"
# ).launch(
#   share = True,
# )

textbox.submit(chat,[textbox,chatbox])
submit_button.click(chat,[textbox,chatbox])
demo.launch()