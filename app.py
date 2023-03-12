import gradio as gr
from axis_finder import gradio_inference

def run_axis_finder(input_str):
    input_str = input_str.split(' ')
    return gradio_inference(input_str[0], input_str[1], input_str[2])

demo = gr.Interface(
    run_axis_finder,
    gr.Textbox(value="Road Access[0.1:0.9], Starting Point(y,x), Carbon Prefix[0:7], \n exp:0.7 (44, 119) 2"),
    "image")
demo.launch()