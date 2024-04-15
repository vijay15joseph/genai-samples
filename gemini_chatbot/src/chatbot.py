import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

from vertexai.preview.vision_models import ImageGenerationModel

generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")


def image_generate(text_prompt):
    prompt = "Atlanta Braves batter facing St.Louis Cardinals Pitcher running to the 3rd base with excitement and thrilling finish"
    if text_prompt:
        text_prompt = text_prompt.replace("#", "")
    print(text_prompt)
    generate_response = generation_model.generate_images(
        prompt=prompt, number_of_images=4,
    )
    images = []
    for index, result in enumerate(generate_response):
        images.append(generate_response[index]._pil_image)
    return images


def remove_special_chars_regex(text):
    result = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return result


# Importing necessary libraries or modules
import gradio as gr

vertexai.init(project="vijay-gcp-demo-project", location="us-central1")
textsi_1 = """You are a baseball historian with player stats and game simulator for baseball fans. Review performance metrics and select a starting lineup.  of historical or current stats. St. Louis Cardinals players that showcases the best player by position."""
model = GenerativeModel(
    "gemini-1.5-pro-preview-0409",
    system_instruction=[textsi_1],
)
chat = model.start_chat()
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.4,
    "top_p": 0.95,
}


def generate(prompt, chat_history):
    print(prompt)
    chat_history = chat_history or []

    if prompt:
        response = chat.send_message(prompt, generation_config=generation_config, safety_settings=safety_settings)

    output = response.candidates[0].content.parts[0].text
    image_prompt = "Generate an image," + output[:50]

    images = image_generate(image_prompt)
    chat_history.append((prompt, output))
    return chat_history, images


with gr.Blocks() as demo:
    with gr.Row():
        chatbot = gr.Chatbot()
        img = gr.Gallery(
            label="Generated Images",
            show_label=True,
            elem_id="gallery",
            columns=[4],
            rows=[1],
            object_fit="contain",
            height="auto",
        )

    with gr.Row():
        msg = gr.Textbox(label="Kick off a Baseball game")
    with gr.Row():
        btn = gr.Button("Submit")
        clear = gr.ClearButton([msg, chatbot])
    btn.click(fn=generate, inputs=[msg, chatbot], outputs=[chatbot, img])

# demo.launch(show_error=True,share=True, debug=True)
demo.launch(server_name="0.0.0.0", server_port=7860)
