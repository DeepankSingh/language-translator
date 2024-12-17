from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

# Define available language models
language_models = {
    "Punjabi": "Helsinki-NLP/opus-mt-pa-en",  # Punjabi to English
    "Marathi": "Helsinki-NLP/opus-mt-mr-en",  # Marathi to English
    "Bengali": "Helsinki-NLP/opus-mt-bn-en"   # Bengali to English
}

def load_model(language):
    model_name = language_models.get(language)
    if model_name:
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer
    else:
        raise ValueError("Model not found for selected language.")

def translate_text(source_text, source_language, target_language):
    model, tokenizer = load_model(source_language)

    # Tokenize the input text
    inputs = tokenizer(source_text, return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True)

    # Decode the translated text
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translation

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>Language Translator</h1>")

        with gr.Row():
            source_language = gr.Dropdown(choices=["Punjabi", "Marathi", "Bengali"], label="Source Language", value="Punjabi")
            target_language = gr.Dropdown(choices=["English"], label="Target Language", value="English")

        with gr.Row():
            source_text = gr.Textbox(placeholder="Type text here...", label="Text to Translate", lines=4)

        with gr.Row():
            translate_button = gr.Button("Translate", variant="primary")

        with gr.Row():
            output_text = gr.Textbox(label="Translated Text", interactive=False, lines=4)

        # Define a function to be triggered when the "Translate" button is clicked
        def on_translate_click(source_text, source_language, target_language):
            translation = translate_text(source_text, source_language, target_language)
            return translation

        # Attach the translation function to the translate button
        translate_button.click(on_translate_click,
                               inputs=[source_text, source_language, target_language],
                               outputs=[output_text])

    # Launch the Gradio interface
    demo.launch()

# Call the function to build and launch the UI
build_ui()

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>Language Translator</h1>")

        with gr.Row():
            source_language = gr.Dropdown(choices=["Punjabi", "Marathi", "Bengali"], label="Source Language", value="Punjabi")
            target_language = gr.Dropdown(choices=["English"], label="Target Language", value="English")

        with gr.Row():
            source_text = gr.Textbox(placeholder="Type text here...", label="Text to Translate", lines=4)

        with gr.Row():
            translate_button = gr.Button("Translate", variant="primary")

        with gr.Row():
            output_text = gr.Textbox(label="Translated Text", interactive=False, lines=4)

        # Define a function to be triggered when the "Translate" button is clicked
        def on_translate_click(source_text, source_language, target_language):
            translation = translate_text(source_text, source_language, target_language)
            return translation

        # Attach the translation function to the translate button
        translate_button.click(on_translate_click,
                               inputs=[source_text, source_language, target_language],
                               outputs=[output_text])

    # Launch the Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

# Call the function to build and launch the UI
build_ui()