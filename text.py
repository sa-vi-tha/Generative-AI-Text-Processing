
import gradio as gr

from transformers import pipeline


# Pre-trained Models

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr") 
text_generator = pipeline("text-generation", model="gpt2")
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def translate_text(text):
    translation = translator(text)
    return translation[0]['translation_text']

def generate_text(prompt):
    response = text_generator(prompt, max_length=50, do_sample=True)
    return response[0]['generated_text']

def classify_emotion(text):
    emotion = emotion_classifier(text)
    return emotion[0]['label']

iface = gr.Interface(
    fn=lambda text, mode: (
        summarize_text(text) if mode == "Summarization" else
        translate_text(text) if mode == "Translation (EN → FR)" else
        generate_text(text) if mode == "Text Generation" else
        classify_emotion(text)
    ),
    inputs=[
        gr.Textbox(label="Enter text"),
        gr.Radio(["Summarization", "Translation (EN → FR)", "Text Generation", "Emotion Classification"], label="Choose Task")
    ],
    outputs="text",
    title="Generative AI Text Processing",
    description="Perform text summarization, translation, text generation, or emotion classification using transformers."
)

iface.launch()
