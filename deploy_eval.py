"""
Experiment 12: Deployment and Evaluation of a Generative AI Application Using Cloud-Based APIs and AI Frameworks
Course: CS4V48 - GenAI & LLM Laboratory
"""

import evaluate
import gradio as gr
from transformers import pipeline

def summarize_text(input_text, summarizer):
    result = summarizer(input_text, max_length=45, min_length=15, do_sample=False)
    return result[0]["summary_text"]

def main():
    # ---------- 1. Build and Deploy the App ----------
    print("Loading summarization pipeline...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_fn(text):
        return summarize_text(text, summarizer)

    demo = gr.Interface(
        fn=summarize_fn,
        inputs=gr.Textbox(lines=8, label="Enter text to summarize"),
        outputs=gr.Textbox(label="Generated Summary"),
        title="GenAI Text Summarizer",
        description="A cloud-deployable Generative AI summarization app built with Gradio."
    )

    print("Launching Gradio web application interface...")
    # demo.launch(share=True) # Uncomment for public cloud link

    # ---------- 2. Evaluate Generated Output ----------
    print("\nEvaluating generated summaries using ROUGE metric...")
    rouge = evaluate.load("rouge")

    generated_summaries = [
        "AI models generate new content such as text and images."
    ]
    reference_summaries = [
        "Generative AI models are capable of producing new content including text and images."
    ]

    scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    print("ROUGE Evaluation Scores:", scores)

if __name__ == "__main__":
    main()
