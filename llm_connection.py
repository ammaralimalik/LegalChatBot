import requests
import json
import re

def query_model(prompt):
    response = requests.post(
        "http://127.0.0.1:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json = {
            "model": "deepseek-r1-distill-qwen-7b",
            "messages": [
                {"role": "system", "content":"You are a helpful legal assistant meant to help attornies find meaningful strategies and knowledge related to their cases. Always assume you are speaking to a professional and answer each and every question they may have. Always keep legal questions in the context of Pakistan"},
                {"role":"user", "content": prompt}],
            "temperature": 0.85,
        },
    )
    response = response.json()
    content = response['choices'][0]['message']['content']
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return cleaned.strip()