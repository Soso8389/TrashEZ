from flask import Flask, request
from anthropic import Anthropic
from dotenv import load_dotenv

# load my API key from the .env file
load_dotenv()

app = Flask(__name__)
client = Anthropic()


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():

    # headers so the browser lets us talk to this server
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }

    # browser sends a check first before the real request
    if request.method == "OPTIONS":
        return "", 204, headers

    # get the picture from the webpage
    image = request.get_json()["image"]

    # ask Claude what bin it goes in
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image
                    }
                },
                {
                    "type": "text",
                    "text": 'Which bin does this go in? Bins: trash, recycling, compost, E-waste. You must be 100% certain of both what the item is AND which bin it belongs in. If there is ANY doubt — the image is unclear, the item is ambiguous, or you are not completely sure of the correct bin — you MUST set bin to "unknown". Do not guess. Only classify if you are absolutely certain. Reply ONLY in this JSON format: {"item": "<name>", "bin": "trash|recycling|compost|E-waste|unknown", "reason": "<one sentence> explaining why it goes in that bin"}'
                }
            ]
        }]
    )

    # clean up Claude's answer
    result = message.content[0].text.strip()

    # sometimes Claude wraps the answer in ``` so we remove that
    if result.startswith("```"):
        result = result.strip("`")
        if result.startswith("json"):
            result = result[4:]
        result = result.strip()

    return result, 200, headers


# start the server
if __name__ == "__main__":
    app.run(port=5000, debug=True)