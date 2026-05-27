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

    # get the picture and the user's saved rules from the webpage
    data = request.get_json()
    image = data["image"]
    rules = data.get("rules", "[]")

    # build the prompt, adding the user's past corrections if any
    prompt = 'Which bin does this go in? Bins: trash, recycling, compost, E-waste. Take your time to carefully analyze the item. Look closely at the material — identify whether it is plastic, metal, glass, paper, cardboard, styrofoam, organic waste, etc. Be especially careful: styrofoam is NOT recyclable and should go in trash. Make your best guess about what the item is and which bin it belongs in, but maintain a high confidence threshold — aim for 100% certainty. If you are not sufficiently confident in your classification, return "unknown" instead of guessing. Only classify if you are fairly sure.'

    if rules and rules != "[]":
        prompt += f'\n\nIMPORTANT: The user has corrected past mistakes. Use these corrections as strong guidance:\n{rules}'

    prompt += '\n\nReply ONLY in this JSON format: {"item": "<name>", "bin": "trash|recycling|compost|E-waste|unknown", "reason": "<one sentence> explaining why it goes in that bin"}'

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
                    "text": prompt
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