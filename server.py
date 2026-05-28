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

    # build the prompt with step-by-step thinking
    prompt = '''Analyze this image step-by-step to classify the waste item.

Step 1: What is the object? Describe its material, shape, condition, and any visible labels.

Step 2: Is the object clean or contaminated with food/liquid?

Step 3: What is the primary material (plastic type, metal, glass, paper, food, electronic, mixed)?

Step 4: Based on common municipal waste rules, which bin does it belong in?

Bins:
- TRASH: styrofoam, plastic bags, plastic wrap, dirty/contaminated paper, ceramics, broken glass, mixed-material packaging
- RECYCLING: CLEAN paper, CLEAN cardboard, aluminum cans, steel cans, glass bottles/jars, rigid plastic bottles/containers (#1, #2, #5)
- COMPOST: food scraps, fruit/vegetable peels, coffee grounds, eggshells, yard waste, food-soiled uncoated paper
- E-WASTE: batteries, electronics, phones, cables, light bulbs, paint, chemicals, aerosol cans

Rules:
- Be CONSERVATIVE — when unsure between recycling and trash, choose trash to avoid contaminating recycling streams
- Styrofoam is ALWAYS trash, never recycling
- Plastic bags are ALWAYS trash, never recycling
- Dirty pizza boxes go in compost, clean ones in recycling
- You must be 100% certain of both what the item is AND which bin it belongs in
- If there is ANY doubt, set bin to "unknown" — do not guess'''

    if rules and rules != "[]":
        prompt += f'\n\nIMPORTANT: The user has corrected past mistakes. Use these as strong guidance:\n{rules}'

    prompt += '\n\nReply ONLY in this JSON format: {"item": "<name>", "bin": "trash|recycling|compost|E-waste|unknown", "reason": "<one sentence> explaining why it goes in that bin"}'

    # ask Claude what bin it goes in
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
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