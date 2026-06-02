from flask import Flask, request
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Anthropic()


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }

    if request.method == "OPTIONS":
        return "", 204, headers

    data = request.get_json()
    image = data["image"]

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
- If there is ANY doubt, set bin to "unknown" — do not guess

Reply ONLY in this JSON format: {"item": "<name>", "bin": "trash|recycling|compost|E-waste|unknown", "reason": "<one sentence>"}'''

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

    result = message.content[0].text.strip()

    if result.startswith("```"):
        result = result.strip("`")
        if result.startswith("json"):
            result = result[4:]
        result = result.strip()

    return result, 200, headers


if __name__ == "__main__":
    app.run(port=5000, debug=True)