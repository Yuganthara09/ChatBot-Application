from flask import Flask, render_template, request, jsonify
import chatbot  # Import the chatbot module

app = Flask(__name__)

# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle chatbot responses
@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    if user_message:
        try:
            bot_response = chatbot.generate_response(user_message)
            
            # Fallback check for irrelevant responses
            if not bot_response or len(bot_response.split()) < 3:
                bot_response = "I'm here to assist! Could you clarify your question?"

            return jsonify({"response": bot_response})
        except Exception as e:
            return jsonify({"response": f"Error generating response: {e}"})
    return jsonify({"response": "Please type a message to get started."})

if __name__ == "__main__":
    app.run(debug=True)
