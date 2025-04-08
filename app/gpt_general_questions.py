from flask import Flask, request, jsonify, session, render_template_string
from GPTClient import *
from gpt_config import *
from logging_utils import *
import os
import sys
from gpt_config import *

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Initialize Flask
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session to work

# Initialize logging
LOG_FILE_PATH = "./context_log.log"
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logging.info("Logging initialized")

# Initialize GPTClient
gpt_client = GPTClient(role=f"Here is your role for this conversation:\n{ROLE_ANSWER}\n[END ROLE DESCRIPTION]\n")
rag_status = True

# Replace FAISS_INDEX_PATH with FAISS_FILE_NAME
FAISS_FILE_NAME = "faiss_index.bin"
FAISS_INDEX_PATH = os.path.join(current_dir, FAISS_FILE_NAME)

if not os.path.exists(FAISS_INDEX_PATH):
		print("No FAISS index found, please check for file!")
		raise FileNotFoundError("No FAISS index found, please check for file!")
else:
	print("Loading existing FAISS index...")

@app.route("/ask", methods=["POST"])
def ask():
	if "rag_status" not in session:
		session["rag_status"] = False  # Initialize RAG status in session

	user_input = request.json.get("user_input")

	try:
		role_answer = request.json.get('role_answer') # Use custom role if provided
		gpt_client.role = role_answer
	except:
		role_answer = gpt_client.role

	logging.info(f"Received request with user_input: {user_input}")

	if user_input.lower() == "reset":
		gpt_client.clear_context()
		return jsonify({"response": "Conversation context cleared. How can I assist you?"})
	elif user_input.lower() == "context":
		return jsonify({"response": gpt_client.context})
	elif user_input.lower() == "rag":
		session["rag_status"] = not session["rag_status"]  # Toggle RAG status
		return jsonify({"response": f"Toggling RAG to {session['rag_status']}"})

	# gpt_client.role = role_answer
	gpt_client.prompt = user_input
	
	if rag_status:
		gpt_client.update_context_with_rag(user_input)

	response = gpt_client.send_prompt()
	response_text = gpt_client.extract_text(response)
	# Log the current context
	logging.info(f"\n{'-'*40}[CONTEXT]{'-'*40}\n")
	logging.info(f"\n{gpt_client.context}\n")
	logging.info(f"\n{'-'*38}[END CONTEXT]{'-'*38}\n")
	logging.info(f"\n\nRETURNED RESPONSE:\n\n{response_text}")
	return jsonify({"response": response_text})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    # A simple HTML page that sends a question to /ask and displays the answer.
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>GPT Chat Interface</title>
        <script>
          async function askQuestion() {
            console.log("askQuestion triggered");

            const userInput = document.getElementById("userInput").value;
            console.log("User input:", userInput);

            // Show a loading message immediately
            document.getElementById("response").innerText = "Loading...";

            try {
              const response = await fetch("/ask", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json"
                },
                body: JSON.stringify({
                  user_input: userInput,
                  role_answer: "{ROLE_ANSWER}" // Optional custom role (e.g., "Here is your role for this conversation: ...")
                })
              });

              console.log("Fetch response status:", response.status);

              if (response.ok) {
                const data = await response.json();
                console.log("Server data:", data);
                document.getElementById("response").innerText = data.response;
              } else {
                console.error("Fetch failed:", response.status, response.statusText);
                document.getElementById("response").innerText = "Error: " + response.status;
              }

            } catch (err) {
              console.error("Error in askQuestion:", err);
              document.getElementById("response").innerText = "Error: " + err.message;
            }
          }
        </script>
      </head>
      <body>
        <h1>Ask GPT about Soleria</h1>
        <input id="userInput" type="text" placeholder="Enter your question" style="width: 80%;">
        <button onclick="askQuestion()">Ask</button>
        <h2>Response:</h2>
        <pre id="response" style="white-space: pre-wrap; word-wrap: break-word;"></pre>
      </body>
    </html>

    """
    return render_template_string(html, role_answer=ROLE_ANSWER)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)