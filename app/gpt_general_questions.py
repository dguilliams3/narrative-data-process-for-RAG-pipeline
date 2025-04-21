from flask import Flask, request, Response, jsonify, session, render_template_string
from GPTClient import *
from gpt_config import *
from logging_utils import *
import os
import sys
from gpt_config import *
from sanitize_output import sanitize_text
from functools import wraps

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Initialize Flask
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session to work

# Initialize logging
LOG_FILE_PATH = "./logs/context_log.log"
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logging.info("Logging initialized")

def check_auth(username, password):
    return username == LOG_USERNAME and password == LOG_PASSWORD

def authenticate():
    return Response(
        "Authentication required", 401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Initialize GPTClient
gpt_client = GPTClient()
gpt_client.role = f"Here is your role for this conversation:\n{ROLE_ANSWER}\n[END ROLE DESCRIPTION]\n"

# Global flag; note: this global is separate from session "rag_status"
rag_status = True

# Replace FAISS_INDEX_PATH with FAISS_FILE_NAME
FAISS_FILE_NAME = "faiss_index.bin"
FAISS_INDEX_PATH = os.path.join(current_dir, FAISS_FILE_NAME)

if not os.path.exists(FAISS_INDEX_PATH):
    print("No FAISS index found, please check for file!")
    raise FileNotFoundError("No FAISS index found, please check for file!")
else:
    print("Loading existing FAISS index...")

@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_data = {
        "current_context_token_count": gpt_client.count_tokens(gpt_client.context),
        "current_context_length": len(gpt_client.context)
    }
    return jsonify(metrics_data), 200

@app.route("/logs", methods=["GET"])
@requires_auth
def view_logs():
    log_path = LOG_FILE_PATH
    if not os.path.exists(log_path):
        return "Log file not found.", 404
    with open(log_path, "r", encoding="utf-8") as f:
        return "<pre>" + f.read() + "</pre>"

@app.route("/ask", methods=["POST"])
def ask():
    if "rag_status" not in session:
        session["rag_status"] = False  # Initialize RAG status in session

    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"response": "Error: No input received."}), 400

    logging.info(f"Received request with user_input: {user_input}")

    if user_input.lower() == "reset":
        gpt_client.clear_context()
        return jsonify({"response": "Conversation context cleared. How can I assist you?"})
    elif user_input.lower().strip() == "context":
        full_context = f"{gpt_client.role}\n\n{gpt_client.context}"
        return jsonify({"response": full_context})
    elif user_input.lower() == "rag":
        session["rag_status"] = not session["rag_status"]  # Toggle RAG status
        return jsonify({"response": f"Toggling RAG to {session['rag_status']}"})

    gpt_client.prompt = user_input

    # Use the session value for rag_status in case it differs from the global flag
    if session.get("rag_status", False):
        gpt_client.update_context_with_rag(user_input)

    # Step 1: Send prompt and get the full response object
    response = gpt_client.send_prompt()

    # Step 2: Log raw response object BEFORE extraction
    logging.info(f"\n\n[RAW OPENAI RESPONSE OBJECT]:\n{repr(response)}\n\n")

    # Step 3: Extract message content
    response_text = gpt_client.extract_text(response)
    if not response_text:
        response_text = "[Error: No content returned]"

    # Step 4: Log extracted text before sanitization
    logging.info(f"\n\n[EXTRACTED TEXT PRE-SANITIZATION]:\n{repr(response_text)}\n\n")

    # Step 5: Sanitize the text
    try:
        response_text = sanitize_text(response_text)
    except Exception as e:
        logging.exception("Error during sanitize_text")
        response_text = "[Error: sanitize_text failed]"

    # Step 6: Final log
    logging.info(f"\n\n[FINAL SANITIZED TEXT]:\n{repr(response_text)}\n\n")
    logging.info(f"\n{'-'*40}[CONTEXT]{'-'*40}\n")
    logging.info(f"\n{gpt_client.context}\n")
    logging.info(f"\n{'-'*38}[END CONTEXT]{'-'*38}\n")
    logging.info(f"\n\nRETURNED RESPONSE:\n\n{response_text}")

    # Step 7: Return to frontend
    return jsonify({"response": response_text})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    html = render_template_string("""
    <!DOCTYPE html>
    <html>
      <head>
        <style>
          body {
            background-color: #121212;
            color: #e0e0e0;
            font-family: Arial, sans-serif;
            padding: 20px;
          }
          h1, h2 {
            color: #ffffff;
          }
          input[type="text"] {
            background-color: #1e1e1e;
            color: #f0f0f0;
            border: 1px solid #333;
            padding: 10px;
            border-radius: 5px;
          }
          button {
            background-color: #333;
            color: #fff;
            border: none;
            padding: 10px 15px;
            margin-left: 10px;
            border-radius: 5px;
            cursor: pointer;
          }
          button:hover {
            background-color: #444;
          }
          pre {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #333;
          }
          em {
            color: #b0b0ff;
          }
          a {
            color: #999;
          }
          footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.9em;
            color: #888;
          }
        </style>
        <title>GPT Chat Interface</title>
        <script>
          async function askQuestion() {
            console.log("askQuestion triggered");

            const userInputElem = document.getElementById("userInput");
            const userInput = userInputElem.value;
            const responseElem = document.getElementById("response");

            // Clear previous result
            responseElem.innerText = "";

            // Step 1: Show initial loading message
            responseElem.innerText = "Retrieving relevant lore and documents...";

            // Step 2: Simulate pipeline phases with time-based updates
            const phaseTimers = [
              { delay: 1200, text: "Analyzing content with fine-tuned GPT" },
              { delay: 3600, text: "Analyzing content with fine-tuned GPT." },
              { delay: 5000, text: "Analyzing content with fine-tuned GPT.." },
              { delay: 7400, text: "Analyzing content with fine-tuned GPT..." },
              { delay: 9800, text: "Analyzing content with fine-tuned GPT...." },
              { delay: 12200, text: "Analyzing content with fine-tuned GPT....." },
              { delay: 15000, text: "Generating final response from synthesized context..." }
            ];

            let active = true;
            let timeouts = [];
            phaseTimers.forEach(({ delay, text }) => {
              const id = setTimeout(() => {
                if (active) {
                  responseElem.innerText = text;
                  console.log("Updated loading message to:", text);
                }
              }, delay);
              timeouts.push(id);
            });

            try {
              const response = await fetch("/ask", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json"
                },
                body: JSON.stringify({ user_input: userInput })
              });

              if (response.ok) {
                const data = await response.json();
                active = false;
                timeouts.forEach(timeoutId => clearTimeout(timeoutId));
                responseElem.innerText = "Response ready:\\n\\n" + data.response;
              } else {
                active = false;
                timeouts.forEach(timeoutId => clearTimeout(timeoutId));
                console.error("Fetch failed:", response.status, response.statusText);
                responseElem.innerText = "Error: " + response.status;
              }
            } catch (err) {
              active = false;
              timeouts.forEach(timeoutId => clearTimeout(timeoutId));
              console.error("Error in askQuestion:", err);
              responseElem.innerText = "Error: " + err.message;
            }
          }

          window.addEventListener("load", function () {
            const inputElem = document.getElementById("userInput");
            inputElem.addEventListener("keydown", function (event) {
              if (event.key === "Enter") {
                event.preventDefault();
                askQuestion();
              }
            });
          });
        </script>
      </head>
      <body>
        <h1>Ask GPT about Soleria</h1>
        <input id="userInput" type="text" placeholder="Enter your question" style="width: 80%;">
        <button onclick="askQuestion()">Ask</button>
        <p style="font-size: 0.9em; color: #aaa; margin-top: 10px;">
          You’re speaking to a narrator embedded in the world of <strong>Soleria</strong>. Try prompts that:<br>
          <em>1. Analyze as literature: “What are the major themes of the story? How do the protagonists and larger story reflect those themes?”</em><br>
          <em>2. Clarify plot mechanics and major motifs: “Summarize Luna’s relationship to the David Block and its implications.”</em><br>
          <em>3. Explore the world: “What is the setting of the narrative/world? What major powers are involved in the political landscape?”</em><br>
          <em>4. Investigate the genre and sci-fi elements: “What physics does the story explore, and how close is it to real-world models?”</em><br>
          <em>5. Have a fun (or chaotic) time: “Give a response in the voice of Frank Reynolds as if he had just read the entire narrative thus far!”</em><br>
          The system handles high-context narrative and speculative science. Be as specific as you like.
        </p>
        <h2>Response:</h2>
        <pre id="response" style="white-space: pre-wrap; word-wrap: break-word;"></pre>
        <footer style="margin-top: 50px; text-align: center; font-size: 0.9em; color: #888;">
          Built by Dan Guilliams | <a href="https://github.com/dguilliams3" target="_blank" style="color: #888;">GitHub</a>
        </footer>
      </body>
    </html>
    """)
    logging.info("\n=== START RAW HTML ===\n")
    logging.info(repr(html))
    logging.info("\n=== END RAW HTML ===\n")
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
