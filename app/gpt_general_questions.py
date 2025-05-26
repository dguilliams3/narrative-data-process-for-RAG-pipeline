import os
import sys
import logging
import time
from flask import Flask, request, Response, jsonify, session, render_template_string
from flask_swagger_ui import get_swaggerui_blueprint
from functools import wraps

# ensure current dir is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from GPTClient import GPTClient
from gpt_config import *
from logging_utils import configure_logging, ensure_log_dir
from sanitize_output import sanitize_text
from chunk_manager import ChunkManager

# --- Logging setup ---
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logger = logging.getLogger(__name__)
logger.info("Starting Flask app")

# --- Flask/app-level settings ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-flask-key")

def check_auth(u, p):
    return u == LOG_USERNAME and p == LOG_PASSWORD

def authenticate():
    return Response("Auth required", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def wrapper(*a, **k):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*a, **k)
    return wrapper

# instantiate client
client = GPTClient(gpt_service_url=os.getenv("GPT_SERVICE_URL"))
client.role = f"System Role:\n{ROLE_ANSWER}[END ROLE]\n"

# RAG toggle default
DEFAULT_RAG = True

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handle user questions and return GPT responses.

    This endpoint processes user input, optionally performs RAG retrieval,
    and returns a response from the GPT service.

    Request:
        JSON body with:
        - user_input (str): The user's question or prompt

    Response:
        JSON object containing:
        - response (str): The sanitized GPT response text
        - duration (float): Total request duration in seconds
        - model (str, optional): The model used for generation
        - usage (dict, optional): Token usage statistics
            - prompt_tokens (int): Tokens used in the prompt
            - completion_tokens (int): Tokens used in the response
            - total_tokens (int): Total tokens used
        - finish_reason (str, optional): Why the model stopped generating
        - duration (float, optional): How long the GPT service took

    Special Commands:
        - "reset": Clears the conversation context
        - "context": Shows the current conversation context
        - "rag": Toggles RAG functionality on/off

    Returns:
        JSON response with status code 200 for success, 400 for invalid input
    """
    data = request.json or {}
    user_input = data.get("user_input", "").strip()
    if not user_input:
        return jsonify({"response": "No input received"}), 400

    # RAG toggle in session
    if "rag_status" not in session:
        session["rag_status"] = DEFAULT_RAG

    # special commands
    if user_input.lower() == "reset":
        client.chunker = ChunkManager(client.model, client.max_context_tokens, client)  # reset chunker
        return jsonify({"response": "Context cleared."})
    if user_input.lower() == "context":
        return jsonify({"response": f"CURRENT CONTEXT:\n\n{client.role}\n{client.chunker.get_context()}"})
    if user_input.lower() == "rag":
        session["rag_status"] = not session["rag_status"]
        return jsonify({"response": f"RAG now {'ON' if session['rag_status'] else 'OFF'}"})
    
    request_start = time.time()
    # maybe inject RAG
    if session.get("rag_status"):
        client.update_context_with_rag(user_input)
    # send to GPT
    resp = client.send_prompt(user_input)
    
    # If the total response is called, extract the text, if not, just return the text.
    # This can help with modularity latter if we begin introducing toggles or options.
    if isinstance(resp, str):
        text = resp
    else:
        text = client.extract_text(resp)
    
    safe = sanitize_text(text)
    logger.info("Final answer: %s", safe)
    duration = time.time() - request_start
    logger.info("Request duration: %.2fs", duration)

    # Include metadata in response if available
    response_data = {
        "response": safe,
        "duration": duration
    }
    if hasattr(client, 'last_response_metadata'):
        response_data.update(client.last_response_metadata)
    
    return jsonify(response_data)

@app.route("/logs", methods=["GET"])
@requires_auth
def view_logs():
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        return "<pre>" + f.read() + "</pre>"

@app.route("/health")
def health():
    return "OK", 200

from flask import jsonify

@app.route("/metrics", methods=["GET"])
def metrics():
    # grab the current conversation context from the chunk manager
    ctx = client.chunker.get_context()
    return jsonify({
        # how many tokens in that context
        "current_context_token_count": client.count_tokens(ctx),
        # and how long the raw string is
        "current_context_length": len(ctx)
    }), 200

@app.route("/")
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
          You're speaking to a narrator embedded in the world of <strong>Soleria</strong>. Try prompts that:<br>
          <em>1. Analyze as literature: "What are the major themes of the story? How do the protagonists and larger story reflect those themes?"</em><br>
          <em>2. Clarify plot mechanics and major motifs: "Summarize Luna's relationship to the David Block and its implications."</em><br>
          <em>3. Explore the world: "What is the setting of the narrative/world? What major powers are involved in the political landscape?"</em><br>
          <em>4. Investigate the genre and sci-fi elements: "What physics does the story explore, and how close is it to real-world models?"</em><br>
          <em>5. Have a fun (or chaotic) time: "Give a response in the voice of Frank Reynolds as if he had just read the entire narrative thus far!"</em><br>
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
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)), debug=True)
