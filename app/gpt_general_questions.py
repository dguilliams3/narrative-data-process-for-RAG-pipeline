from flask import Flask, request, jsonify, session, render_template_string
from GPTClient import *
from gpt_config import *
from logging_utils import *
import os
import sys
from gpt_config import *
from sanitize_output import sanitize_text

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
gpt_client = GPTClient()
gpt_client.role = f"Here is your role for this conversation:\n{ROLE_ANSWER}\n[END ROLE DESCRIPTION]\n"

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
  elif user_input.lower().strip() == "context":
    full_context = f"{gpt_client.role}\n\n{gpt_client.context}"
    return jsonify({"response": full_context})
  elif user_input.lower() == "rag":
    session["rag_status"] = not session["rag_status"]  # Toggle RAG status
    return jsonify({"response": f"Toggling RAG to {session['rag_status']}"})

  # gpt_client.role = role_answer
  gpt_client.prompt = user_input
  
  if rag_status:
    gpt_client.update_context_with_rag(user_input)

  response = gpt_client.send_prompt()
  response_text = gpt_client.extract_text(response)
  response_text = sanitize_text(response_text)
  
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
          role_answer: {{ role_answer|tojson }}
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
      
      // Bind ENTER key on the user input to call askQuestion
      window.addEventListener("load", function() {
      const inputElem = document.getElementById("userInput");
      inputElem.addEventListener("keydown", function(event) {
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
      <em>    1. Analyze as literature: “What are the major themes of the story? How do the protagonists and larger story reflect those themes?”</em><br> 
      <em>    2. Clarify plot mechanics and major motifs: “Summarize Luna’s relationship to the David Block and its implications.”</em><br>
      <em>    3. Explore the world: “What is the setting of the narrative/world? What major powers are involved in the political landscape?”</em><br>
      <em>    4. Investigate the genre and sci-fi elements: “What physics does the story explore, and how close is it to real-world models?”</em><br>
      <em>    5. Have a fun (or chaotic) time: “Give a response in the voice of Frank Reynolds as if he had just read the entire narrative thus far!”</em><br> 
      The system handles high-context narrative and speculative science. Be as specific as you like.
    </p>
    <h2>Response:</h2>
    <pre id="response" style="white-space: pre-wrap; word-wrap: break-word;"></pre>
    <footer style="margin-top: 50px; text-align: center; font-size: 0.9em; color: #888;">
      Built by Dan Guilliams | <a href="https://github.com/dguilliams3" target="_blank" style="color: #888;">GitHub</a>
    </footer>
    </body>
  </html>

  """, role_answer=ROLE_ANSWER)
  return html

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)