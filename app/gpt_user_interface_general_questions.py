import requests
from gpt_config import *
from logging_utils import *

def conversation_interaction(role_answer="You are an advanced conversationalist..."):
	url = "http://127.0.0.1:5000/ask"
	client_session = requests.Session()  # Create a session to persist cookies

	print("Type 'exit' to end the conversation.")
	print(f"Using role: {role_answer[:500]}...[redacted]")

	while True:
		# Get user input
		user_input = input("\n\n---------------------------------------\n\nYou: ")
		
		# Exit the loop if the user types 'exit'
		if user_input.lower() == 'exit':
			print("Conversation ended.")
			break

		# Send the input and custom role to the Flask server
		data = {
			'user_input': user_input,
			'role_answer': role_answer  # Include the role in the request data
		}
		
		try:
			response = client_session.post(url, json=data)
			response.raise_for_status()  # Check if the request was successful
			print(f"\n\nGPT: {response.json()['response']}\n\n---------------------------------------\n\n")
		except requests.exceptions.HTTPError as http_err:
			print(f"HTTP error occurred: {http_err}")
		except requests.exceptions.RequestException as err:
			print(f"Error occurred: {err}")
		except ValueError:
			print("Failed to parse JSON. Please check the server response.")

if __name__ == "__main__":
	# Example of using a custom ROLE_ANSWER
	conversation_interaction(ROLE_ANSWER if ROLE_ANSWER else "You are an advanced conversationalist")