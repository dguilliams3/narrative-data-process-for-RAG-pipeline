from openai import OpenAI
import tiktoken
from gpt_config import *
from logging_utils import *
import os
import test_elasticsearch_and_faiss_query_line

class GPTClient:
	def __init__(
				self, 
				api_key=None, 
				model=GPT_MODEL, 
				max_tokens=GPT_MAX_TOKENS, 
				prompt = "Hey I'm ready to roll!", 
				role="You are a sophisticated LLM that assumes the user has a default bassic knowledge about all topics and treat them as a peer in a conversational and colloquial manner", 
				context="", 
				max_context_tokens=GPT_MAX_CONTEXT_TOKENS, 
				context_file_names=None
		):
		self._api_key = api_key if api_key else os.getenv('OPENAI_API_KEY')
		self._model = model
		self._max_tokens = max_tokens
		self._max_context_tokens = max_context_tokens
		self._prompt = prompt
		self._context = role + context
		self.client = OpenAI(api_key=self._api_key)
		self.last_response = None  # Initialize last response attribute
		self._role = role
		self._context_file_names = context_file_names if context_file_names else []  # Initialize an empty list to store filenames

		if not self._api_key:
			logging.error("API Key not found. Ensure it's set in the environment variables.")
			raise ValueError("API Key not found. Ensure it's set in the environment variables.")

	@property
	def api_key(self):
		return self._api_key

	@api_key.setter
	def api_key(self, value):
		self._api_key = value
		self.client = OpenAI(api_key=self._api_key)  # Update the client if the API key changes
		logging.info("API Key updated.")

	@property
	def model(self):
		return self._model

	@model.setter
	def model(self, value=""):
		self._model = value
		logging.info(f"Model set to {value}")

	@property
	def role(self):
		return self._role

	@role.setter
	def role(self, value):
		self._role = value
		logging.info(f"Role set to {value}")

	@property
	def max_tokens(self):
		return self._max_tokens

	@max_tokens.setter
	def max_tokens(self, value):
		if value > 0:
			self._max_tokens = value
		else:
			raise ValueError("Max tokens must be a positive integer")

	@property
	def max_context_tokens(self):
		return self._max_context_tokens

	@max_context_tokens.setter
	def max_context_tokens(self, value):
		if value > 0:
			self._max_context_tokens = value
		else:
			raise ValueError("Max context tokens must be a positive integer")

	@property
	def prompt(self):
		return self._prompt

	@prompt.setter
	def prompt(self, value):
		if isinstance(value, str):
			self._prompt = value
		else:
			raise ValueError("Prompt must be a string")

	@property
	def context(self):
		return self._context

	@context.setter
	def context(self, value):
		if isinstance(value, str):
			self._context = value
		else:
			raise ValueError("Context must be a string")

	@property
	def context_file_names(self):
		return self._context_file_names

	@context_file_names.setter
	def context_file_names(self, value):
		if isinstance(value, list):  # Correct type check
			self._context_file_names = value
		else:
			raise ValueError("context_file_names must be a list")


	def send_prompt(self, prompt=None):
		"""Send a prompt to the GPT model. Use the currently assigned prompt if none is provided."""
		if prompt is None:
			prompt = self._prompt  # Use the instance's prompt if none provided
			self.prompt = prompt

		full_prompt = f"""
  {self.role}\n
  Here is the background context of the current chat and relevant topics:\n{self._context}\n[END CONTEXT]\n
  And here is the prompt:\n{prompt}"""  # Context first, then the prompt
  
		response = self.client.chat.completions.create(
			model=self._model,
			messages=[{"role": "user", "content": full_prompt}],
			max_tokens=self._max_tokens,
			temperature=0.7,
		)
		logging.info(f"\n\n{'-'*80}\nSENT:\n\n{full_prompt}{'-'*80}\n\n")
		self.last_response = response  # Store the last response
		self.update_context_after_response(prompt, response.choices[0].message.content) # Update the context
		return response

	def update_context_after_response(self, user_input, response_text, max_context_tokens=GPT_MAX_CONTEXT_TOKENS):
		self.context += f"\nUser: {user_input}\nGPT: {response_text}"
		current_tokens = self.count_tokens(self.context)
		logging.info(f"Current tokens: {current_tokens}")
		# Check if we have hit the max allocated tokens before resummarizing the context
		if current_tokens > max_context_tokens:
			self.context = self.summarize_context(self.context, max_context_tokens)
   
	def send_prompt_for_RAG(self, new_content, user_prompt=None):
		"""Send a prompt to the GPT model. Use the currently assigned prompt if none is provided."""
		if user_prompt is None:
			user_prompt = self.prompt  # Use the instance's prompt if none provided

		full_prompt = f"""
	{self.role}\n
	Here is the context of the chat:\n{self.context}\n[END CHAT CONTEXT]\n
	Here is what was retrieved from the RAG implementation: {new_content}\n\n [END RAG-RETRIEVED CONTENT] \n\n 
	Here is the user's current prompt:\n{user_prompt}\n[END CURRENT PROMPT]\n
	Please return just a summary of the RAG-RETRIEVED CONTENT, a robust and comprehensive one, of what you think relevant to the chat given the context. Remember that some of the RAG results may be out of scope, but also note the chronology and relevant characters and tags if they relate to the prompt and context.
	Note that this will be added to the context, so don't add in anything about what the prompt and context are saying are currently happening - just give a relevant summary from what was RAG-RETRIEVED.
	"""
  
		response = self.client.chat.completions.create(
			model=self._model,
			messages=[{"role": "user", "content": full_prompt}],
			max_tokens=self._max_tokens,
			temperature=0.7,
		)

		return self.extract_text(response)

	def update_context_with_rag(self, query):
		# Query via FAISS and then the ES instance for context from the prompt
		faiss_results = test_elasticsearch_and_faiss_query_line.query_faiss(query)
		document_data = test_elasticsearch_and_faiss_query_line.retrieve_documents_from_elasticsearch(faiss_results, query)
		new_content = ""
		# Initialize GPTClient
		rag_check_object = GPTClient(role="""
			A user is interacting with GPT via API calls, and they have implemented a RAG system to retrieve context on top of the existing chat history.  
			Please look at the returned summaries from the RAG query and return the most relevant and robust summary that contains what you take to be relevant to the current converation above it and the user's current prompt.
			Be sure to make an effort to prioritize giving details about characters that aren't currently in context as summaries of who they are and what their roles in general are followed by then talking about their dynamics in the current context.
			""",
			max_tokens=5000,
			context = self.context,
			model = "ft:gpt-4o-2024-08-06:personal:soleria:ARq2SEpI"
			)
		# Iterate through the returned filenames and summaries and add them to the context
		for docinfo in document_data:
			filename = docinfo["filename"]
			doc = docinfo["summary"]
			# if filename in self.context_file_names:
			# 	logging.info(f"Context for filename ({filename}) already included, skipping...")
			# 	continue
			# Construct a detailed string of all key-value pairs in the document
			print("DEBUG: doc =", doc, "Type:", type(doc), flush=True)
			if isinstance(doc, str):
				print("DEBUG: doc is a string—Expected a dictionary!")  # Debug message
				doc = {"text": doc}  # Wrap it in a dictionary to prevent errors

			doc_details = "\n".join([f"{key}: {value}" for key, value in doc.items() if key != 'filename'])  # Exclude the filename from details
			next_line = f"\nRelated Info from File: {filename}\nInfo from file:\n{doc_details}"
			logging.info(next_line)

			new_content_summary = rag_check_object.send_prompt_for_RAG(doc, query)
			logging.info(f"\n\nSummarized version: {new_content_summary}")
			
			new_content += new_content_summary
			self.context_file_names.append(filename)
			# Summarize iteratively if we hit the context threshold while adding the returned summaries
			if self.count_tokens(self.context) + self.count_tokens(new_content) > self.max_context_tokens:
				logging.info("Summarizing context...")
				self.context = self.summarize_context(self.context)
		
		logging.info(f"\n\n{'-'*20}\nSummary of retrieved documents from GPT:\n\n\n{new_content}\n{'-'*20}\n\n")
		self.context += new_content
		return new_content

	def extract_text(self, response):
		"""Extract the text from the response assuming it has attribute access."""
		try:
			# Access the content directly via attributes
			return response.choices[0].message.content
		except (AttributeError, IndexError) as e:
			# Log the error or handle it appropriately
			logging.error(f"Error extracting text from response: {str(e)}")
			return None  # Return None or handle the error as needed

	def count_tokens(self, text):
		# Load the tokenizer for the specific model
		encoding = tiktoken.encoding_for_model(self.model)
		# Tokenize the input text
		tokens = encoding.encode(text)
		# Return the number of tokens
		return len(tokens)

	def summarize_context(self, max_tokens=None):
		# Implement context summarization logic here
			logging.info("Context exceeds maximum token threshold. Summarizing...")
			context_prompt = f"""
				Given the following conversation context, provide a comprehensive summary that encapsulates the key themes, insights, conclusions, 
				and any pivotal nuances necessary for sustaining context-aware dialogue continuation. Ensure the summary serves as a robust replacement for the entire text, 
				maintaining coherence and relevance for subsequent interactions: {self.context}"""
			try:
				logging.info("Context exceeds maximum token threshold. Summarizing...")
				context_prompt = f"Given the following conversation context, provide a comprehensive summary: {self._context}"
				response = self.client.chat.completions.create(
					model=self._model,
					messages=[{"role": "user", "content": context_prompt}],
					max_tokens=max_tokens if max_tokens else self.max_context_tokens,
					temperature=0.7,
				)
				context_summary = self.extract_text(response)

				if context_summary:
					self.context = context_summary  # Reset context to the summarized content
					self.context_file_names = [] # Reset the file names being counted as in the summary
					logging.info(f"{'-'*40}\n\nSummarized Context:\n{context_summary}\n{'-'*40}\n\n")
				else:
					logging.error("Failed to summarize context or received an empty summary.")
			except Exception as e:
				logging.error(f"Error during context summarization: {str(e)}")
				# Handle failed summarization: perhaps revert to a default prompt or clear context
				self.context = "Failed to summarize context; starting fresh."

	def clear_context(self):
		self.context = ""
  
	def store_retrieved_documents(self, filename):
		if filename not in self.context_file_names:
			self.context_file_names.append(filename)
