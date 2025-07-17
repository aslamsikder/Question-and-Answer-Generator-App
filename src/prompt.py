prompt_template = """ 
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

----------------
{text}
----------------

Create questions that will prepare the coders or programers for their tests.
Make sure not to lose any important information.

QUESTION:
"""

refined_template = ("""
You are an expert at creating practice questions based on coding material and documentation.
Your goal is to help coder or programmer prepare for a coding test.
We have received some practice questions to a certain extent: {existing_answer}.

--------------
{text}
--------------

Given the new context, refining the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
""")