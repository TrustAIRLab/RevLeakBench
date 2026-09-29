ANSWER_SUFFIX = " (**Answer Length:** Keep answers concise and to the point, ideally within 1-5 words.)"


ANSWER_JSON = '\n\nReturn only a JSON object of the form {"reasoning": string, "answer": string}.'


EQ_TEMPLATE = 'Do "{value}" and "{answer}" have the same meaning? score from 0 to 1. 0 means no, 1 means yes.'


EQ_JSON = '\n\nReturn only a JSON object of the form {"reasoning": string, "score": number}.'

INSTRUCTION = """You are a professional content analyst who checks the truthfulness of propositions based on a given text.
Read the text provided below and evaluate each proposition for its truthfulness based on the content of the text.

**[Requirements]**
1.  **Truthfulness:** For each proposition, determine if it is true or false based solely on the information in the given text. Never use external knowledge or make guesses.
2.  **Format:** Organize your evaluation in the JSON format, with each proposition paired with a reasoning and a boolean indicating its truthfulness.

**[Truthfulness Evaluation Criteria]**

True:
* When the information is explicitly stated in the text
* When it is not directly mentioned but can be necessarily inferred from the context of the text

False:
* When the information contradicts the text
* When the information is not supported by the text at all and would require external knowledge to judge

Now, please start working with the text below."""


JSON_NOTE = '\n\nReturn only a JSON object of the form {"evaluations": [{"proposition": string, "reasoning": string, "is_true": boolean}, ...]} with one entry per proposition, in the given order.'
