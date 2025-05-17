from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents import tool


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word/Count of characters."""
    return len(word)

# # @tool
# def pythonRepl():
#     """
#     Returns a Tool instance that allows executing Python code in a REPL environment.
#     """
#     python_repl = PythonREPL()
#     return Tool(
#         name="python_repl",
#         description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#         func=python_repl.run,
#     )

@tool
def python_repl(query: str) -> str:
    """
    Execute Python code and return the result. 
    Input should be valid Python code. 
    Returns either the output of the code or an error message. 
    In case code execution is successful but produces no output, a message is returned indicating that. 
    If the code execution fails, an error message is returned.
    """
    repl = PythonREPL()
    try:
        result = repl.run(query)
        if not result or result.strip() == "":
            return "Code executed successfully, but produced no output. If you want to see values, use print() statements in your code."
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"