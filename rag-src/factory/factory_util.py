import importlib
import logging

logger = logging.getLogger(__name__)

def module_and_name_str_2_module_and_name_tuple(module_and_name_str: str) -> tuple[str, str]:
    """
    Extracz the module name and class or function name of a fully qualified name

    Args:
        fully qualified name: module name dot class of function name, e.g. "my.module.MyClass"

    Returns:
        (str, str): module name, class name, e.g. ("my.module", "MyClass")
    """
    result = module_and_name_str.rsplit('.', 1)
    return (result[0], result[1])



def call_function_or_constructor(
    module_and_name_str: str,
    kwargs: dict,
    context_str_for_logging: str
) -> any:
    """
    Call a function or constructor by name with kwargs
  
    Args:
    
        module_and_name_str (str): fully qualified name of the function or constructor, e.g. "my.module.MyClass"
        kwargs (dict): keyword arguments for the function or constructor
        context_str_for_logging (str): context string for logging, e.g. "LLM setup"

    Returns:
        any: result of the function or constructor
    """

    try:

        # Extract module and class/function name
        (module_name, function_or_constructor_name) = module_and_name_str_2_module_and_name_tuple(module_and_name_str)

        # Call the function or calls the cinstructor/instantiate the class (dynamic instantiation)
        module = importlib.import_module(module_name)
        function_or_constructor = getattr(module, function_or_constructor_name)
        result = function_or_constructor(**kwargs)
        # Incomplete(!!!) alternative would to call a constructor would be:
        #   instance = globals()[class_name](**class_kwargs)
        #   instance = langchain_openai.ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

        return result

    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Error in {context_str_for_logging}: {e}", e)
        else:
            logger.error(f"Error in {context_str_for_logging}: {e}")
        return None
