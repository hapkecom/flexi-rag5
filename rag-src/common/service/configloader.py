"""
Settings are loaded from YAML files:
- default-config.yaml
- config.yaml

These settings can be overwritten by environment variables:
Examples:
  export RAG_FOO=BAR                      # string value
                                          # with RAG_ prefix

  export RAG_NESTED__LEVEL__KEY=1         # Double underlines
                                          # denotes nested settings
                                          # nested = {
                                          #     "level": {"key": 1}
                                          # }
(More details on this: https://www.dynaconf.com/#on-env-vars )
                                          
The internal implementation is based on the Dynaconf library,
but special Dynaconf features/syntax are not allowed to be used
because the internal implementation can change in the future.

Hydra/omegaconf is not used internally
because it does not support overwriting by environment variables.
"""

from functools import cache
import json
from typing import Dict, List
import yaml
import os

import logging

logger = logging.getLogger(__name__)

from dynaconf import Dynaconf

settings = Dynaconf(
    #root_path="/data/aisbreaker-workspace/hapkecom-github/flexi-rag/",
    settings_files=[
        "rag-default-config.yaml",
        "rag-config.yaml",
        #".secrets.yaml",
    ],
    merge_enabled=True,
    #environments=True,
    #env_switcher="MYAPP_MODE",         # `export MYAPP_MODE=production`
    envvar_prefix="RAG"
)

print(f"Loading configuration ... with Dynaconf settings: {str(settings.as_dict())}")

# magic value
_no_default = "_NO_DEFAULT_VALUE_PLACEHOLDER_sdknfcw9845cvnhkjrvzh6"

#
# Relevant functions
#

def deep_get(config: Dict, keys_str: str, default_value: any = _no_default) -> any:
    """
    Get a value from a nested dictionary by a dot-separated key string.

    String values with ${var.*} variables and with ${env.*} environment variables are replaced,
    also in results with dictionaries and nested values.

    If no default value is given and the key is not found, a KeyError is raised.
    """

    return deep_get_with_vars(config, keys_str, get_all_env_and_global_vars(), default_value)

def deep_get_with_vars(config: Dict[str,any], keys_str: str, vars: List[Dict[str,str]], default_value: any = _no_default) -> any:
    """
    Get a value from a nested dictionary by a dot-separated key string.

    String values with ${*} variables of the provided vars are replaced,
    also in results with dictionaries and nested values.

    If no default value is given and the key is not found, a KeyError is raised.
    """

    # get thesul value(s)
    raw_result = deep_get_raw(config, keys_str, default_value)

    # replace variables
    if isinstance(raw_result, str):
        r = replaceVarsInString(raw_result, vars)
        return r
    elif isinstance(raw_result, dict):
        return replaceVarsInDict(raw_result, vars)
    else:
        return raw_result

def deep_get_raw(config: Dict[str,any], keys_str: str, default_value: any = _no_default) -> any:
    """
    Get a value from a nested dictionary by a dot-separated key string.

    If no default value is given and the key is not found, a KeyError is raised.

    No variable replacement is done.
    """
    # pre-check
    if keys_str is None or keys_str == "":
        return config

    # actual work
    keys = keys_str.split(".")
    configOrResult = config
    used_keys = ""
    for key in keys:
        # try to access value
        used_keys += key
        configOrResult = configOrResult.get(key)
        if configOrResult is None:
            # key not found
            if default_value == _no_default:
                raise KeyError(f"Key '{used_keys}' (of '{keys_str}') not found in dictionary")
            else:
                return default_value
        # key found

        # prepare next loop
        used_keys += "."

    # done
    return configOrResult

@cache
def get_all_env_and_global_vars() -> Dict[str,str]:
    """
    Load all environment and global variables into a dictionary.
    """
    return {**load_all_env_vars(), **load_all_gobal_vars()}

def get_all_vars(config: Dict[str,any], keys_str: str, key_prefix_to_add: str) -> Dict[str,str]:
    """
    Load all variables and return them as a (variable) dictionary together with all env and global vars.
    """
    local_vars: Dict[str,str] = load_all_config_vars_from_a_dict(config, keys_str, key_prefix_to_add)
    return {**local_vars, **get_all_env_and_global_vars()}

#
# Helper functions
#


def load_all_env_vars() -> Dict[str,str]:
    """
    Load all environment variables into a dictionary.
    """
    return {(f"env.{key}"): value for key, value in os.environ.items()}

def load_all_gobal_vars() -> Dict[str,str]:
    """
    Load all global variables into a dictionary.
    """
    return load_all_config_vars_from_a_dict(settings, "vars", "var")

def load_all_config_vars_from_a_dict(config: Dict[str,any], keys_str: str, key_prefix_to_add: str) -> Dict[str,str]:
    # Load all config variables from a dictionary
    vars = deep_get_raw(config, keys_str, {})
    result = {}

    # Add prefix to keys
    for key, value in vars.items():
        if isinstance(value, str):
            result[f"{key_prefix_to_add}.{key}"] = value

    return result



def replaceVarsInString(str: str, vars: Dict[str,str]) -> str:
    """
    Replace all provided variables (e.g. "var.foo") in a string (if it contains e.g. "${var.foo}").
    """
    for key, value in vars.items():
        str = str.replace("${"+key+"}", value)
    return str

def replaceVarsInDict(aDict: Dict, vars: Dict[str,str]) -> Dict:
    """
    Replace all provided variables (e.g. "var.foo") in a dictionary (if it contains e.g. "${var.foo}").
    """
    for key, value in aDict.items():
        if isinstance(value, str):
            aDict[key] = replaceVarsInString(value, vars)
        elif isinstance(value, dict):
            aDict[key] = replaceVarsInDict(value, vars)
    return aDict



# Accessing a setting . trigger loading now to see config errors here
# Hint: info-logging is disabled here:
logger.info(f"config.rag_loading.enabled={settings.config.rag_loading.enabled}")

#print(f"config.common.databases.vectorstore.args.persist_directory: {deep_get(settings, 'config.common.databases.vectorstore.args.persist_directory')}")

def config_str(config, indent=0) -> str:
    """Create a formatted multiline-string of the configuration dictionary with proper indentation."""

    result = ""
    if config is not None and hasattr(config, "items") is True:
        for key, value in config.items():
            linestart = "\n" + ("  " * indent)
            if isinstance(value, dict):
                result += linestart + f"{key}:"
                result += config_str(value, indent + 1)
            else:
                if "pass" in key.lower() or "token" in key.lower():
                    result += linestart + f"{key}: {'*' * len(str(value))}"
                else:
                    result += linestart + f"{key}: {value}"
    return result

logger.info("## Final Configuration:"+config_str(settings.as_dict()))

def main():
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(filename)s:%(funcName)s() - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    # load the default configuration
    default_config = load_yaml_file('default-values.yaml')
    logger.debug("default_config:"+str(default_config))

    # load the specific configuration
    values_config = load_yaml_file('values.yaml')
    
    # merge the configurations
    merged_config = merge_configs(default_config, values_config)
    logger.debug("merged_config:"+str(merged_config))

    # Overwrite with environment variables
    final_config = overwrite_with_env_variables(merged_config)
    
    # use the final configuration
    logger.info("## Final Configuration:"+config_str(final_config))

if __name__ == "__main__":
    main()
