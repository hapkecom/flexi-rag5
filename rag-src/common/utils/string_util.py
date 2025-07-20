from typing import (
    List,
    Optional,
)
import logging

logger = logging.getLogger(__name__)

#
# String utility functions for text processing, mainily for logging.
#

def str_limit(s: any, max_len: int = 40) -> str:
    if s is None:
        return None
    s0 = str(s)

    # remove newlines
    s1 = s0.replace('\n', ' ').replace('\r', ' ')

    # remove multiple spaces
    s1 = ' '.join(s1.split())

    # limit the len
    s = s1 if len(s1) <= max_len else (s1.strip()[:max_len] + (('...['+str(len(s0))+']') if len(s1) > max_len else ''))

    return s

def str_limit_hard_cut(s: any, max_len: int = 40) -> str:
    if s is None:
        return None
    s = str(s)

    # remove newlines
    s = s.replace('\n', ' ').replace('\r', ' ')

    # remove multiple spaces
    s = ' '.join(s.split())

    # limit the len
    s = s if len(s) <= max_len else (s.strip()[:max_len])
    
    return s




#
# String helper functions: overlap detection and merging
#

def merge_strings_with_with_overlap_detection_and_tail_recursion(
        merged: str,
        remaining: List[str],
        separator_in_case_of_simple_concatenation: str = "\n\n...\n\n"
        ) -> str:
    """
    Merge strings with overlap detection and tail recursion.

    Overlap handling is used to compensate overlapping from document splitting.
    
    Args:
        merged (str): The merged string so far. Start with an empty string.
        strings (List[str]): List of (remaining) strings to merge.
        separator_in_case_of_simple_concatenation (str):
            Separator to use if no overlap is detected. Defaults to "\n\n...\n\n".
    
    Returns:
        str: Merged string with overlaps handled.
    """
    if not remaining:
        return merged

    # Take the first string from the remaining list
    current_string = remaining[0]
    remaining_strings = remaining[1:]

    # Check for overlap with the merged string
    merged = merge_two_strings_with_with_overlap_detection(merged, current_string, separator_in_case_of_simple_concatenation)
    if not remaining_strings:
        # If there are no remaining strings, return the merged result
        return merged
    else:
        # Recursively merge the remaining strings
        return merge_strings_with_with_overlap_detection_and_tail_recursion(merged, remaining_strings, separator_in_case_of_simple_concatenation)


def merge_two_strings_with_with_overlap_detection(
        s1: str,
        s2: str, 
        separator_in_case_of_simple_concatenation: str = "\n\n...\n\n"
        ) -> str:
    """
    Merge two strings with overlap detection.

    Overlap handling is used to compensate overlapping from document splitting.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        separator_in_case_of_simple_concatenation (str):
            Separator to use if no overlap is detected. Defaults to "\n\n...\n\n".
    
    Returns:
        str: Merged string with overlaps handled.
    """
    extra_info_logging = False or logger.isEnabledFor(logging.DEBUG)
    if extra_info_logging:
        logger.info(f"Merging len = {len(s1)+len(s2)} started ...")

    # Check for overlap with the first string in a range of half the length of the second string to 1
    max_allowed_overlap_len = 10*1000
    min_overlap_len = 10
    overlap_length = min(min(len(s1), len(s2)), max_allowed_overlap_len)
    for i in range(overlap_length, min_overlap_len, -1):
        if s1.endswith(s2[:i]):
            # Overlap detected, merge without overlap
            if extra_info_logging:
                logger.info( f"  Overlap detected of len: {i}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Overlap detected: '{s1[-i:]}'")
                logger.debug(f"    Overlap detected in s1: '{s1}'")
                logger.debug(f"    Overlap detected in s2: '{s2}'")

            return s1 + s2[i:]

    # No overlap, just simple concatenation
    if extra_info_logging:
        logger.info("  No overlap detected, simple concatenation")
    return s1 + separator_in_case_of_simple_concatenation + s2
