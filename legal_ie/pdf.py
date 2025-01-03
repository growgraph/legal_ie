import re


def chop_footer(text):
    """
    Removes a specific footer pattern from the end of the given text.

    The footer pattern consists of:
    1. A "Page x / X" line.
    2. A line starting with "Pourvoi".
    3. A line with a date in the format "day month year".

    Args:
        text (str): The input text.

    Returns:
        str: The text with the footer removed.
    """
    # Define the regex pattern for the footer
    footer_pattern = re.compile(
        r"Page \d+ / \d+\nPourvoi .*?\n\d{1,2} [a-zA-Zéû]+ \d{4}$", re.MULTILINE
    )

    footer_pattern2 = re.compile(r"Page \d+ / \d+\n", re.MULTILINE)
    text0 = re.sub(footer_pattern, "", text)
    text0 = re.sub(footer_pattern2, "", text0)

    return text0.strip()
