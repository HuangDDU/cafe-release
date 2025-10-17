import datetime
import random
import string

from .._logging import logger


def random_time_string(name=None):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # formated time
    random_chars = "".join(random.choices(string.ascii_letters + string.digits, k=10))  # random string
    # combine them
    if name:
        time_string = f"{current_time}__{name}__{random_chars}"
    else:
        time_string = f"{current_time}__{random_chars}"

    return time_string


def parse_random_time_string(str):
    if "__" in str:
        parse_str_list = str.split("__")
        if len(parse_str_list) == 3:
            return parse_str_list[1]
        else:
            return ""
    else:
        logger.debug(f"'{str}' is not a valid random_time_string, don't need parse")
        return str
