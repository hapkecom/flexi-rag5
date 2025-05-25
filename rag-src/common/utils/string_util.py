

def str_limit(s: any, max_len: int = 40) -> str:
    if s is None:
        return None
    s = str(s)

    # remove newlines
    s = s.replace('\n', ' ').replace('\r', ' ')

    # remove multiple spaces
    s = ' '.join(s.split())

    # limit the len
    s = s if len(s) <= max_len else (s.strip()[:max_len] + (('...['+str(len(s))+']') if len(s) > max_len else ''))
    
    return s
