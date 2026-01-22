def normalize(text):
    """
    lower case
    """
    if text is None:
        return ""
    return text.lower().strip()

def is_correct(pred, gt):
    """
    match
    """
    return normalize(pred) == normalize(gt)
