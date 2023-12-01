import editdistance
# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text == "":
            return 0.0
        return 1.0
    target_text_splitted = target_text.split()
    predicted_text_splitted = predicted_text.split()
    return editdistance.eval(target_text_splitted, predicted_text_splitted) / len(target_text_splitted)
