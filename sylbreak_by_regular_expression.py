import re

"""
Reference Code https://github.com/ye-kyaw-thu/myWord
"""


def create_break_pattern():
    """Creates and returns the regular expression pattern for Myanmar syllable breaking."""
    my_consonant = r"က-အ"
    en_char = r"a-zA-Z0-9"
    other_char = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@[-`{-~\s"
    subscript_symbol = r'္'
    a_that = r'်'

    # Regular expression pattern for Myanmar syllable breaking
    return re.compile(
        r"((?<!" + subscript_symbol + r")[" + my_consonant + r"]"
                                                             r"(?![" + a_that + subscript_symbol + r"])"
                                                                                                   r"|[" + en_char + other_char + r"])"
    )


def break_syllables(line, break_pattern, separator):
    """Applies syllable breaking rules to a line."""
    line = re.sub(r'\s+', ' ', line.strip())
    segmented_line = break_pattern.sub(separator + r"\1", line)

    # Remove the leading delimiter if it exists
    if segmented_line.startswith(separator):
        segmented_line = segmented_line[len(separator):]

    # Replace delimiter+space+delimiter with a single space
    double_delimiter = separator + " " + separator
    segmented_line = segmented_line.replace(double_delimiter, " ")

    return segmented_line


def process_input(sentence_str, separator=", "):
    sentence_removed_space = sentence_str.replace(" ", "")
    break_pattern = create_break_pattern()
    segmented_line = break_syllables(sentence_removed_space, break_pattern, separator)
    print("\nSylbreaked:\n" + segmented_line)


# process_input(
#     "တစ်နိုင်ငံလုံးပို့ဆောင်ခအသ က်သာဆုံး နှင့် အရွယ်အစား(၂ပေပတ်လည်အောက်) ပစ္စည်းများ ပစ္စည်းရောက်ငွေချေနိုင်သော(CODစနစ်) ဖြင့် 3 ရက်မှ 10 ရက်အတွင်း သင့်အိမ်ရှေ့အရောက် မှာယူတဲ့ပစ္စည်းလေး ရောက်ရှိလာပါမယ်။",
#     '\n')
# process_input("နေ့ရောညရောမြန်မာစာနဲ့ကွန်ပျူတာနဲ့ပဲအလုပ်များပါတယ်", " - ")
