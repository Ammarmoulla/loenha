arabic_characters = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
diacritics_list = ['َ', 'ً', 'ِ', 'ٍ', 'ُ', 'ٌ', 'ْ', 'ّ']

classes = {'': 0, 'َ': 1, 'ً': 2, 'ُ': 3, 'ٌ': 4, 'ِ': 5, 'ٍ': 6, 'ْ': 7, 'ّ': 8, 'َّ': 9, 'ًّ': 10, 
           'ُّ': 11, 'ٌّ': 12, 'ِّ': 13, 'ٍّ': 14, '<PAD>': 15, '<SOS>': 16, '<EOS>': 17, '<N/A>': 18}

revers_classes = {0: '', 1: 'َ', 2: 'ً', 3: 'ُ', 4: 'ٌ', 5: 'ِ', 6: 'ٍ', 7: 'ْ', 8: 'ّ', 9: 'َّ', 10: 'ًّ',
                 11: 'ُّ', 12: 'ٌّ', 13: 'ِّ', 14: 'ٍّ', 15: '<PAD>', 16: '<SOS>', 17: '<EOS>', 18: '<N/A>'}

symbol = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, '\n': 4, ' ': 5, '!': 6, '"': 7, "'": 8, '(': 9, ')': 10,
           '*': 11, ',': 12, '-': 13, '.': 14, '/': 15, '0': 16, '1': 17, '2': 18, '3': 19, '4': 20,
           '5': 21, '6': 22, '7': 23, '8': 24, '9': 25, ':': 26, ';': 27, '[': 28, ']': 29, '`': 30,
           '{': 31, '}': 32, '~': 33, '«': 34, '»': 35, '،': 36, '؛': 37, '؟': 38, 'ء': 39, 'آ': 40,
           'أ': 41, 'ؤ': 42, 'إ': 43, 'ئ': 44, 'ا': 45, 'ب': 46, 'ة': 47, 'ت': 48, 'ث': 49, 'ج': 50,
           'ح': 51, 'خ': 52, 'د': 53, 'ذ': 54, 'ر': 55, 'ز': 56, 'س': 57, 'ش': 58, 'ص': 59, 'ض': 60,
           'ط': 61, 'ظ': 62, 'ع': 63, 'غ': 64, 'ف': 65, 'ق': 66, 'ك': 67, 'ل': 68, 'م': 69, 'ن': 70,
           'ه': 71, 'و': 72, 'ى': 73, 'ي': 74, '–': 75, '\u200f': 76}

# small_symbol = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, '\n': 4, ' ': 5, '!': 6, '"': 7, '&': 8, "'": 9, '(': 10,
#       ')': 11, '*': 12, '+': 13, ',': 14, '-': 15, '.': 16, '/': 17, '0': 18, '1': 19, '2': 20,
#       '3': 21, '4': 22, '5': 23, '6': 24, '7': 25, '8': 26, '9': 27, ':': 28, ';': 29, '=': 30,
#       '[': 31, ']': 32, '_': 33, '`': 34, '{': 35, '}': 36, '~': 37, '«': 38, '»': 39, '،': 40,
#       '؛': 41, '؟': 42, 'ء': 43, 'آ': 44, 'أ': 45, 'ؤ': 46, 'إ': 47, 'ئ': 48, 'ا': 49, 'ب': 50,
#       'ة': 51, 'ت': 52, 'ث': 53, 'ج': 54, 'ح': 55, 'خ': 56, 'د': 57, 'ذ': 58, 'ر': 59, 'ز': 60,
#       'س': 61, 'ش': 62, 'ص': 63, 'ض': 64, 'ط': 65, 'ظ': 66, 'ع': 67, 'غ': 68, 'ف': 69, 'ق': 70,
#       'ك': 71, 'ل': 72, 'م': 73, 'ن': 74, 'ه': 75, 'و': 76, 'ى': 77, 'ي': 78, '٠': 79, '١': 80,
#       '٢': 81, '٤': 82, '\u200d': 83, '\u200f': 84, '–': 85, '’': 86, '“': 87, '…': 88, '﴾': 89, '﴿': 90}