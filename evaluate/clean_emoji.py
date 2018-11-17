import re
import unicodedata
from unidecode import unidecode

def deEmojify(inputString):
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            #replaced = unidecode(str(character))
            returnString += ''
            # if replaced != '':
            #     returnString += replaced
            # else:
            #     try:
            #          returnString += "[" + unicodedata.name(character) + "]"
            #     except ValueError:
            #          returnString += "[x]"

    return returnString

msg = 'Youre so sweet. 😂😭❤️ Love you.'
mm = 'Follow and support my big siste business; Source Afrique🇬🇭🇬🇭🇬🇭🇬🇭on Instagram and Facebook! https://t.co/eLJtwPskLN'
print(msg)
print(deEmojify(msg).lower())


