import collections
import re

class Indexer():
    """
    base class for various text taggers

    takes in text; main data structure is a list of tuples
           [tag, start, end]

    where: tag is any data type
           start and end are integers representing the start and end indices in the string
    """

    def __init__(self):
        pass

    def tag(self, text):
        pass


class WordTagger(Indexer):
    """
    simple regular expression word tokenizer
    """

    def tag(self, text):
        self.tags = self.get_words(text)

    def get_words(self, text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer("([\.\,\;']|[a-z0-9]+)", text, re.IGNORECASE) if m.group() not in ['and', ',']]


class NumberTagger(WordTagger):

    def __init__(self):
        self.load_numberwords()
        Indexer.__init__(self)

    def load_numberwords(self):
        self.numberwords = {
            # 'a':      1,
            'one':      1,
            'two':      2,
            'three':    3,
            'four':     4,
            'five':     5,
            'six':      6,
            'seven':    7,
            'eight':    8,
            'nine':     9,
            'ten':      10,
            'eleven':   11,
            'twelve':   12,
            'thirteen': 13,
            'fourteen': 14,
            'fifteen':  15,
            'sixteen':  16,
            'seventeen':17,
            'eighteen': 18,
            'nineteen': 19,
            'twenty':   20,
            'thirty':   30,
            'forty':    40,
            'fifty':    50,
            'sixty':    60,
            'seventy':  70,
            'eighty':   80,
            'ninety':   90,
            'hundred':  100,
            'thousand': 1000,
            'million': 1000000,
            'billion': 1000000000,
            'trillion': 1000000000000
        }

    def swap(self, text):
        """
        returns string with number words replaced with digits
        """
        text = re.sub(r"(\D[0-9]{1,3})[\s\,]([0-9]{3}\D)", r"\1\2", text)
        tags = self.tag(text)
        # tags.sort(key=lambda (number, start, end): start) # get tags and sort by start index
        tags.sort(key=lambda indices: indices[1])

        output_list = []
        progress_index = 0

        for (number, start_index, end_index) in tags:
            output_list.append(text[progress_index:start_index]) # add the unedited string from the last marker up to the number
            output_list.append(str(number)) # add the string digits of the number
            progress_index = end_index # skip the marker forward to the end of the original number words

        output_list.append(text[progress_index:]) # if no tags, this will append the whole unchanged string

        return ''.join(output_list)


    def tag(self, text):
        """
        produces a list of tuples (number, start_index, end_index)
        """
        words = self.get_words(text)
        words.reverse()

        number_parts = []
        number_parts_index = -1

        last_word_was_a_number = False

        # first get groups of consecutive numbers from the reversed word list



        for word, start, end in words:

            word_num = self.numberwords.get(word.lower())

            if word_num is None:
                last_word_was_a_number = False
            else:
                if last_word_was_a_number == False:
                    number_parts.append([])
                    number_parts_index += 1
                last_word_was_a_number = True

                number_parts[number_parts_index].append((word_num, start, end))

        output = []


        # then calculate the number for each part

        for number_part in number_parts:
            number = self.recursive_nums([word_num for word_num, start, end in number_part])
            start = min([start for word_num, start, end in number_part])
            end = max([end for word_num, start, end in number_part])

            output.append((number, start, end))
        return(output)

    def recursive_nums(self, numlist):

        # first split list up

        tens_index = 0
        tens = [100, 1000, 1000000, 1000000000, 1000000000000]

        current_multiplier = 1

        split_list = collections.defaultdict(list)

        for num in numlist:
            if num in tens[tens_index:]:
                tens_index = tens.index(num)+1
                current_multiplier = num
            else:
                split_list[current_multiplier].append(num)

        counter = 0

        # then sum up the component parts

        for multiplier, numbers in split_list.items():
            # check if multiples of ten left

            for number in numbers:
                if number in tens:
                    counter += multiplier * self.recursive_nums(numbers)
                    break
            else:
                counter += multiplier * sum(numbers)

        return counter




